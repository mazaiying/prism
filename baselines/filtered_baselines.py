#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive GPU & 48-Thread C++ CPU Filtered ANNS Baselines for PRISM Evaluation.

Baselines Implemented:
  - B1  : GPU-IVF-PreFilter (Exact FP32 resident on GPU 0)
  - B2a : cuVS-IVF-Flat (Filter)  [NVIDIA cuVS 26.02 native `from_bitset`]
  - B2b : cuVS-CAGRA (Filter)     [NVIDIA cuVS 26.02 flagship GPU graph `from_bitset`]
  - B3  : Faiss-GPU IVF-PostFilter [Adaptive Overfetch up to k_over=2048]
  - B4  : VecFlow-Style Label-GPU  [SIGMOD'26 VecFlow dual specificity label/bucket GPU index]
  - B5  : ACORN-gamma (48T CPU)    [SIGMOD'24 2-hop predicate-aware graph index in C++17/OpenMP]
  - B6  : HNSW-Filter (48T CPU)    [48-thread C++17/OpenMP filtered proximity graph]
  - B7  : Host-Offload PreFilter   [Out-of-Core GPU SLAP filter + UVA PCIe exact gather, no sketch]
"""

from __future__ import annotations
import hashlib
import importlib.util
import os
import subprocess
import sysconfig
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils import cpp_extension
import faiss

from src.hybrid_datasets import (
    HybridDataset,
    PredicateBatch,
    PRED_COL0_RANGE,
    PRED_COL1_RANGE,
    PRED_COL2_CAT,
    PRED_ALG_EXPR,
    get_paths,
)
from src.prism_engine import PrismEngine

_CPU_EXT = None


def get_usable_vram_gib() -> float:
    free_b, tot_b = torch.cuda.mem_get_info(0)
    return min(tot_b / (1024 ** 3) - 0.55, free_b / (1024 ** 3) - 0.25)


def load_cpu_graph_ext():
    global _CPU_EXT
    if _CPU_EXT is not None:
        return _CPU_EXT

    src_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_path = os.path.join(src_dir, "cpu_graph_ext.cpp")
    with open(cpp_path, "rb") as f:
        sig = hashlib.sha256(f.read()).hexdigest()[:12]

    prism_root = os.path.expanduser(
        os.environ.get("PRISM_ROOT", os.path.abspath(os.path.join(src_dir, "..")))
    )
    build_dir = os.path.join(prism_root, "build_ext")
    os.makedirs(build_dir, exist_ok=True)
    mod_name = f"cpu_graph_omp_{sig}"
    so_path = os.path.join(build_dir, f"{mod_name}.so")

    if not os.path.exists(so_path):
        py_inc = sysconfig.get_path("include")
        inc_flags = [f"-I{py_inc}"] + [f"-I{p}" for p in cpp_extension.include_paths()]
        lib_flags = [f"-L{p}" for p in cpp_extension.library_paths()]
        abi = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
        cmd = [
            "g++", "-O3", "-std=c++17", "-march=native", "-fopenmp",
            "-shared", "-fPIC",
            f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
            f"-DTORCH_EXTENSION_NAME={mod_name}",
            "-DTORCH_API_INCLUDE_EXTENSION_H",
            *inc_flags, cpp_path, *lib_flags,
            "-lc10", "-ltorch_cpu", "-ltorch", "-ltorch_python",
            "-o", so_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"g++ failed:\n{res.stderr}")

    spec = importlib.util.spec_from_file_location(mod_name, so_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    _CPU_EXT = mod
    return mod


def pack_slap_u64_numpy(ds: HybridDataset) -> np.ndarray:
    return (
        ds.col0.astype(np.uint64)
        | (ds.col1.astype(np.uint64) << np.uint64(16))
        | (ds.col3.astype(np.uint64) << np.uint64(32))
        | ((ds.col2.astype(np.uint64) & np.uint64(63)) << np.uint64(48))
    )


class FaissGPUPostFilterBaseline:
    """B3: Faiss-GPU IVF-Flat with Adaptive Overfetch (`k_over <= 2048`) + GPU Predicate Filter."""
    def __init__(self, nlist: int = 1024):
        self.nlist = nlist
        self.gpu_res = None
        self.gpu_ivf = None
        self.vram_gib = 0.0
        self.build_time_s = 0.0

    def build(self, ds: HybridDataset) -> Dict[str, float]:
        usable_gib = get_usable_vram_gib()
        est_gib = (ds.N * ds.D * 4 * 1.25 + ds.N * 16) / (1024 ** 3)
        if est_gib > usable_gib:
            raise MemoryError(f"Faiss-GPU IVF-Flat requires {est_gib:.2f} GiB > {usable_gib:.2f} GiB")

        t0 = time.perf_counter()
        self.ds = ds
        xb = np.asarray(ds.base_vectors, dtype=np.float32)
        quant = faiss.IndexFlatL2(ds.D)
        cpu_ivf = faiss.IndexIVFFlat(quant, ds.D, self.nlist, faiss.METRIC_L2)
        train_n = min(ds.N, max(65536, self.nlist * 40))
        cpu_ivf.train(xb[:: max(1, ds.N // train_n)][:train_n])
        cpu_ivf.add(xb)
        self.gpu_res = faiss.StandardGpuResources()
        self.gpu_ivf = faiss.index_cpu_to_gpu(self.gpu_res, 0, cpu_ivf)
        self.c0_g = torch.from_numpy(ds.col0.astype(np.int32)).cuda()
        self.c1_g = torch.from_numpy(ds.col1.astype(np.int32)).cuda()
        self.c2_g = torch.from_numpy(ds.col2.astype(np.int64)).cuda()
        self.c3_g = torch.from_numpy(ds.col3.astype(np.int32)).cuda()
        torch.cuda.synchronize()
        self.build_time_s = time.perf_counter() - t0
        self.vram_gib = est_gib
        return {"total_build_s": self.build_time_s, "total_gpu_vram_gib": self.vram_gib}

    @torch.no_grad()
    def search(self, queries: np.ndarray, preds: PredicateBatch, k: int = 10, nprobe: int = 32) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        B = queries.shape[0]
        self.gpu_ivf.nprobe = int(nprobe)
        min_sel = float(np.clip(preds.target_sel[:B].min(), 1e-4, 1.0))
        k_over = min(2048, max(k * 8, int(np.ceil(k / min_sel * 2.5))))

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        d_raw, i_raw = self.gpu_ivf.search(queries, k_over)

        # Vectorized GPU post-filtering of (B, k_over) candidate IDs
        ids_g = torch.from_numpy(i_raw).cuda()
        dists_g = torch.from_numpy(d_raw).cuda()
        valid = ids_g >= 0
        safe_ids = ids_g.clamp(min=0)

        c0 = self.c0_g[safe_ids]
        c1 = self.c1_g[safe_ids]
        c2 = self.c2_g[safe_ids]
        c3 = self.c3_g[safe_ids]

        op = torch.from_numpy(preds.op_flags[:B].astype(np.int32)).cuda().unsqueeze(1)
        l0 = torch.from_numpy(preds.l0[:B]).cuda().unsqueeze(1)
        r0 = torch.from_numpy(preds.r0[:B]).cuda().unsqueeze(1)
        l1 = torch.from_numpy(preds.l1[:B]).cuda().unsqueeze(1)
        r1 = torch.from_numpy(preds.r1[:B]).cuda().unsqueeze(1)
        cm = torch.from_numpy(preds.cat_mask[:B].view(np.int64)).cuda().unsqueeze(1)
        w0 = torch.from_numpy(preds.w0[:B]).cuda().unsqueeze(1)
        w3 = torch.from_numpy(preds.w3[:B]).cuda().unsqueeze(1)
        ab = torch.from_numpy(preds.alg_bound[:B]).cuda().unsqueeze(1)

        ok = valid.clone()
        ok &= ((op & PRED_COL0_RANGE) == 0) | ((c0 >= l0) & (c0 <= r0))
        ok &= ((op & PRED_COL1_RANGE) == 0) | ((c1 >= l1) & (c1 <= r1))
        bit_c2 = torch.bitwise_left_shift(torch.ones_like(c2), c2)
        ok &= ((op & PRED_COL2_CAT) == 0) | (torch.bitwise_and(bit_c2, cm) != 0)
        ok &= ((op & PRED_ALG_EXPR) == 0) | ((w0 * c0 + w3 * c3) <= ab)

        dists_masked = torch.where(ok, dists_g, torch.full_like(dists_g, float("inf")))
        top_d, loc = torch.topk(dists_masked, k=k, dim=1, largest=False, sorted=True)
        top_i = torch.gather(ids_g, 1, loc)
        top_i = torch.where(torch.isfinite(top_d), top_i, torch.full_like(top_i, -1))

        out_i = top_i.cpu().numpy()
        out_d = top_d.cpu().numpy()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        return out_i, out_d, {"latency_ms": ms, "qps": (B * 1000.0) / max(ms, 1e-6)}


class VecFlowStyleLabelGPUBaseline:
    """
    B4: VecFlow-Style Dual-Specificity Label/Bucket Partitioned GPU Index (SIGMOD'26).

    Replicates/partitions data by discrete label `col2` (64 categories) x `G_bkt=4` static
    `col1` quantile buckets (256 cells) + global fallback index for unindexed `col0`-only
    queries (mirroring specialized per-attribute index overhead in multi-column tables).
    High-specificity cells (>= 4096 vectors) use per-cell IVF centroids; low-specificity
    cells (< 4096 vectors) use coalesced flat GPU tensors, followed by exact GPU residual
    predicate filtering for continuous range boundaries and algebraic expressions.
    """
    def __init__(self, g_buckets: int = 4, spec_thresh: int = 4096):
        self.g_buckets = g_buckets
        self.spec_thresh = spec_thresh
        self.vram_gib = 0.0
        self.build_time_s = 0.0

    def build(self, ds: HybridDataset) -> Dict[str, float]:
        usable_gib = get_usable_vram_gib()
        # Dual index storage: 256 (cat, col1_bkt) cells + 16 col0 static range buckets + build workspace -> ~2.65x vector storage
        est_gib = (ds.N * ds.D * 4 * 2.65 + ds.N * 32 * 2) / (1024 ** 3)
        if est_gib > usable_gib:
            raise MemoryError(f"VecFlow-Style multi-attribute GPU index requires {est_gib:.2f} GiB > {usable_gib:.2f} GiB")

        t0 = time.perf_counter()
        self.ds = ds
        xb = np.asarray(ds.base_vectors, dtype=np.float32)

        # Primary 256-cell (col2 x col1_bkt) GPU layout + Secondary 16-bucket col0 GPU layout
        bkt1 = np.minimum(self.g_buckets - 1, (ds.col1.astype(np.int32) * self.g_buckets) // 65536)
        cell_id = ds.col2.astype(np.int32) * self.g_buckets + bkt1
        order1 = np.argsort(cell_id, kind="mergesort")
        counts1 = np.bincount(cell_id, minlength=64 * self.g_buckets)
        self.cell_off = np.zeros(64 * self.g_buckets + 1, dtype=np.int64)
        np.cumsum(counts1, out=self.cell_off[1:])

        self.xb_cell_g = torch.from_numpy(xb[order1]).cuda()
        self.ids_cell_g = torch.from_numpy(order1.astype(np.int64)).cuda()
        self.c0_cell_g = torch.from_numpy(ds.col0[order1].astype(np.int32)).cuda()
        self.c1_cell_g = torch.from_numpy(ds.col1[order1].astype(np.int32)).cuda()
        self.c3_cell_g = torch.from_numpy(ds.col3[order1].astype(np.int32)).cuda()

        # Secondary col0 16-bucket index (required to support arbitrary col0 predicates!)
        bkt0 = np.minimum(15, (ds.col0.astype(np.int32) * 16) // 65536)
        order0 = np.argsort(bkt0, kind="mergesort")
        counts0 = np.bincount(bkt0, minlength=16)
        self.bkt0_off = np.zeros(17, dtype=np.int64)
        np.cumsum(counts0, out=self.bkt0_off[1:])
        self.xb_b0_g = torch.from_numpy(xb[order0]).cuda()
        self.ids_b0_g = torch.from_numpy(order0.astype(np.int64)).cuda()
        self.c0_b0_g = torch.from_numpy(ds.col0[order0].astype(np.int32)).cuda()

        # Build coarse centroids per high-specificity cell
        self.cell_centroids: Dict[int, Tuple[torch.Tensor, np.ndarray]] = {}
        for cid in range(64 * self.g_buckets):
            s, e = int(self.cell_off[cid]), int(self.cell_off[cid + 1])
            cnt = e - s
            if cnt >= self.spec_thresh:
                nc = min(64, max(8, cnt // 512))
                sub_x = xb[order1[s:e]]
                step = max(1, cnt // (nc * 32))
                km = faiss.Kmeans(ds.D, nc, niter=10, verbose=False, gpu=True, seed=2027)
                km.train(np.array(sub_x[::step], dtype=np.float32, copy=True))
                cg = torch.from_numpy(km.centroids).cuda()
                # Reorder cell slice by sub-cluster
                d_sub = (cg * cg).sum(dim=1).unsqueeze(0) - 2.0 * (self.xb_cell_g[s:e] @ cg.T)
                asg = torch.argmin(d_sub, dim=1).cpu().numpy()
                sub_ord = np.argsort(asg, kind="mergesort")
                sub_cnt = np.bincount(asg, minlength=nc)
                sub_off = np.zeros(nc + 1, dtype=np.int64)
                np.cumsum(sub_cnt, out=sub_off[1:])
                sub_ord_g = torch.from_numpy(sub_ord).cuda()
                self.xb_cell_g[s:e] = self.xb_cell_g[s:e].index_select(0, sub_ord_g)
                self.ids_cell_g[s:e] = self.ids_cell_g[s:e].index_select(0, sub_ord_g)
                self.c0_cell_g[s:e] = self.c0_cell_g[s:e].index_select(0, sub_ord_g)
                self.c1_cell_g[s:e] = self.c1_cell_g[s:e].index_select(0, sub_ord_g)
                self.c3_cell_g[s:e] = self.c3_cell_g[s:e].index_select(0, sub_ord_g)
                self.cell_centroids[cid] = (cg, sub_off + s)

        torch.cuda.synchronize()
        self.build_time_s = time.perf_counter() - t0
        self.vram_gib = est_gib
        return {"total_build_s": self.build_time_s, "total_gpu_vram_gib": self.vram_gib}

    @torch.no_grad()
    def search(self, queries: np.ndarray, preds: PredicateBatch, k: int = 10, probe_frac: float = 0.35) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        B = queries.shape[0]
        q_g = torch.from_numpy(queries).cuda()
        qn2 = (q_g * q_g).sum(dim=1)
        out_i = np.full((B, k), -1, dtype=np.int64)
        out_d = np.full((B, k), np.inf, dtype=np.float32)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for qi in range(B):
            qv = q_g[qi : qi + 1]
            q_n2 = qn2[qi]
            op = int(preds.op_flags[qi])
            l0, r0 = int(preds.l0[qi]), int(preds.r0[qi])
            l1, r1 = int(preds.l1[qi]), int(preds.r1[qi])
            cm = int(preds.cat_mask[qi])
            w0, w3, ab = int(preds.w0[qi]), int(preds.w3[qi]), int(preds.alg_bound[qi])

            if op == PRED_COL0_RANGE:
                # Route to secondary 16-bucket col0 index
                b_lo = max(0, min(15, (l0 * 16) // 65536))
                b_hi = max(0, min(15, (r0 * 16) // 65536))
                s, e = int(self.bkt0_off[b_lo]), int(self.bkt0_off[b_hi + 1])
                if s < e:
                    c0_s = self.c0_b0_g[s:e]
                    ok = (c0_s >= l0) & (c0_s <= r0)
                    idx_ok = torch.nonzero(ok, as_tuple=False).squeeze(1)
                    if idx_ok.numel() > 0:
                        xs = self.xb_b0_g[s:e].index_select(0, idx_ok)
                        d2 = (q_n2 + (xs * xs).sum(dim=1) - 2.0 * (qv @ xs.T).squeeze(0)).clamp(min=0.0)
                        kc = min(k, int(idx_ok.numel()))
                        td, tl = torch.topk(d2, k=kc, largest=False, sorted=True)
                        out_d[qi, :kc] = td.cpu().numpy()
                        out_i[qi, :kc] = self.ids_b0_g[s:e].index_select(0, idx_ok)[tl].cpu().numpy()
                continue

            # Determine active (cat, bkt1) cells
            b1_lo = max(0, min(self.g_buckets - 1, (l1 * self.g_buckets) // 65536)) if (op & PRED_COL1_RANGE) else 0
            b1_hi = max(0, min(self.g_buckets - 1, (r1 * self.g_buckets) // 65536)) if (op & PRED_COL1_RANGE) else (self.g_buckets - 1)

            slices: List[Tuple[int, int]] = []
            for cat in range(64):
                if (op & PRED_COL2_CAT) and (((cm >> cat) & 1) == 0):
                    continue
                for b1 in range(b1_lo, b1_hi + 1):
                    cid = cat * self.g_buckets + b1
                    if cid in self.cell_centroids:
                        cg, sub_off = self.cell_centroids[cid]
                        nc = cg.shape[0]
                        np_c = max(2, int(np.ceil(nc * probe_frac)))
                        cd = (cg * cg).sum(dim=1) - 2.0 * (qv @ cg.T).squeeze(0)
                        _, tc = torch.topk(cd, k=min(np_c, nc), largest=False, sorted=False)
                        for sc_idx in tc.cpu().tolist():
                            slices.append((int(sub_off[sc_idx]), int(sub_off[sc_idx + 1])))
                    else:
                        s, e = int(self.cell_off[cid]), int(self.cell_off[cid + 1])
                        if s < e:
                            slices.append((s, e))

            if not slices:
                continue
            # Coalesce contiguous slices to minimize kernel launches
            slices.sort()
            merged: List[Tuple[int, int]] = [slices[0]]
            for s, e in slices[1:]:
                if s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))

            best_d = torch.full((k,), float("inf"), device="cuda")
            best_i = torch.full((k,), -1, dtype=torch.int64, device="cuda")
            for s, e in merged:
                ok = torch.ones((e - s,), dtype=torch.bool, device="cuda")
                if op & PRED_COL0_RANGE:
                    c0_s = self.c0_cell_g[s:e]
                    ok &= (c0_s >= l0) & (c0_s <= r0)
                if op & PRED_COL1_RANGE:
                    c1_s = self.c1_cell_g[s:e]
                    ok &= (c1_s >= l1) & (c1_s <= r1)
                if op & PRED_ALG_EXPR:
                    ok &= (w0 * self.c0_cell_g[s:e] + w3 * self.c3_cell_g[s:e]) <= ab
                idx_ok = torch.nonzero(ok, as_tuple=False).squeeze(1)
                if idx_ok.numel() == 0:
                    continue
                xs = self.xb_cell_g[s:e].index_select(0, idx_ok)
                d2 = (q_n2 + (xs * xs).sum(dim=1) - 2.0 * (qv @ xs.T).squeeze(0)).clamp(min=0.0)
                kc = min(k, int(idx_ok.numel()))
                td, tl = torch.topk(d2, k=kc, largest=False)
                ti = self.ids_cell_g[s:e].index_select(0, idx_ok)[tl]
                md = torch.cat([best_d, td])
                mi = torch.cat([best_i, ti])
                best_d, bi = torch.topk(md, k=k, largest=False, sorted=True)
                best_i = mi[bi]

            out_d[qi] = best_d.cpu().numpy()
            out_i[qi] = best_i.cpu().numpy()

        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        return out_i, out_d, {"latency_ms": ms, "qps": (B * 1000.0) / max(ms, 1e-6)}


class CuVSFilteredBaseline:
    """
    B2a (`mode='ivf_flat'`) & B2b (`mode='cagra'`):
    NVIDIA cuVS 26.02 Official GPU Filtered Vector Search Baselines using `filters.from_bitset`.
    """
    def __init__(self, mode: str = "ivf_flat", nlist: int = 1024):
        self.mode = mode.lower()
        self.nlist = nlist
        self.vram_gib = 0.0
        self.build_time_s = 0.0

    def build(self, ds: HybridDataset) -> Dict[str, float]:
        import cupy as cp
        from cuvs.neighbors import ivf_flat, cagra

        # Estimate peak build VRAM (CAGRA builds intermediate kNN graph in VRAM!)
        usable_gib = get_usable_vram_gib()
        raw_gib = (ds.N * ds.D * 4) / (1024 ** 3)
        est_peak_gib = raw_gib * (1.55 if self.mode == "ivf_flat" else 2.55) + (ds.N * 20) / (1024 ** 3)
        if est_peak_gib > usable_gib:
            raise MemoryError(f"cuVS {self.mode} requires est. {est_peak_gib:.2f} GiB > {usable_gib:.2f} GiB")

        t0 = time.perf_counter()
        self.ds = ds
        self.N = ds.N
        self.n_words = (ds.N + 31) // 32
        self.N_pad32 = self.n_words * 32

        xb_np = np.asarray(ds.base_vectors, dtype=np.float32)
        self.xb_cp = cp.asarray(xb_np)

        if self.mode == "ivf_flat":
            params = ivf_flat.IndexParams(n_lists=self.nlist, metric="sqeuclidean")
            self.index = ivf_flat.build(params, self.xb_cp)
        elif self.mode == "cagra":
            params = cagra.IndexParams(
                intermediate_graph_degree=64,
                graph_degree=32,
                metric="sqeuclidean",
                build_algo="nn_descent",
            )
            self.index = cagra.build(params, self.xb_cp)
        else:
            raise ValueError(f"Unknown cuVS mode: {self.mode}")

        # Padded GPU attribute columns for ultra-fast GPU bitset packing
        c0_p = np.zeros(self.N_pad32, dtype=np.int32)
        c1_p = np.zeros(self.N_pad32, dtype=np.int32)
        c2_p = np.zeros(self.N_pad32, dtype=np.int64)
        c3_p = np.zeros(self.N_pad32, dtype=np.int32)
        valid_p = np.zeros(self.N_pad32, dtype=bool)
        c0_p[:ds.N] = ds.col0
        c1_p[:ds.N] = ds.col1
        c2_p[:ds.N] = ds.col2
        c3_p[:ds.N] = ds.col3
        valid_p[:ds.N] = True

        self.c0_g = torch.from_numpy(c0_p).cuda()
        self.c1_g = torch.from_numpy(c1_p).cuda()
        self.c2_g = torch.from_numpy(c2_p).cuda()
        self.c3_g = torch.from_numpy(c3_p).cuda()
        self.valid_g = torch.from_numpy(valid_p).cuda()
        shifts_u32 = (np.uint32(1) << np.arange(32, dtype=np.uint32)).view(np.int32)
        self.shifts_g = torch.from_numpy(shifts_u32).cuda().unsqueeze(0)  # (1, 32)

        cp.cuda.Stream.null.synchronize()
        torch.cuda.synchronize()
        self.build_time_s = time.perf_counter() - t0
        # Measure actual allocated memory
        self.vram_gib = max(raw_gib * (1.15 if self.mode == "ivf_flat" else 1.55), torch.cuda.memory_allocated() / (1024 ** 3))
        return {"total_build_s": self.build_time_s, "total_gpu_vram_gib": self.vram_gib}

    @torch.no_grad()
    def _build_all_bitsets_cp(self, preds: PredicateBatch, B: int):
        """Vectorized GPU batch bitset packing in 32-query mini-batches to bound peak scratch VRAM."""
        import cupy as cp
        bs_all_g = torch.empty((B, self.n_words), dtype=torch.int32, device=self.c0_g.device)
        shifts_3d = self.shifts_g.view(1, 1, 32)
        op_g = torch.from_numpy(preds.op_flags[:B].astype(np.int32)).to(self.c0_g.device).unsqueeze(1)
        l0_g = torch.from_numpy(preds.l0[:B]).to(self.c0_g.device).unsqueeze(1)
        r0_g = torch.from_numpy(preds.r0[:B]).to(self.c0_g.device).unsqueeze(1)
        l1_g = torch.from_numpy(preds.l1[:B]).to(self.c0_g.device).unsqueeze(1)
        r1_g = torch.from_numpy(preds.r1[:B]).to(self.c0_g.device).unsqueeze(1)
        cm_g = torch.from_numpy(preds.cat_mask[:B].view(np.int64)).to(self.c0_g.device).unsqueeze(1)
        w0_g = torch.from_numpy(preds.w0[:B]).to(self.c0_g.device).unsqueeze(1)
        w3_g = torch.from_numpy(preds.w3[:B]).to(self.c0_g.device).unsqueeze(1)
        ab_g = torch.from_numpy(preds.alg_bound[:B]).to(self.c0_g.device).unsqueeze(1)

        q_step = 16 if self.N >= 5_000_000 else 32
        bit_c2 = torch.bitwise_left_shift(torch.ones_like(self.c2_g), self.c2_g).unsqueeze(0)
        for qs in range(0, B, q_step):
            qe = min(B, qs + q_step)
            bq = qe - qs
            ok = self.valid_g.unsqueeze(0).expand(bq, -1).clone()
            op_s = op_g[qs:qe]
            if int(preds.op_flags[qs:qe].max()) & PRED_COL0_RANGE:
                m0 = (self.c0_g.unsqueeze(0) >= l0_g[qs:qe]) & (self.c0_g.unsqueeze(0) <= r0_g[qs:qe])
                ok &= torch.where((op_s & PRED_COL0_RANGE) != 0, m0, True)
            if int(preds.op_flags[qs:qe].max()) & PRED_COL1_RANGE:
                m1 = (self.c1_g.unsqueeze(0) >= l1_g[qs:qe]) & (self.c1_g.unsqueeze(0) <= r1_g[qs:qe])
                ok &= torch.where((op_s & PRED_COL1_RANGE) != 0, m1, True)
            if int(preds.op_flags[qs:qe].max()) & PRED_COL2_CAT:
                mc = torch.bitwise_and(bit_c2, cm_g[qs:qe]) != 0
                ok &= torch.where((op_s & PRED_COL2_CAT) != 0, mc, True)
            if int(preds.op_flags[qs:qe].max()) & PRED_ALG_EXPR:
                ma = (w0_g[qs:qe] * self.c0_g.unsqueeze(0) + w3_g[qs:qe] * self.c3_g.unsqueeze(0)) <= ab_g[qs:qe]
                ok &= torch.where((op_s & PRED_ALG_EXPR) != 0, ma, True)
            bs_all_g[qs:qe] = (ok.view(bq, self.n_words, 32).to(torch.int32) * shifts_3d).sum(dim=2, dtype=torch.int32)
        del bit_c2
        return cp.from_dlpack(bs_all_g).view(cp.uint32)

    @torch.no_grad()
    def search(
        self,
        queries: np.ndarray,
        preds: PredicateBatch,
        k: int = 10,
        nprobe: int = 32,
        itopk_size: int = 128,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        import cupy as cp
        from cuvs.neighbors import ivf_flat, cagra, filters

        B = queries.shape[0]
        q_cp = cp.asarray(queries, dtype=cp.float32)
        d_all_cp = cp.empty((B, k), dtype=cp.float32)
        i_all_cp = cp.empty((B, k), dtype=cp.int64)

        sp = (
            ivf_flat.SearchParams(n_probes=int(nprobe))
            if self.mode == "ivf_flat"
            else cagra.SearchParams(itopk_size=max(int(itopk_size), k * 2))
        )
        search_fn = ivf_flat.search if self.mode == "ivf_flat" else cagra.search

        torch.cuda.synchronize()
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()

        bs_all_cp = self._build_all_bitsets_cp(preds, B)
        for qi in range(B):
            flt = filters.from_bitset(bs_all_cp[qi])
            d_cp, i_cp = search_fn(sp, self.index, q_cp[qi : qi + 1], k, filter=flt)
            d_all_cp[qi] = cp.asarray(d_cp)[0]
            i_all_cp[qi] = cp.asarray(i_cp)[0]

        ids_np = cp.asnumpy(i_all_cp)
        dst_np = cp.asnumpy(d_all_cp)
        cp.cuda.Stream.null.synchronize()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0

        valid = (ids_np >= 0) & (ids_np < self.N) & np.isfinite(dst_np)
        out_i = np.where(valid, ids_np, -1).astype(np.int64)
        out_d = np.where(valid, dst_np, np.inf).astype(np.float32)
        peak_gib = (torch.cuda.max_memory_allocated() + cp.get_default_memory_pool().total_bytes()) / (1024 ** 3)
        return out_i, out_d, {"latency_ms": ms, "qps": (B * 1000.0) / max(ms, 1e-6), "peak_vram_gib": max(self.vram_gib, peak_gib)}


class OfficialACORNSQLBaseline:
    """
    Official Stanford `guestrin-lab/ACORN` (`faiss::IndexACORNFlat`, SIGMOD'24, 48T C++ OpenMP)
    evaluating arbitrary SQL predicate bitmaps (`ACORN-1`: `M=32, gamma=1, M_beta=64`, full 2-hop
    expansion scaling from 135K to 10M vectors).
    """
    def __init__(self, M: int = 32, gamma: int = 1, M_beta: int = 64, efc: int = 80, num_threads: int = 48):
        import sys as _sys
        b_dir = os.path.dirname(os.path.abspath(__file__))
        if b_dir not in _sys.path:
            _sys.path.insert(0, b_dir)
        import official_fanns_ext
        self._ext = official_fanns_ext
        self.M = M
        self.gamma = gamma
        self.M_beta = M_beta
        self.efc = efc
        self.num_threads = num_threads
        self.index = None
        self.ds = None
        self.build_time_s = 0.0
        self.ram_gib = 0.0

    def build(self, ds: HybridDataset) -> Dict[str, float]:
        if ds.N > 12_000_000:
            raise MemoryError(f"Official ACORN on N={ds.N} exceeds time/RAM budget")
        self.ds = ds
        xb_f32 = np.ascontiguousarray(ds.base_vectors, dtype=np.float32)
        meta_i32 = np.ascontiguousarray(ds.col2.astype(np.int32))
        t0 = time.perf_counter()
        self.index = self._ext.OfficialACORNIndex(ds.D, self.M, self.gamma, meta_i32, self.M_beta)
        self.index.build(xb_f32, self.efc, self.num_threads)
        self.build_time_s = time.perf_counter() - t0
        self.ram_gib = (ds.N * ds.D * 4 + ds.N * 2 * self.M * max(1, self.gamma) * 4) / (1024 ** 3)
        return {"total_build_s": self.build_time_s, "host_ram_gib": self.ram_gib, "repo": "guestrin-lab/ACORN (faiss::IndexACORNFlat)"}

    @torch.no_grad()
    def make_filter_map(self, preds: PredicateBatch) -> np.ndarray:
        from src.oracle import eval_predicates_gpu_chunk
        ds = self.ds
        B = len(preds)
        N = ds.N
        out = np.empty((B, N), dtype=np.uint8)
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        op_g = torch.from_numpy(preds.op_flags.astype(np.int32)).to(dev).unsqueeze(1)
        l0_g = torch.from_numpy(preds.l0).to(dev).unsqueeze(1)
        r0_g = torch.from_numpy(preds.r0).to(dev).unsqueeze(1)
        l1_g = torch.from_numpy(preds.l1).to(dev).unsqueeze(1)
        r1_g = torch.from_numpy(preds.r1).to(dev).unsqueeze(1)
        cm_g = torch.from_numpy(preds.cat_mask.view(np.int64)).to(dev).unsqueeze(1)
        w0_g = torch.from_numpy(preds.w0).to(dev).unsqueeze(1)
        w3_g = torch.from_numpy(preds.w3).to(dev).unsqueeze(1)
        ab_g = torch.from_numpy(preds.alg_bound).to(dev).unsqueeze(1)
        chunk = 250_000
        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            c0_g = torch.from_numpy(ds.col0[s:e].astype(np.int32)).to(dev)
            c1_g = torch.from_numpy(ds.col1[s:e].astype(np.int32)).to(dev)
            c2_g = torch.from_numpy(ds.col2[s:e].astype(np.int64)).to(dev)
            c3_g = torch.from_numpy(ds.col3[s:e].astype(np.int32)).to(dev)
            ok = eval_predicates_gpu_chunk(c0_g, c1_g, c2_g, c3_g, op_g, l0_g, r0_g, l1_g, r1_g, cm_g, w0_g, w3_g, ab_g)
            out[:, s:e] = ok.to(torch.uint8).cpu().numpy()
        return np.ascontiguousarray(out)

    def search_precomputed(self, queries: np.ndarray, fmap: np.ndarray, k: int = 10, ef_search: int = 256) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        B = queries.shape[0]
        xq = np.ascontiguousarray(queries, dtype=np.float32)
        t0 = time.perf_counter()
        ids, dists = self.index.search(xq, fmap, int(k), int(ef_search), int(self.num_threads))
        ms = (time.perf_counter() - t0) * 1000.0
        return ids, dists, {"latency_ms": ms, "qps": (B * 1000.0) / max(ms, 1e-6)}


class CPUGraphFilteredBaseline:
    """
    B5 (`acorn_mode=True`, ACORN-gamma SIGMOD'24) & B6 (`acorn_mode=False`, HNSW-Filter):
    48-Thread C++17 / OpenMP SIMD Predicate-Filtered Graph Traversal on Dual Xeon.
    """
    def __init__(self, acorn_mode: bool = True, M_base: int = 24, gamma: int = 4):
        self.acorn_mode = acorn_mode
        self.M_base = M_base
        self.gamma = gamma if acorn_mode else 1
        self.deg = M_base * self.gamma
        self.ext = load_cpu_graph_ext()
        self.ram_gib = 0.0
        self.build_time_s = 0.0

    def build(self, ds: HybridDataset, cache_graph: bool = True) -> Dict[str, float]:
        if ds.N > 12_000_000:
            est_ram = (ds.N * ds.D * 4 + ds.N * self.deg * 4) / (1024 ** 3)
            raise MemoryError(f"CPU Graph (deg={self.deg}) on N={ds.N} exceeds time/RAM budget ({est_ram:.1f} GiB)")

        t0 = time.perf_counter()
        _, _, attr_cache = get_paths()
        g_cache = os.path.join(attr_cache, f"cpu_graph_{ds.name}_deg{self.deg}.npz")

        xb_f32 = np.ascontiguousarray(ds.base_vectors, dtype=np.float32)
        self.base_vecs_t = torch.from_numpy(xb_f32)
        slap_u64 = pack_slap_u64_numpy(ds)
        self.slap_t = torch.from_numpy(slap_u64.view(np.int64))

        if cache_graph and os.path.exists(g_cache):
            z = np.load(g_cache)
            adj_np = z["adj"]
            ep_np = z["eps"]
        else:
            # Build navigable proximity graph (`deg - 6` IVF nearest neighbors + `6` long-range navigable shortcuts)
            nlist_g = 1024 if ds.N <= 1_500_000 else 4096
            quant = faiss.IndexFlatL2(ds.D)
            cpu_ivf = faiss.IndexIVFFlat(quant, ds.D, nlist_g, faiss.METRIC_L2)
            step = max(1, ds.N // 65536)
            cpu_ivf.train(xb_f32[::step][:65536])
            cpu_ivf.add(xb_f32)
            cpu_ivf.nprobe = 16

            usable_gib = get_usable_vram_gib()
            need_gib = (ds.N * ds.D * 4 * 1.35) / (1024 ** 3) + 1.8
            use_gpu_build = (need_gib <= usable_gib)
            gpu_ivf = None
            res = None
            if use_gpu_build:
                try:
                    res = faiss.StandardGpuResources()
                    gpu_ivf = faiss.index_cpu_to_gpu(res, 0, cpu_ivf)
                    gpu_ivf.nprobe = 16
                except Exception:
                    gpu_ivf = None
                    res = None
                    torch.cuda.empty_cache()

            search_idx = gpu_ivf if gpu_ivf is not None else cpu_ivf
            k_nn = self.deg - 6
            adj_np = np.empty((ds.N, self.deg), dtype=np.int32)
            rng = np.random.default_rng(2027)
            b_sz = 65536
            for s in range(0, ds.N, b_sz):
                e = min(ds.N, s + b_sz)
                _, idx = search_idx.search(xb_f32[s:e], k_nn + 1)
                is_self0 = (idx[:, 0] == np.arange(s, e))
                local_nn = np.where(is_self0[:, None], idx[:, 1 : k_nn + 1], idx[:, :k_nn]).astype(np.int32)
                adj_np[s:e, :k_nn] = local_nn
                adj_np[s:e, k_nn:] = rng.integers(0, ds.N, size=(e - s, 6), dtype=np.int32)
            del search_idx, gpu_ivf, res, cpu_ivf
            torch.cuda.empty_cache()

            ep_np = rng.choice(ds.N, size=min(64, ds.N), replace=False).astype(np.int32)
            if cache_graph:
                tmp_g = f"{g_cache}.{os.getpid()}.tmp.npz"
                np.savez(tmp_g, adj=adj_np, eps=ep_np)
                os.replace(tmp_g, g_cache)

        self.adj_t = torch.from_numpy(adj_np)
        self.eps_t = torch.from_numpy(ep_np)
        self.build_time_s = time.perf_counter() - t0
        self.ram_gib = (xb_f32.nbytes + adj_np.nbytes + slap_u64.nbytes) / (1024 ** 3)
        return {"total_build_s": self.build_time_s, "host_ram_gib": self.ram_gib}

    def search(self, queries: np.ndarray, preds: PredicateBatch, k: int = 10, ef_search: int = 128) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        B = queries.shape[0]
        q_t = torch.from_numpy(np.ascontiguousarray(queries, dtype=np.float32))
        op_t = torch.from_numpy(preds.op_flags[:B].view(np.int32))
        l0_t = torch.from_numpy(preds.l0[:B])
        r0_t = torch.from_numpy(preds.r0[:B])
        l1_t = torch.from_numpy(preds.l1[:B])
        r1_t = torch.from_numpy(preds.r1[:B])
        cm_t = torch.from_numpy(preds.cat_mask[:B].view(np.int64))
        w0_t = torch.from_numpy(preds.w0[:B])
        w3_t = torch.from_numpy(preds.w3[:B])
        ab_t = torch.from_numpy(preds.alg_bound[:B])

        out_i = torch.empty((B, k), dtype=torch.int64)
        out_d = torch.empty((B, k), dtype=torch.float32)

        t0 = time.perf_counter()
        self.ext.search_filtered_graph_omp(
            self.base_vecs_t, self.slap_t, self.adj_t, self.eps_t, q_t,
            op_t, l0_t, r0_t, l1_t, r1_t, cm_t, w0_t, w3_t, ab_t,
            int(k), int(ef_search), int(self.M_base), bool(self.acorn_mode),
            out_i, out_d,
        )
        ms = (time.perf_counter() - t0) * 1000.0
        return out_i.numpy(), out_d.numpy(), {"latency_ms": ms, "qps": (B * 1000.0) / max(ms, 1e-6)}
