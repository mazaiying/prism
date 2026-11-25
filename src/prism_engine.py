#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRISM Core Engine (M1 Layout + M2 Warp-Compacted Co-Pruning + M3 Three-Regime Optimizer).

Key Architectural Properties:
- M1: GPU-resident `TileHeader` (32B / 32 rows = 1B/row) + `SLAP` 64-bit packed columns (8B/row)
      + `Sign-Refined INT4 + 1-bit` Subspace Ball Sketch (~5.25 bits/dim + 4B/block), while
      exact full-precision vectors + `perm_to_orig_id` reside in Host Pinned UVA Memory
      (or optionally GPU DRAM for `B1: GPU-IVF-PreFilter` baseline).
- M2: `TileHeader` O(1) skip/full-hit -> `SLAP` boundary predicate -> Shared-Memory Per-Warp
      Stack Compaction (`sh_ring[8][64]`) -> Progressive M-Block ULP-Safe Subspace Ball Pruning
      -> 32-Lane Coalesced UVA Exact Refinement.
- M3: Sub-millisecond TileHeader selectivity & survivor-skew estimator + NLMS-calibrated
      Analytical Three-Regime Cost Model dispatching queries across:
        * Regime 1 (`Global Columnar Pushdown`, ultra-low selectivity)
        * Regime 2 (`Prewarmed Warp-Compacted IVF Co-Pruning`, medium/low selectivity)
        * Regime 3 (`Sketch-First Late-Predicate Pipeline`, high selectivity)
"""

from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import faiss

from src.hybrid_datasets import HybridDataset, PredicateBatch, get_paths
from src.ext_loader import load_prism_ext


@dataclass
class CostModelParams:
    """Analytical cost model coefficients (in milliseconds) with online NLMS calibration."""
    # Regime 1: c1_base + c1_tile * N_wtiles + c1_surv * (est_sel * N * D_blocks)
    c1_base: float = 0.035
    c1_tile: float = 4.2e-7
    c1_surv: float = 1.8e-6
    # Regime 2: c2_base + c2_tile * probe_tiles + c2_sketch * probe_surv * D_blocks + alpha_skew * skew
    c2_base: float = 0.025
    c2_tile: float = 5.5e-7
    c2_sketch: float = 1.6e-6
    alpha_skew: float = 0.008
    # Regime 3: c3_base + c3_b0 * probe_rows * b0_Cost + c3_rem * (0.28 * probe_surv * (D_blocks - 1))
    c3_base: float = 0.025
    c3_b0: float = 4.5e-7
    c3_rem: float = 1.4e-6
    # Regime crossover bounds
    sel_low_thresh: float = 0.008
    sel_high_thresh: float = 0.35
    # History for E7 cost model validation
    records: List[Dict[str, float]] = field(default_factory=list)

    def predict_batch_ms(
        self,
        regime: int,
        B: int,
        N: int,
        N_wtiles: int,
        D_pad: int,
        nlist: int,
        mean_probes: float,
        mean_global_sel: float,
        mean_probe_surv: float,
        mean_skew: float,
    ) -> float:
        d_scale = D_pad / 128.0
        if regime == 1:
            per_q = (
                self.c1_base
                + self.c1_tile * N_wtiles
                + self.c1_surv * (mean_global_sel * N) * d_scale
            )
        elif regime == 2:
            probe_tiles = (mean_probes / max(1, nlist)) * N_wtiles
            per_q = (
                self.c2_base
                + self.c2_tile * probe_tiles
                + self.c2_sketch * mean_probe_surv * d_scale
                + self.alpha_skew * mean_skew * (mean_probes / 32.0)
            )
        else:
            probe_rows = (mean_probes / max(1, nlist)) * N
            per_q = (
                self.c3_base
                + self.c3_b0 * probe_rows * d_scale
                + self.c3_rem * (0.28 * mean_probe_surv) * d_scale
            )
        # Account for multi-CTA GPU batch parallelism saturation (~64 queries in flight)
        par_factor = max(1.0, B / 48.0) if B >= 16 else max(0.35, B / 16.0)
        return float(per_q * par_factor)

    def nlms_update(self, regime: int, pred_ms: float, actual_ms: float, lr: float = 0.08) -> None:
        if pred_ms <= 1e-6 or actual_ms <= 1e-6:
            return
        # Fit non-negative linear coefficients from recorded feature vectors when >= 3 calibration samples exist
        sub = [r for r in self.records if r.get("regime") == regime and "feat" in r]
        if len(sub) >= 3:
            from scipy.optimize import nnls
            X = np.array([r["feat"] for r in sub], dtype=np.float64)
            y = np.array([r["actual_ms"] for r in sub], dtype=np.float64)
            w, _ = nnls(X, y)
            if w.sum() > 0:
                if regime == 1:
                    self.c1_base, self.c1_tile, self.c1_surv = float(max(w[0], 1e-4)), float(w[1]), float(w[2])
                elif regime == 2:
                    self.c2_base, self.c2_tile, self.c2_sketch, self.alpha_skew = (
                        float(max(w[0], 1e-4)), float(w[1]), float(w[2]), float(w[3])
                    )
                else:
                    self.c3_base, self.c3_b0, self.c3_rem = float(max(w[0], 1e-4)), float(w[1]), float(w[2])
                return
        ratio = np.clip(actual_ms / pred_ms, 0.5, 2.0)
        scale = (1.0 - lr) + lr * float(ratio)
        if regime == 1:
            self.c1_base *= scale
            self.c1_tile *= scale
            self.c1_surv *= scale
        elif regime == 2:
            self.c2_base *= scale
            self.c2_tile *= scale
            self.c2_sketch *= scale
        else:
            self.c3_base *= scale
            self.c3_b0 *= scale
            self.c3_rem *= scale


class PrismEngine:
    def __init__(
        self,
        nlist: Optional[int] = None,
        M_blocks: int = 4,
        use_sign_bit: bool = True,
        sketch_mode: int = 2,          # 2 = INT4+1b, 1 = INT4-only, 0 = No-sketch PreFilter
        host_pinned_exact: bool = True, # True = Host Pinned UVA (PRISM), False = GPU Resident FP32 (B1)
        quantile_clip: float = 0.005,   # [0.5%, 99.5%] outlier-immune grid placement
        device: str = "cuda:0",
    ):
        self.nlist = nlist
        self.M_blocks = M_blocks
        self.use_sign_bit = use_sign_bit and (sketch_mode == 2)
        self.sketch_mode = sketch_mode
        self.host_pinned_exact = host_pinned_exact
        self.quantile_clip = quantile_clip
        self.device = torch.device(device)
        self.ext = load_prism_ext()
        self.cost_model = CostModelParams()

        # Populated in build()
        self.N = 0
        self.D = 0
        self.D_pad = 0
        self.chunks32 = 0
        self.M = 0
        self.N_wtiles = 0
        self.N_pad = 0
        self.build_stats: Dict[str, float] = {}

    def _default_nlist(self, N: int) -> int:
        if N <= 1_500_000:
            return 1024
        if N <= 15_000_000:
            return 4096
        return 8192

    @torch.no_grad()
    def build(
        self,
        ds: HybridDataset,
        cache_kmeans: bool = True,
        mutable_headroom: float = 0.0,
    ) -> Dict[str, float]:
        t0_all = time.perf_counter()
        self.N = ds.N
        self.D = ds.D
        self.D_pad = ((self.D + 31) // 32) * 32
        self.chunks32 = self.D_pad // 32
        self.M = min(self.M_blocks, self.chunks32)
        if self.nlist is None:
            self.nlist = self._default_nlist(self.N)

        _, _, attr_cache = get_paths()
        km_cache = os.path.join(attr_cache, f"kmeans_{ds.name}_n{self.N}_nl{self.nlist}.npz")
        legacy_km = os.path.join(attr_cache, f"kmeans_{ds.name}_nl{self.nlist}.npz")

        # Step 1: Train or load IVF centroids + cluster assignments
        t_km0 = time.perf_counter()
        if cache_kmeans and os.path.exists(legacy_km) and mutable_headroom == 0.0:
            z = np.load(legacy_km)
            if z["assign"].shape[0] == self.N:
                centroids_np = z["centroids"].astype(np.float32)
                assign_np = z["assign"].astype(np.int32)
            else:
                legacy_km = ""
        else:
            legacy_km = ""

        if not legacy_km:
            if cache_kmeans and os.path.exists(km_cache):
                z = np.load(km_cache)
                centroids_np = z["centroids"].astype(np.float32)
                assign_np = z["assign"].astype(np.int32)
            else:
                train_n = min(self.N, max(65536, self.nlist * 40))
                step = max(1, self.N // train_n)
                x_train = np.array(ds.base_vectors[::step][:train_n], dtype=np.float32, copy=True)
                km = faiss.Kmeans(self.D, self.nlist, niter=20, verbose=False, gpu=True, seed=2027)
                km.train(x_train)
                centroids_np = np.array(km.centroids, dtype=np.float32, copy=True)

                # Assign all N vectors in bounded GPU chunks
                assign_np = np.empty(self.N, dtype=np.int32)
                cent_g = torch.from_numpy(centroids_np).to(self.device)
                cn2 = (cent_g * cent_g).sum(dim=1).unsqueeze(0)
                a_chunk = 65_536
                for s in range(0, self.N, a_chunk):
                    e = min(self.N, s + a_chunk)
                    xb_c = torch.from_numpy(np.array(ds.base_vectors[s:e], dtype=np.float32, copy=True)).to(self.device)
                    dists = cn2 - 2.0 * (xb_c @ cent_g.T)
                    assign_np[s:e] = torch.argmin(dists, dim=1).to(torch.int32).cpu().numpy()
                del cent_g, cn2
                if cache_kmeans:
                    tmp_km = f"{km_cache}.{os.getpid()}.tmp.npz"
                    np.savez(tmp_km, centroids=centroids_np, assign=assign_np)
                    os.replace(tmp_km, km_cache)
        t_kmeans = time.perf_counter() - t_km0

        # Step 2: Compute cluster Warp-Tile layout (32-tuple aligned per cluster)
        t_lay0 = time.perf_counter()
        order = np.argsort(assign_np, kind="mergesort")
        counts = np.bincount(assign_np, minlength=self.nlist).astype(np.int64)
        c_row_off = np.zeros(self.nlist + 1, dtype=np.int64)
        np.cumsum(counts, out=c_row_off[1:])

        c_tiles_active = ((counts + 31) // 32).astype(np.int32)
        if mutable_headroom > 0.0:
            mean_t = max(12, int(np.ceil(float(c_tiles_active.mean()))))
            extra_t = np.maximum(
                int(np.ceil(mean_t * mutable_headroom * 2.5)),
                np.ceil(c_tiles_active * (mutable_headroom * 2.5)).astype(np.int32),
            )
            c_tiles = c_tiles_active + extra_t
        else:
            c_tiles = c_tiles_active
        c_tile_off = np.zeros(self.nlist + 1, dtype=np.int32)
        np.cumsum(c_tiles, out=c_tile_off[1:])
        self.N_wtiles = int(c_tile_off[-1])
        self.N_pad = int(self.N_wtiles) * 32

        # Compute per-cluster quantile-clipped [o_{c,j}, delta_{c,j}] (only needed when sketch_mode >= 1)
        cluster_o_np = np.zeros((self.nlist, self.D_pad), dtype=np.float32)
        cluster_d_np = np.zeros((self.nlist, self.D_pad), dtype=np.float32)

        if self.sketch_mode >= 1:
            q_lo_p = self.quantile_clip * 100.0
            q_hi_p = (1.0 - self.quantile_clip) * 100.0
            for c in range(self.nlist):
                s_idx, e_idx = int(c_row_off[c]), int(c_row_off[c + 1])
                cnt = e_idx - s_idx
                if cnt == 0:
                    continue
                rows_c = order[s_idx:e_idx]
                if cnt > 256:
                    rows_c = rows_c[:: max(1, cnt // 256)][:256]
                xc = np.asarray(ds.base_vectors[rows_c], dtype=np.float32)
                if cnt >= 16 and self.quantile_clip > 0.0:
                    lo = np.percentile(xc, q_lo_p, axis=0).astype(np.float32)
                    hi = np.percentile(xc, q_hi_p, axis=0).astype(np.float32)
                else:
                    lo = xc.min(axis=0).astype(np.float32)
                    hi = xc.max(axis=0).astype(np.float32)
                cluster_o_np[c, :self.D] = lo
                cluster_d_np[c, :self.D] = np.maximum((hi - lo) / 15.0, 1e-7)

        # Macro-block boundaries in units of 32-dim chunks
        bounds = np.linspace(0, self.chunks32, self.M + 1, dtype=np.int32)
        self.centroids_g = torch.from_numpy(centroids_np).to(self.device)
        self.cent_norm2_g = (self.centroids_g * self.centroids_g).sum(dim=1).unsqueeze(0)
        self.cluster_o_g = torch.from_numpy(cluster_o_np).to(self.device)
        self.cluster_delta_g = torch.from_numpy(cluster_d_np).to(self.device)
        self.cluster_t_off_g = torch.from_numpy(c_tile_off).to(self.device)
        self.block_bounds_g = torch.from_numpy(bounds).to(self.device)
        t_layout = time.perf_counter() - t_lay0

        # Step 3: Allocate GPU-resident SLAP columns + Subspace Ball Sketch & Host Pinned Exact Store
        t_enc0 = time.perf_counter()
        self.headers_g = torch.empty((self.N_wtiles, 32), dtype=torch.uint8, device=self.device)
        self.slap_cols_g = torch.empty((self.N_pad,), dtype=torch.int64, device=self.device)

        if self.sketch_mode >= 1:
            self.int4_g = torch.empty((self.chunks32, self.N_pad, 4), dtype=torch.int32, device=self.device)
            self.sign_g = torch.empty((self.chunks32, self.N_pad), dtype=torch.int32, device=self.device)
            self.ball_meta_g = torch.empty((self.M, self.N_pad), dtype=torch.int32, device=self.device)
        else:
            # Dummy 1-element tensors when sketch_mode == 0 (Exact PreFilter baseline)
            self.int4_g = torch.empty((1, 1, 4), dtype=torch.int32, device=self.device)
            self.sign_g = torch.empty((1, 1), dtype=torch.int32, device=self.device)
            self.ball_meta_g = torch.empty((1, 1), dtype=torch.int32, device=self.device)

        vec_is_u8 = (ds.base_vectors.dtype == np.uint8)
        exact_dtype = torch.uint8 if vec_is_u8 else torch.float32
        if self.host_pinned_exact:
            self.exact_vecs_store = torch.zeros((self.N_pad, self.D), dtype=exact_dtype, device="cpu", pin_memory=True)
            self.perm_ids_store = torch.full((self.N_pad,), -1, dtype=torch.int64, device="cpu", pin_memory=True)
        else:
            self.exact_vecs_store = torch.zeros((self.N_pad, self.D), dtype=exact_dtype, device=self.device)
            self.perm_ids_store = torch.full((self.N_pad,), -1, dtype=torch.int64, device=self.device)

        # Vectorized O(N) tile metadata & padded index construction (supports 100M rows in < 0.2s)
        tile_valid_all = np.zeros(self.N_wtiles, dtype=np.uint16)
        for c in range(self.nlist):
            ta = int(c_tiles_active[c])
            if ta > 0:
                t0 = int(c_tile_off[c])
                tile_valid_all[t0 : t0 + ta - 1] = 32
                rem = int(counts[c] % 32)
                tile_valid_all[t0 + ta - 1] = 32 if rem == 0 else rem

        tile_cid_all = np.repeat(np.arange(self.nlist, dtype=np.int32), c_tiles)
        pad_to_orig = np.full(self.N_pad, -1, dtype=np.int64)
        cluster_pad_shift = c_tile_off[:-1].astype(np.int64) * 32 - c_row_off[:-1]
        dest_positions = np.arange(self.N, dtype=np.int64) + np.repeat(cluster_pad_shift, counts)
        pad_to_orig[dest_positions] = order

        self._c_tile_off_np = c_tile_off.copy()
        self._c_tile_cap_np = c_tiles.copy()
        self._c_row_counts_np = counts.copy()
        self._mutable_enabled = (mutable_headroom > 0.0)
        if self._mutable_enabled:
            self._orig_to_pad = np.full(int(self.N * (1.0 + mutable_headroom)) + 65536, -1, dtype=np.int64)
            self._orig_to_pad[order] = dest_positions
            self._slap_shadow = np.zeros((self.N_pad, 4), dtype=np.uint16)
            self._slap_shadow[dest_positions, 0] = ds.col0[order]
            self._slap_shadow[dest_positions, 1] = ds.col1[order]
            self._slap_shadow[dest_positions, 2] = ds.col2[order]
            self._slap_shadow[dest_positions, 3] = ds.col3[order]
            self._tombstone_mask = np.zeros(self.N_pad, dtype=bool)

        if self.host_pinned_exact:
            self.perm_ids_store.numpy()[:] = pad_to_orig
        else:
            self.perm_ids_store.copy_(torch.from_numpy(pad_to_orig))

        tile_batch = 16_384  # 524,288 rows per batch -> < 250 MB temporary VRAM!
        np_vec_dtype = np.uint8 if vec_is_u8 else np.float32

        for tb in range(0, self.N_wtiles, tile_batch):
            te = min(self.N_wtiles, tb + tile_batch)
            nt = te - tb
            r_beg, r_end = tb * 32, te * 32
            orig_slice = pad_to_orig[r_beg:r_end]
            valid_mask = orig_slice >= 0
            safe_idx = np.where(valid_mask, orig_slice, 0)

            v_chunk = np.zeros((nt * 32, self.D), dtype=np_vec_dtype)
            v_chunk[valid_mask] = ds.base_vectors[orig_slice[valid_mask]]

            if self.host_pinned_exact:
                self.exact_vecs_store.numpy()[r_beg:r_end] = v_chunk
            else:
                self.exact_vecs_store[r_beg:r_end].copy_(torch.from_numpy(v_chunk))

            c0_c = np.where(valid_mask, ds.col0[safe_idx], 0).astype(np.uint16)
            c1_c = np.where(valid_mask, ds.col1[safe_idx], 0).astype(np.uint16)
            c2_c = np.where(valid_mask, ds.col2[safe_idx], 0).astype(np.uint16)
            c3_c = np.where(valid_mask, ds.col3[safe_idx], 0).astype(np.uint16)

            if self.sketch_mode >= 1:
                self.ext.encode_prism_tiles(
                    torch.from_numpy(v_chunk).to(self.device),
                    torch.from_numpy(c0_c).to(self.device),
                    torch.from_numpy(c1_c).to(self.device),
                    torch.from_numpy(c2_c).to(self.device),
                    torch.from_numpy(c3_c).to(self.device),
                    torch.from_numpy(tile_valid_all[tb:te]).to(self.device),
                    torch.from_numpy(tile_cid_all[tb:te]).to(self.device),
                    self.cluster_o_g,
                    self.cluster_delta_g,
                    self.block_bounds_g,
                    tb, self.N_pad, self.D, self.D_pad, self.M,
                    bool(self.use_sign_bit),
                    self.headers_g,
                    self.slap_cols_g,
                    self.int4_g,
                    self.sign_g,
                    self.ball_meta_g,
                )
            else:
                # Encode only headers & SLAP columns using dummy 1-row sketch buffers?
                # Wait: when sketch_mode == 0, encode headers + slap_cols on CPU/GPU cleanly!
                slap_u64 = (
                    c0_c.astype(np.uint64)
                    | (c1_c.astype(np.uint64) << np.uint64(16))
                    | (c3_c.astype(np.uint64) << np.uint64(32))
                    | ((c2_c.astype(np.uint64) & np.uint64(63)) << np.uint64(48))
                )
                self.slap_cols_g[r_beg:r_end].copy_(torch.from_numpy(slap_u64.view(np.int64)))
                # Vectorized TileHeader construction on CPU
                hdr_np = np.zeros((nt, 32), dtype=np.uint8)
                v_mask2d = valid_mask.reshape(nt, 32)
                c0_2d = c0_c.reshape(nt, 32)
                c1_2d = c1_c.reshape(nt, 32)
                c2_2d = c2_c.reshape(nt, 32)
                c3_2d = c3_c.reshape(nt, 32)
                hdr_u16 = hdr_np.view(np.uint16).reshape(nt, 16)
                hdr_u16[:, 0] = np.where(v_mask2d, c0_2d, 65535).min(axis=1)
                hdr_u16[:, 1] = np.where(v_mask2d, c0_2d, 0).max(axis=1)
                hdr_u16[:, 2] = np.where(v_mask2d, c1_2d, 65535).min(axis=1)
                hdr_u16[:, 3] = np.where(v_mask2d, c1_2d, 0).max(axis=1)
                hdr_u16[:, 4] = np.where(v_mask2d, c3_2d, 65535).min(axis=1)
                hdr_u16[:, 5] = np.where(v_mask2d, c3_2d, 0).max(axis=1)
                hdr_u16[:, 6] = tile_valid_all[tb:te]
                blm_bits = np.where(v_mask2d, np.uint64(1) << c2_2d.astype(np.uint64), np.uint64(0))
                hdr_np.view(np.uint64).reshape(nt, 4)[:, 2] = np.bitwise_or.reduce(blm_bits, axis=1)
                hdr_np.view(np.uint32).reshape(nt, 8)[:, 6] = (np.arange(tb, te, dtype=np.uint32) * 32)
                hdr_np.view(np.int32).reshape(nt, 8)[:, 7] = tile_cid_all[tb:te]
                self.headers_g[tb:te].copy_(torch.from_numpy(hdr_np))

        torch.cuda.synchronize(self.device)
        t_encode = time.perf_counter() - t_enc0

        mem = self.memory_breakdown()
        self.build_stats = {
            "kmeans_time_s": float(t_kmeans),
            "layout_time_s": float(t_layout),
            "encode_time_s": float(t_encode),
            "total_build_s": float(time.perf_counter() - t0_all),
            **mem,
        }
        return self.build_stats

    def memory_breakdown(self) -> Dict[str, float]:
        def _bytes(t: torch.Tensor) -> int:
            return int(t.numel() * t.element_size()) if t.is_cuda else 0

        vram_headers = _bytes(self.headers_g)
        vram_slap = _bytes(self.slap_cols_g)
        vram_sketch = _bytes(self.int4_g) + _bytes(self.sign_g) + _bytes(self.ball_meta_g) if self.sketch_mode >= 1 else 0
        vram_meta = _bytes(self.centroids_g) + _bytes(self.cluster_o_g) + _bytes(self.cluster_delta_g) + _bytes(self.cluster_t_off_g)
        vram_exact = _bytes(self.exact_vecs_store) + _bytes(self.perm_ids_store)
        total_vram = vram_headers + vram_slap + vram_sketch + vram_meta + vram_exact
        host_pinned = 0 if not self.host_pinned_exact else int(
            self.exact_vecs_store.numel() * self.exact_vecs_store.element_size()
            + self.perm_ids_store.numel() * self.perm_ids_store.element_size()
        )
        return {
            "vram_headers_mb": vram_headers / (1024 ** 2),
            "vram_slap_cols_mb": vram_slap / (1024 ** 2),
            "vram_sketch_mb": vram_sketch / (1024 ** 2),
            "vram_cluster_meta_mb": vram_meta / (1024 ** 2),
            "vram_exact_vecs_mb": vram_exact / (1024 ** 2),
            "total_gpu_vram_mb": total_vram / (1024 ** 2),
            "total_gpu_vram_gib": total_vram / (1024 ** 3),
            "host_pinned_gib": host_pinned / (1024 ** 3),
        }

    @torch.no_grad()
    def insert_batch(
        self,
        new_vecs: np.ndarray,
        col0: np.ndarray,
        col1: np.ndarray,
        col2: np.ndarray,
        col3: np.ndarray,
        new_orig_ids: np.ndarray,
    ) -> Dict[str, float]:
        """Incremental GPU insertion against frozen centroids & quantile grids (Theorem 4.1 lossless)."""
        assert getattr(self, "_mutable_enabled", False), "Build engine with mutable_headroom > 0 for inserts"
        m_new = int(new_vecs.shape[0])
        if m_new == 0:
            return {"insert_ms": 0.0, "insert_ips": 0.0, "touched_tiles": 0}

        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()

        # 1. Assign new vectors to nearest frozen IVF centroids in bounded 65,536-row GPU chunks
        c_assign = np.empty(m_new, dtype=np.int32)
        a_chunk = 65_536
        for s_c in range(0, m_new, a_chunk):
            e_c = min(m_new, s_c + a_chunk)
            xv_c = torch.from_numpy(np.ascontiguousarray(new_vecs[s_c:e_c], dtype=np.float32)).to(self.device)
            dists_c = self.cent_norm2_g - 2.0 * (xv_c @ self.centroids_g.T)
            c_assign[s_c:e_c] = torch.argmin(dists_c, dim=1).to(torch.int32).cpu().numpy()

        # 2. Group by target cluster and place into tail / fresh reserved 32-row warp tiles
        ord_ins = np.argsort(c_assign, kind="mergesort")
        c_sorted = c_assign[ord_ins]
        uniq_c, idx_start, c_add = np.unique(c_sorted, return_index=True, return_counts=True)

        max_oid = int(new_orig_ids.max())
        if max_oid >= self._orig_to_pad.shape[0]:
            grown = np.full(max_oid + 65536, -1, dtype=np.int64)
            grown[: self._orig_to_pad.shape[0]] = self._orig_to_pad
            self._orig_to_pad = grown

        touched_tiles_list: List[int] = []
        touched_cids_list: List[int] = []
        touched_vcnt_list: List[int] = []

        exact_np = self.exact_vecs_store.numpy() if self.host_pinned_exact else None
        perm_np = self.perm_ids_store.numpy() if self.host_pinned_exact else None

        for c, s_i, add_cnt in zip(uniq_c.tolist(), idx_start.tolist(), c_add.tolist()):
            old_cnt = int(self._c_row_counts_np[c])
            new_cnt = old_cnt + int(add_cnt)
            max_rows_c = int(self._c_tile_cap_np[c]) * 32
            if new_cnt > max_rows_c:
                raise RuntimeError(f"Cluster {c} exceeded reserved mutable headroom ({new_cnt} > {max_rows_c}); increase mutable_headroom")

            base_pad = int(self._c_tile_off_np[c]) * 32
            dst_rows = np.arange(base_pad + old_cnt, base_pad + new_cnt, dtype=np.int64)
            src_rows = ord_ins[s_i : s_i + add_cnt]
            oids = new_orig_ids[src_rows].astype(np.int64)

            if exact_np is not None and perm_np is not None:
                exact_np[dst_rows] = new_vecs[src_rows]
                perm_np[dst_rows] = oids
            else:
                self.exact_vecs_store[dst_rows] = torch.from_numpy(new_vecs[src_rows]).to(self.device)
                self.perm_ids_store[dst_rows] = torch.from_numpy(oids).to(self.device)

            self._orig_to_pad[oids] = dst_rows
            self._slap_shadow[dst_rows, 0] = col0[src_rows]
            self._slap_shadow[dst_rows, 1] = col1[src_rows]
            self._slap_shadow[dst_rows, 2] = col2[src_rows]
            self._slap_shadow[dst_rows, 3] = col3[src_rows]
            self._c_row_counts_np[c] = new_cnt

            first_mod_tile = int(self._c_tile_off_np[c]) + (old_cnt // 32)
            last_mod_tile = int(self._c_tile_off_np[c]) + ((new_cnt - 1) // 32)
            for gt in range(first_mod_tile, last_mod_tile + 1):
                rel_t = gt - int(self._c_tile_off_np[c])
                vc = min(32, new_cnt - rel_t * 32)
                touched_tiles_list.append(gt)
                touched_cids_list.append(c)
                touched_vcnt_list.append(vc)

        # 3. Single sparse-tile GPU kernel launch to quantize & round ball radii upward on touched tiles
        n_touch = len(touched_tiles_list)
        if n_touch > 0:
            t_tiles_np = np.array(touched_tiles_list, dtype=np.int32)
            t_cids_np = np.array(touched_cids_list, dtype=np.int32)
            t_vcnt_np = np.array(touched_vcnt_list, dtype=np.uint16)

            gather_rows = (t_tiles_np[:, None].astype(np.int64) * 32 + np.arange(32, dtype=np.int64)[None, :]).ravel()
            if exact_np is not None:
                v_chunk_g = torch.from_numpy(exact_np[gather_rows].copy()).to(self.device)
            else:
                v_chunk_g = self.exact_vecs_store[gather_rows].contiguous()

            c0_g = torch.from_numpy(self._slap_shadow[gather_rows, 0].copy()).to(self.device)
            c1_g = torch.from_numpy(self._slap_shadow[gather_rows, 1].copy()).to(self.device)
            c2_g = torch.from_numpy(self._slap_shadow[gather_rows, 2].copy()).to(self.device)
            c3_g = torch.from_numpy(self._slap_shadow[gather_rows, 3].copy()).to(self.device)

            if self.sketch_mode >= 1:
                self.ext.encode_prism_sparse_tiles(
                    v_chunk_g, c0_g, c1_g, c2_g, c3_g,
                    torch.from_numpy(t_vcnt_np).to(self.device),
                    torch.from_numpy(t_cids_np).to(self.device),
                    torch.from_numpy(t_tiles_np).to(self.device),
                    self.cluster_o_g, self.cluster_delta_g, self.block_bounds_g,
                    self.N_pad, self.D, self.D_pad, self.M, bool(self.use_sign_bit),
                    self.headers_g, self.slap_cols_g, self.int4_g, self.sign_g, self.ball_meta_g,
                )
            else:
                c0_c = self._slap_shadow[gather_rows, 0]
                c1_c = self._slap_shadow[gather_rows, 1]
                c2_c = self._slap_shadow[gather_rows, 2]
                c3_c = self._slap_shadow[gather_rows, 3]
                slap_u64 = (
                    c0_c.astype(np.uint64)
                    | (c1_c.astype(np.uint64) << np.uint64(16))
                    | (c3_c.astype(np.uint64) << np.uint64(32))
                    | ((c2_c.astype(np.uint64) & np.uint64(63)) << np.uint64(48))
                )
                g_rows_t = torch.from_numpy(gather_rows.astype(np.int64)).to(self.device)
                self.slap_cols_g.index_copy_(0, g_rows_t, torch.from_numpy(slap_u64.view(np.int64)).to(self.device))
                hdr_np = np.zeros((n_touch, 32), dtype=np.uint8)
                v_mask2d = (np.arange(32)[None, :] < t_vcnt_np[:, None])
                c0_2d = c0_c.reshape(n_touch, 32)
                c1_2d = c1_c.reshape(n_touch, 32)
                c2_2d = c2_c.reshape(n_touch, 32)
                c3_2d = c3_c.reshape(n_touch, 32)
                hdr_u16 = hdr_np.view(np.uint16).reshape(n_touch, 16)
                hdr_u16[:, 0] = np.where(v_mask2d, c0_2d, 65535).min(axis=1)
                hdr_u16[:, 1] = np.where(v_mask2d, c0_2d, 0).max(axis=1)
                hdr_u16[:, 2] = np.where(v_mask2d, c1_2d, 65535).min(axis=1)
                hdr_u16[:, 3] = np.where(v_mask2d, c1_2d, 0).max(axis=1)
                hdr_u16[:, 4] = np.where(v_mask2d, c3_2d, 65535).min(axis=1)
                hdr_u16[:, 5] = np.where(v_mask2d, c3_2d, 0).max(axis=1)
                hdr_u16[:, 6] = t_vcnt_np
                blm_bits = np.where(v_mask2d, np.uint64(1) << c2_2d.astype(np.uint64), np.uint64(0))
                hdr_np.view(np.uint64).reshape(n_touch, 4)[:, 2] = np.bitwise_or.reduce(blm_bits, axis=1)
                hdr_np.view(np.uint32).reshape(n_touch, 8)[:, 6] = (t_tiles_np.astype(np.uint32) * 32)
                hdr_np.view(np.int32).reshape(n_touch, 8)[:, 7] = t_cids_np
                t_idx_g = torch.from_numpy(t_tiles_np.astype(np.int64)).to(self.device)
                self.headers_g.index_copy_(0, t_idx_g, torch.from_numpy(hdr_np).to(self.device))
            # Re-apply any tombstones that existed in partially-filled tail tiles before append
            tomb_in_touch = gather_rows[self._tombstone_mask[gather_rows]]
            if tomb_in_touch.size > 0:
                self.ext.mark_deleted_rows(
                    self.headers_g, self.slap_cols_g,
                    torch.from_numpy(tomb_in_touch.astype(np.int64)).to(self.device),
                    self.N_pad,
                )

        self.N += m_new
        torch.cuda.synchronize(self.device)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "insert_ms": float(dt_ms),
            "insert_ips": float(m_new * 1000.0 / max(dt_ms, 1e-6)),
            "touched_tiles": int(n_touch),
        }

    @torch.no_grad()
    def delete_rows(self, orig_ids: np.ndarray) -> Dict[str, float]:
        """O(1) GPU tombstone deletion via SLAP bit-63 & TileHeader flags."""
        assert getattr(self, "_mutable_enabled", False), "Build engine with mutable_headroom > 0 for deletes"
        oids = np.asarray(orig_ids, dtype=np.int64)
        valid_oids = oids[(oids >= 0) & (oids < self._orig_to_pad.shape[0])]
        pad_rows = self._orig_to_pad[valid_oids]
        pad_rows = pad_rows[pad_rows >= 0]
        if pad_rows.size == 0:
            return {"delete_ms": 0.0, "deleted_count": 0}

        self._tombstone_mask[pad_rows] = True
        pad_g = torch.from_numpy(np.ascontiguousarray(pad_rows, dtype=np.int64)).to(self.device)
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        self.ext.mark_deleted_rows(self.headers_g, self.slap_cols_g, pad_g, self.N_pad)
        ev1.record()
        torch.cuda.synchronize(self.device)
        dt_ms = float(ev0.elapsed_time(ev1))
        return {
            "delete_ms": dt_ms,
            "delete_dps": float(pad_rows.size * 1000.0 / max(dt_ms, 1e-6)),
            "deleted_count": int(pad_rows.size),
        }

    @torch.no_grad()
    def search(
        self,
        queries: np.ndarray,
        preds: PredicateBatch,
        k: int = 10,
        nprobe: int = 32,
        regime_override: int = 0,         # 0 = Auto M3, 1 = Global, 2 = Warp-Compacted IVF, 3 = Sketch-First
        enable_warp_compaction: bool = True,
        enable_tile_header: bool = True,
        enable_prewarm: bool = True,
        adaptive_probe_expand: bool = True,
        record_telemetry: bool = False,
        update_cost_model: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        B = queries.shape[0]
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_after_plan = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()

        q_g = torch.from_numpy(queries).to(self.device, dtype=torch.float32)
        op_g = torch.from_numpy(preds.op_flags[:B].view(np.int32)).to(self.device)
        l0_g = torch.from_numpy(preds.l0[:B]).to(self.device)
        r0_g = torch.from_numpy(preds.r0[:B]).to(self.device)
        l1_g = torch.from_numpy(preds.l1[:B]).to(self.device)
        r1_g = torch.from_numpy(preds.r1[:B]).to(self.device)
        cm_g = torch.from_numpy(preds.cat_mask[:B].view(np.int64)).to(self.device)
        w0_g = torch.from_numpy(preds.w0[:B]).to(self.device)
        w3_g = torch.from_numpy(preds.w3[:B]).to(self.device)
        ab_g = torch.from_numpy(preds.alg_bound[:B]).to(self.device)

        # Centroid GEMM for top candidate clusters
        P_cand = min(self.nlist, max(nprobe * (4 if adaptive_probe_expand else 1), nprobe))
        P_cand = min(P_cand, 512)
        qn2 = (q_g * q_g).sum(dim=1, keepdim=True)
        c_dists = qn2 + self.cent_norm2_g - 2.0 * (q_g @ self.centroids_g.T)
        _, top_c = torch.topk(c_dists, k=P_cand, dim=1, largest=False, sorted=True)
        cand_clusters = top_c.to(torch.int32).contiguous()

        # M3 Fast TileHeader Query Plan Estimation
        ordered_probes = torch.empty_like(cand_clusters)
        n_alive_probes = torch.empty((B,), dtype=torch.int32, device=self.device)
        est_global_sel = torch.empty((B,), dtype=torch.float32, device=self.device)
        est_probe_surv = torch.empty((B,), dtype=torch.float32, device=self.device)
        cluster_skew   = torch.empty((B,), dtype=torch.float32, device=self.device)

        if enable_tile_header:
            self.ext.estimate_query_plan(
                self.headers_g, self.cluster_t_off_g, cand_clusters,
                op_g, l0_g, r0_g, l1_g, r1_g, cm_g, w0_g, w3_g, ab_g,
                int(nprobe), bool(enable_prewarm),
                ordered_probes, n_alive_probes,
                est_global_sel, est_probe_surv, cluster_skew,
            )
        else:
            ordered_probes = cand_clusters[:, :nprobe].contiguous()
            n_alive_probes.fill_(int(nprobe))
            est_global_sel.fill_(0.05)
            est_probe_surv.fill_(float(nprobe * (self.N / self.nlist) * 0.05))
            cluster_skew.fill_(0.5)

        out_d = torch.full((B, k), float("inf"), dtype=torch.float32, device=self.device)
        out_i = torch.full((B, k), -1, dtype=torch.int64, device=self.device)
        tel_g = torch.zeros((16,), dtype=torch.int64, device=self.device)
        if record_telemetry:
            ev_after_plan.record()

        def _run_subset(idx_tensor: Optional[torch.Tensor], chosen_regime: int) -> None:
            if idx_tensor is None:
                sub_B = B
                q_s, pr_s, na_s = q_g, ordered_probes, n_alive_probes
                op_s, l0_s, r0_s, l1_s, r1_s = op_g, l0_g, r0_g, l1_g, r1_g
                cm_s, w0_s, w3_s, ab_s = cm_g, w0_g, w3_g, ab_g
            else:
                sub_B = int(idx_tensor.numel())
                if sub_B == 0:
                    return
                q_s = q_g.index_select(0, idx_tensor).contiguous()
                pr_s = ordered_probes.index_select(0, idx_tensor).contiguous()
                na_s = n_alive_probes.index_select(0, idx_tensor).contiguous()
                op_s = op_g.index_select(0, idx_tensor).contiguous()
                l0_s = l0_g.index_select(0, idx_tensor).contiguous()
                r0_s = r0_g.index_select(0, idx_tensor).contiguous()
                l1_s = l1_g.index_select(0, idx_tensor).contiguous()
                r1_s = r1_g.index_select(0, idx_tensor).contiguous()
                cm_s = cm_g.index_select(0, idx_tensor).contiguous()
                w0_s = w0_g.index_select(0, idx_tensor).contiguous()
                w3_s = w3_g.index_select(0, idx_tensor).contiguous()
                ab_s = ab_g.index_select(0, idx_tensor).contiguous()

            if chosen_regime == 1:
                g_split = min(64, max(4, 256 // max(1, sub_B)))
            else:
                g_split = 1 if sub_B >= 96 else min(max(1, nprobe // 2), max(1, 128 // max(1, sub_B)))

            g_tau = torch.empty((sub_B,), dtype=torch.int32, device=self.device)
            part_d = torch.empty((sub_B, g_split, k), dtype=torch.float32, device=self.device) if g_split > 1 else out_d[:1].view(1, 1, k)
            part_i = torch.empty((sub_B, g_split, k), dtype=torch.int64, device=self.device) if g_split > 1 else out_i[:1].view(1, 1, k)
            sub_d = out_d if idx_tensor is None else torch.empty((sub_B, k), dtype=torch.float32, device=self.device)
            sub_i = out_i if idx_tensor is None else torch.empty((sub_B, k), dtype=torch.int64, device=self.device)

            self.ext.launch_prism_search(
                q_s, self.headers_g, self.slap_cols_g,
                self.int4_g, self.sign_g, self.ball_meta_g,
                self.cluster_o_g, self.cluster_delta_g,
                self.cluster_t_off_g, self.block_bounds_g,
                self.exact_vecs_store, self.perm_ids_store,
                pr_s, na_s,
                op_s, l0_s, r0_s, l1_s, r1_s, cm_s, w0_s, w3_s, ab_s,
                self.N_pad, self.D_pad, self.M, k, g_split,
                chosen_regime, self.sketch_mode,
                bool(enable_warp_compaction), bool(enable_tile_header), bool(record_telemetry),
                g_tau, part_d, part_i, sub_d, sub_i, tel_g,
            )
            if idx_tensor is not None:
                out_d.index_copy_(0, idx_tensor, sub_d)
                out_i.index_copy_(0, idx_tensor, sub_i)

        # M3 Three-Regime Selection (supports mixed-selectivity batches W4 seamlessly!)
        regime_counts = {1: 0, 2: 0, 3: 0}
        pred_ms_total = 0.0
        if regime_override in (1, 2, 3):
            regime_counts[regime_override] = B
            m_pr = float(n_alive_probes.float().mean().item())
            m_gs = float(est_global_sel.mean().item())
            m_ps = float(est_probe_surv.mean().item())
            m_sk = float(cluster_skew.mean().item())
            pred_ms_total = self.cost_model.predict_batch_ms(
                regime_override, B, self.N, self.N_wtiles, self.D_pad, self.nlist,
                m_pr, m_gs, m_ps, m_sk,
            )
            _run_subset(None, regime_override)
        else:
            # Per-query analytical regime selection on GPU
            sel_cpu = est_global_sel.cpu().numpy()
            surv_cpu = est_probe_surv.cpu().numpy()
            nalive_cpu = n_alive_probes.cpu().numpy()
            skew_cpu = cluster_skew.cpu().numpy()

            # Compute analytical cost for each query across Regime 1, 2, 3
            reg_choice = np.full(B, 2, dtype=np.int32)
            for qi in range(B):
                s_q = float(sel_cpu[qi])
                ps_q = float(surv_cpu[qi])
                np_q = float(nalive_cpu[qi])
                sk_q = float(skew_cpu[qi])
                c1 = self.cost_model.predict_batch_ms(1, 1, self.N, self.N_wtiles, self.D_pad, self.nlist, np_q, s_q, ps_q, sk_q)
                c2 = self.cost_model.predict_batch_ms(2, 1, self.N, self.N_wtiles, self.D_pad, self.nlist, np_q, s_q, ps_q, sk_q)
                c3 = self.cost_model.predict_batch_ms(3, 1, self.N, self.N_wtiles, self.D_pad, self.nlist, np_q, s_q, ps_q, sk_q)

                # Guard Regime 1 for ultra-low selectivity where IVF recall collapses (s_q < sel_low_thresh or ps_q < 2.5*k)
                if s_q < self.cost_model.sel_low_thresh or (ps_q < 2.5 * k and s_q < 0.015):
                    reg_choice[qi] = 1
                elif s_q > self.cost_model.sel_high_thresh and c3 <= c2 * 1.05 and self.sketch_mode >= 1:
                    reg_choice[qi] = 3
                else:
                    reg_choice[qi] = 1 if (c1 < c2 * 0.85 and s_q < 0.01) else 2

            reg_g = torch.from_numpy(reg_choice).to(self.device)
            if (reg_choice == reg_choice[0]).all():
                r_all = int(reg_choice[0])
                regime_counts[r_all] = B
                pred_ms_total = self.cost_model.predict_batch_ms(
                    r_all, B, self.N, self.N_wtiles, self.D_pad, self.nlist,
                    float(nalive_cpu.mean()), float(sel_cpu.mean()), float(surv_cpu.mean()), float(skew_cpu.mean())
                )
                _run_subset(None, r_all)
            else:
                for r_id in (1, 2, 3):
                    idx_r = torch.nonzero(reg_g == r_id, as_tuple=False).squeeze(1)
                    cnt_r = int(idx_r.numel())
                    regime_counts[r_id] = cnt_r
                    if cnt_r > 0:
                        mask_np = (reg_choice == r_id)
                        pred_ms_total += self.cost_model.predict_batch_ms(
                            r_id, cnt_r, self.N, self.N_wtiles, self.D_pad, self.nlist,
                            float(nalive_cpu[mask_np].mean()), float(sel_cpu[mask_np].mean()),
                            float(surv_cpu[mask_np].mean()), float(skew_cpu[mask_np].mean())
                        )
                        _run_subset(idx_r, r_id)

        ids_cpu = out_i.cpu().numpy()
        dists_cpu = out_d.cpu().numpy()
        ev_end.record()
        torch.cuda.synchronize(self.device)
        elapsed_ms = float(ev_start.elapsed_time(ev_end))

        if update_cost_model and regime_override in (1, 2, 3):
            par_factor = max(1.0, B / 48.0) if B >= 16 else max(0.35, B / 16.0)
            d_scale = self.D_pad / 128.0
            if regime_override == 1:
                feat = [par_factor, par_factor * self.N_wtiles, par_factor * (m_gs * self.N) * d_scale]
            elif regime_override == 2:
                ptiles = (m_pr / max(1, self.nlist)) * self.N_wtiles
                feat = [par_factor, par_factor * ptiles, par_factor * m_ps * d_scale, par_factor * m_sk * (m_pr / 32.0)]
            else:
                prows = (m_pr / max(1, self.nlist)) * self.N
                feat = [par_factor, par_factor * prows * d_scale, par_factor * (0.28 * m_ps) * d_scale]
            self.cost_model.records.append({
                "regime": int(regime_override),
                "B": int(B),
                "feat": feat,
                "pred_ms": float(pred_ms_total),
                "actual_ms": float(elapsed_ms),
            })
            self.cost_model.nlms_update(regime_override, pred_ms_total, elapsed_ms)

        info: Dict[str, float] = {
            "latency_ms": elapsed_ms,
            "qps": (B * 1000.0) / max(elapsed_ms, 1e-6),
            "pred_ms": float(pred_ms_total),
            "r1_queries": regime_counts[1],
            "r2_queries": regime_counts[2],
            "r3_queries": regime_counts[3],
        }

        if record_telemetry:
            t_arr = tel_g.cpu().numpy()
            pred_warps = max(1, int(t_arr[3]))
            pred_survs = int(t_arr[4])
            comp_warps = max(1, int(t_arr[5]))
            plan_ms = float(ev_start.elapsed_time(ev_after_plan))
            kern_ms = max(0.0, elapsed_ms - plan_ms)
            cyc_flt = max(0, int(t_arr[11]))
            cyc_sk  = max(0, int(t_arr[12]))
            cyc_ref = max(0, int(t_arr[13]))
            cyc_tot = max(1, cyc_flt + cyc_sk + cyc_ref)
            info.update({
                "stage_plan_ms": plan_ms,
                "stage_filter_ms": kern_ms * (cyc_flt / cyc_tot),
                "stage_sketch_ms": kern_ms * (cyc_sk / cyc_tot),
                "stage_refine_ms": kern_ms * (cyc_ref / cyc_tot),
                "tiles_visited": int(t_arr[0]),
                "tiles_skipped_header": int(t_arr[1]),
                "tiles_full_hit": int(t_arr[2]),
                "pred_survivors": pred_survs,
                "uncompacted_active_lane_pct": 100.0 * pred_survs / (32.0 * pred_warps),
                "compacted_active_lane_pct": 100.0 * pred_survs / (32.0 * comp_warps),
                "surv_after_b0": int(t_arr[6]),
                "surv_after_b1": int(t_arr[7]),
                "surv_after_b2": int(t_arr[8]),
                "surv_after_b3": int(t_arr[9]),
                "host_fp32_fetches": int(t_arr[10]),
                "prune_ratio_b0_pct": 100.0 * (1.0 - t_arr[6] / max(1, pred_survs)),
                "prune_ratio_b1_pct": 100.0 * (1.0 - t_arr[7] / max(1, pred_survs)),
                "prune_ratio_b2_pct": 100.0 * (1.0 - t_arr[8] / max(1, pred_survs)),
                "prune_ratio_b3_pct": 100.0 * (1.0 - t_arr[9] / max(1, pred_survs)),
                "overall_sketch_prune_pct": 100.0 * (1.0 - t_arr[10] / max(1, pred_survs)),
            })

        return ids_cpu, dists_cpu, info
