#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark E14: Head-to-Head Evaluation Against Upstream Official Open-Source CPU Filtered ANNS Engines:
  1. Official ACORN (`guestrin-lab/ACORN` — `faiss::IndexACORNFlat`, SIGMOD'24, 48T C++ OpenMP):
     - `Official_ACORN_1` (`M=32, gamma=1, M_beta=64`, full 2-hop expansion, 135K-10M)
     - `Official_ACORN_gamma` (`M=32, gamma=24, M_beta=32`, compressed 2-hop graph, 135K-1M)
     - `Official_ACORN` selects the highest-QPS official ACORN variant achieving Recall@10 >= 0.95
       (or highest recall when both cap below 0.95).
  2. Official SeRF-2D (`rutgers-db/SeRF` — `SeRF::IndexSegmentGraph2D`, SIGMOD'24, 48T C++ OpenMP)
  3. Official Filtered-DiskANN (`microsoft/DiskANN` `cpp_main` — `StitchedVamana` `build_stitched_index`
     + `search_memory_index` 48T C++ OpenMP, WWW'23, 135K-10M)
Evaluated on identical disjoint splits (`Q_val=256`, `Q_test=256`, seed=2027) with 7-run trimmed mean.
"""

from __future__ import annotations
import argparse
import gc
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from typing import Callable, Dict, List, Tuple
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "baselines")))

from src.hybrid_datasets import (
    HybridDataset,
    PredicateBatch,
    PRED_COL1_RANGE,
    PRED_COL2_CAT,
    load_hybrid_dataset,
    generate_sql_workload,
)
from src.oracle import (
    compute_exact_filtered_gt,
    compute_filtered_recall_at_k,
    eval_predicates_gpu_chunk,
)


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def trimmed_mean_eval(fn: Callable, warmup: int = 2, repeats: int = 7, sync_gpu: bool = False):
    for _ in range(warmup):
        out = fn()
    lats = []
    out = None
    for _ in range(repeats):
        if sync_gpu:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if sync_gpu:
            torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1000.0)
    s = sorted(lats)
    mid = s[1:-1] if len(s) >= 5 else s
    return out, float(np.mean(mid))


def make_label_workloads(ds: HybridDataset, n_queries: int, seed: int = 2027) -> Tuple[PredicateBatch, np.ndarray, PredicateBatch, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    vals, cnts = np.unique(ds.col2, return_counts=True)
    good_cats = vals[cnts >= min(500, max(50, ds.N // 400))]
    if len(good_cats) == 0:
        good_cats = vals

    q_lbl_a = rng.choice(good_cats, size=n_queries, replace=True).astype(np.int32)
    z = np.zeros(n_queries, dtype=np.int32)
    cat_mask_1 = (np.uint64(1) << q_lbl_a.astype(np.uint64))
    preds_wl1 = PredicateBatch(
        op_flags=np.full(n_queries, PRED_COL2_CAT, dtype=np.uint32),
        l0=z.copy(), r0=z.copy(), l1=z.copy(), r1=z.copy(),
        cat_mask=cat_mask_1,
        w0=z.copy(), w3=z.copy(), alg_bound=z.copy(),
        target_sel=np.full(n_queries, 1.0 / max(1, len(good_cats)), dtype=np.float32),
        workload_name="WL1_single_label",
    )

    q_lbl_a2 = rng.choice(good_cats, size=n_queries, replace=True).astype(np.int32)
    q_bucket_b = rng.integers(0, 8, size=n_queries, dtype=np.int32)
    l1 = (q_bucket_b * 8192).astype(np.int32)
    r1 = np.minimum(65535, (q_bucket_b + 1) * 8192 - 1).astype(np.int32)
    cat_mask_2 = (np.uint64(1) << q_lbl_a2.astype(np.uint64))
    preds_wl2 = PredicateBatch(
        op_flags=np.full(n_queries, PRED_COL2_CAT | PRED_COL1_RANGE, dtype=np.uint32),
        l0=z.copy(), r0=z.copy(), l1=l1, r1=r1,
        cat_mask=cat_mask_2,
        w0=z.copy(), w3=z.copy(), alg_bound=z.copy(),
        target_sel=np.full(n_queries, 0.008, dtype=np.float32),
        workload_name="WL2_two_label_and",
    )
    q_lbl_b2 = (64 + q_bucket_b).astype(np.int32)
    return preds_wl1, q_lbl_a, preds_wl2, q_lbl_a2, q_lbl_b2


@torch.no_grad()
def build_filter_bitmap_uint8(ds: HybridDataset, preds: PredicateBatch) -> np.ndarray:
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
    free_gpu()
    return out


def calibrate_knob(eval_fn: Callable[[int], Tuple[np.ndarray, np.ndarray]], gt_i: np.ndarray, gt_d: np.ndarray, knobs: List[int], target: float = 0.960) -> Tuple[int, float]:
    best_k = knobs[-1]
    best_r = 0.0
    for kn in knobs:
        pi, pd = eval_fn(kn)
        r = compute_filtered_recall_at_k(pi, gt_i, gt_d, pd, k=10)
        if r >= best_r:
            best_r = r
            best_k = kn
        if r >= target:
            return kn, r
    return best_k, best_r


def write_fbin(path: str, arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    n, d = arr.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", n, d))
        arr.tofile(f)


def read_u32_bin(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        data = np.fromfile(f, dtype=np.uint32, count=n * d)
    return data.reshape(n, d).astype(np.int64)


def read_f32_bin(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        data = np.fromfile(f, dtype=np.float32, count=n * d)
    return data.reshape(n, d)


def run_stitched_diskann_wl1(ds: HybridDataset, xb_f32: np.ndarray, xq_val: np.ndarray, xq_test: np.ndarray,
                             q_lbl_a: np.ndarray, gvi: np.ndarray, gvd: np.ndarray, gti: np.ndarray, gtd: np.ndarray,
                             n_val: int, n_test: int) -> Dict:
    prism_root = os.path.expanduser(
        os.environ.get("PRISM_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    )
    bin_dir = os.path.expanduser(
        os.environ.get("DISKANN_BIN_DIR", os.path.join(prism_root, "third_party", "DiskANN", "build", "apps"))
    )
    build_stitched_bin = os.path.join(bin_dir, "build_stitched_index")
    search_bin = os.path.join(bin_dir, "search_memory_index")
    n_tot = n_val + n_test

    R_sub = "48" if ds.N < 500_000 else "32"
    L_sub = "128" if ds.N < 500_000 else "100"

    tmp_d = tempfile.mkdtemp(prefix=f"da_sv_{ds.name}_")
    try:
        base_fbin = os.path.join(tmp_d, "base.fbin")
        qval_fbin = os.path.join(tmp_d, "qval.fbin")
        qtst_fbin = os.path.join(tmp_d, "qtst.fbin")
        lbl_file = os.path.join(tmp_d, "base_labels.txt")
        qval_lbl = os.path.join(tmp_d, "qval_labels.txt")
        qtst_lbl = os.path.join(tmp_d, "qtst_labels.txt")
        idx_pref = os.path.join(tmp_d, "sv")
        res_pref = os.path.join(tmp_d, "res")

        write_fbin(base_fbin, xb_f32)
        write_fbin(qval_fbin, xq_val)
        write_fbin(qtst_fbin, xq_test)
        with open(lbl_file, "w") as f:
            f.write("\n".join(str(int(v) + 1) for v in ds.col2) + "\n")
        with open(qval_lbl, "w") as f:
            f.write("\n".join(str(int(v) + 1) for v in q_lbl_a[:n_val]) + "\n")
        with open(qtst_lbl, "w") as f:
            f.write("\n".join(str(int(v) + 1) for v in q_lbl_a[n_val:n_tot]) + "\n")

        t0 = time.perf_counter()
        subprocess.run(
            [
                build_stitched_bin,
                "--data_type", "float",
                "--data_path", base_fbin,
                "--index_path_prefix", idx_pref,
                "-R", R_sub,
                "-L", L_sub,
                "--stitched_R", "64",
                "--alpha", "1.2",
                "-T", "48",
                "--label_file", lbl_file,
                "--universal_label", "0",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        build_s = time.perf_counter() - t0
        for suf in ["_labels.txt", "_labels_to_medoids.txt", "_universal_label.txt"]:
            if os.path.exists(f"{idx_pref}_full{suf}") and not os.path.exists(f"{idx_pref}{suf}"):
                shutil.copyfile(f"{idx_pref}_full{suf}", f"{idx_pref}{suf}")

        L_grid = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
        subprocess.run(
            [
                search_bin,
                "--data_type", "float",
                "--dist_fn", "l2",
                "--index_path_prefix", idx_pref,
                "--result_path", res_pref + "_val",
                "--query_file", qval_fbin,
                "--query_filters_file", qval_lbl,
                "-K", "10",
                "-T", "48",
                "-L", *[str(x) for x in L_grid],
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        best_L, best_rv = L_grid[-1], 0.0
        for Lv in L_grid:
            pi = read_u32_bin(f"{res_pref}_val_{Lv}_idx_uint32.bin")
            pd = read_f32_bin(f"{res_pref}_val_{Lv}_dists_float.bin")
            rv = compute_filtered_recall_at_k(pi, gvi, gvd, pd, k=10)
            if rv >= best_rv:
                best_rv, best_L = rv, Lv
            if rv >= 0.960:
                best_L = Lv
                break

        L_seq = [str(best_L)] * 9
        proc = subprocess.run(
            [
                search_bin,
                "--data_type", "float",
                "--dist_fn", "l2",
                "--index_path_prefix", idx_pref,
                "--result_path", res_pref + "_tst",
                "--query_file", qtst_fbin,
                "--query_filters_file", qtst_lbl,
                "-K", "10",
                "-T", "48",
                "-L", *L_seq,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        qps_runs = []
        for line in proc.stdout.splitlines():
            m = re.match(r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", line)
            if m and int(m.group(1)) == best_L:
                qps_runs.append(float(m.group(2)))
        timed = sorted(qps_runs[2:])[1:-1] if len(qps_runs) >= 7 else qps_runs
        mean_qps = float(np.mean(timed))
        pi_t = read_u32_bin(f"{res_pref}_tst_{best_L}_idx_uint32.bin")
        pd_t = read_f32_bin(f"{res_pref}_tst_{best_L}_dists_float.bin")
        rec_t = compute_filtered_recall_at_k(pi_t, gti, gtd, pd_t, k=10)
        return {
            "qps": round(mean_qps, 2),
            "recall10": round(rec_t, 4),
            "latency_ms": round((n_test * 1000.0) / mean_qps, 3),
            "L_search": int(best_L),
            "build_s": round(build_s, 3),
            "variant": f"StitchedVamana(R={R_sub},L={L_sub},SR=64)",
        }
    finally:
        shutil.rmtree(tmp_d, ignore_errors=True)


def run_cpu_suite(out_path: str, target_datasets: List[str]):
    import official_fanns_ext

    n_val, n_test = 256, 256
    n_tot = n_val + n_test

    for ds_name in target_datasets:
        print(f"\n==================== E14 Official CPU Baselines: {ds_name} ====================", flush=True)
        ds = load_hybrid_dataset(ds_name, max_queries=n_tot)
        xb_f32 = np.ascontiguousarray(ds.base_vectors, dtype=np.float32)
        xq_val = np.ascontiguousarray(ds.query_vectors[:n_val], dtype=np.float32)
        xq_test = np.ascontiguousarray(ds.query_vectors[n_val:n_tot], dtype=np.float32)

        wls: Dict[str, PredicateBatch] = {
            "W1_range": generate_sql_workload(ds, "W1_range", num_queries=n_tot, selectivity=0.05),
            "W2_correlated": generate_sql_workload(ds, "W2_correlated", num_queries=n_tot, selectivity=0.02),
            "W3_compound": generate_sql_workload(ds, "W3_compound", num_queries=n_tot, selectivity=0.02),
            "W4_mixed": generate_sql_workload(ds, "W4_mixed", num_queries=n_tot, selectivity=0.05),
        }
        preds_wl1, q_lbl_a, preds_wl2, _, _ = make_label_workloads(ds, n_tot)
        wls["WL1_single_label"] = preds_wl1
        wls["WL2_two_label_and"] = preds_wl2

        gts = {}
        for wname, pb in wls.items():
            gi, gd, _ = compute_exact_filtered_gt(ds, pb, k=10)
            gts[wname] = (gi[:n_val], gd[:n_val], gi[n_val:n_tot], gd[n_val:n_tot])

        def load_cur():
            if os.path.exists(out_path):
                try:
                    c = json.load(open(out_path))
                    if c.get("n_val") == n_val:
                        return c
                except Exception:
                    pass
            return {"n_val": n_val, "n_test": n_test, "datasets": {}}

        def save_cur(c):
            with open(out_path, "w") as f:
                json.dump(c, f, indent=2)

        # ------------------------------------------------------------------
        # 1. Official microsoft/DiskANN C++ (`StitchedVamana`, 48T OpenMP) on WL1
        # ------------------------------------------------------------------
        cur = load_cur()
        ds_e = cur["datasets"].setdefault(ds_name, {"N": ds.N, "D": ds.D, "build": {}, "workloads": {}})
        if "Official_DiskANN" not in ds_e["workloads"].get("WL1_single_label", {}):
            print(f"  [Building Official C++ StitchedVamana (microsoft/DiskANN) on {ds_name}] ...", flush=True)
            gvi, gvd, gti, gtd = gts["WL1_single_label"]
            da_res = run_stitched_diskann_wl1(ds, xb_f32, xq_val, xq_test, q_lbl_a, gvi, gvd, gti, gtd, n_val, n_test)
            cur = load_cur()
            ds_e = cur["datasets"].setdefault(ds_name, {"N": ds.N, "D": ds.D, "build": {}, "workloads": {}})
            ds_e["build"]["Official_DiskANN"] = {
                "build_s": da_res["build_s"],
                "repo": f"microsoft/DiskANN C++ 48T ({da_res['variant']})",
            }
            ds_e["workloads"].setdefault("WL1_single_label", {})["Official_DiskANN"] = da_res
            for w_na in ["W1_range", "W2_correlated", "W3_compound", "W4_mixed", "WL2_two_label_and"]:
                ds_e["workloads"].setdefault(w_na, {})["Official_DiskANN"] = {"status": "N/A (single discrete label API)"}
            save_cur(cur)
            print(f"  [Official StitchedVamana | {ds_name}] build={da_res['build_s']:.2f}s, L={da_res['L_search']}, QPS={da_res['qps']:.1f}, R@10={da_res['recall10']:.4f}", flush=True)

        # ------------------------------------------------------------------
        # 2. Official guestrin-lab/ACORN (`ACORN-1` everywhere + `ACORN-gamma` on <=1M)
        # ------------------------------------------------------------------
        cur = load_cur()
        ds_e = cur["datasets"].setdefault(ds_name, {"N": ds.N, "D": ds.D, "build": {}, "workloads": {}})
        if "Official_ACORN" not in ds_e["build"]:
            meta_i32 = np.ascontiguousarray(ds.col2.astype(np.int32))
            acorn_variants = [("Official_ACORN_1", 32, 1, 64, 80)]
            if ds.N <= 1_000_000:
                acorn_variants.append(("Official_ACORN_gamma", 32, 24, 32, 80))

            var_results: Dict[str, Dict] = {}
            var_builds: Dict[str, float] = {}
            acorn_knobs = [16, 32, 64, 128, 256, 512, 1024] if ds.N > 1_000_000 else [16, 32, 64, 128, 256, 512, 1024, 1536]

            # Precompute filter bitmaps one workload at a time to bound host RAM on 10M
            for vtag, M_p, gam_p, mb_p, efc_p in acorn_variants:
                print(f"  [Building {vtag} (M={M_p}, gamma={gam_p}, Mb={mb_p}) on {ds_name}] ...", flush=True)
                t0 = time.perf_counter()
                ac = official_fanns_ext.OfficialACORNIndex(ds.D, M_p, gam_p, meta_i32, mb_p)
                ac.build(xb_f32, efc_p, 48)
                b_s = time.perf_counter() - t0
                var_builds[vtag] = round(b_s, 3)
                var_results[vtag] = {}
                print(f"  [{vtag} Built in {b_s:.2f}s]", flush=True)

                for wname, pb in wls.items():
                    fmap_val = np.ascontiguousarray(build_filter_bitmap_uint8(ds, pb.slice(0, n_val)))
                    fmap_test = np.ascontiguousarray(build_filter_bitmap_uint8(ds, pb.slice(n_val, n_tot)))
                    gvi, gvd, gti, gtd = gts[wname]
                    best_ef, _ = calibrate_knob(lambda ef: ac.search(xq_val, fmap_val, 10, ef, 48), gvi, gvd, acorn_knobs, 0.960)
                    (ai, ad), s_ms = trimmed_mean_eval(lambda: ac.search(xq_test, fmap_test, 10, best_ef, 48), warmup=1, repeats=5)
                    rec = compute_filtered_recall_at_k(ai, gti, gtd, ad, k=10)
                    var_results[vtag][wname] = {
                        "qps": round((n_test * 1000.0) / s_ms, 2),
                        "recall10": round(rec, 4),
                        "latency_ms": round(s_ms, 3),
                        "efSearch": int(best_ef),
                        "variant": vtag,
                        "build_s": round(b_s, 3),
                    }
                    print(f"    [{vtag} | {wname}] ef={best_ef}, QPS={var_results[vtag][wname]['qps']:.1f}, R@10={rec:.4f}", flush=True)
                    del fmap_val, fmap_test
                    gc.collect()
                del ac
                gc.collect()

            cur = load_cur()
            ds_e = cur["datasets"].setdefault(ds_name, {"N": ds.N, "D": ds.D, "build": {}, "workloads": {}})
            for vtag, b_s in var_builds.items():
                ds_e["build"][vtag] = {"build_s": b_s, "repo": "guestrin-lab/ACORN (faiss::IndexACORNFlat)"}
            # Primary `Official_ACORN` build reports ACORN-1 on 10M and calibrated ACORN-1 / ACORN-gamma envelope on <=1M
            ds_e["build"]["Official_ACORN"] = {
                "build_s": var_builds.get("Official_ACORN_gamma", var_builds["Official_ACORN_1"]),
                "build_s_acorn1": var_builds["Official_ACORN_1"],
                "repo": "guestrin-lab/ACORN (faiss::IndexACORNFlat)",
            }
            for wname in wls:
                w_dict = ds_e["workloads"].setdefault(wname, {})
                cands = []
                for vtag in var_results:
                    w_dict[vtag] = var_results[vtag][wname]
                    cands.append(var_results[vtag][wname])
                # Best official ACORN: if any hits >= 0.950, pick fastest QPS among >= 0.950; else pick highest recall
                r95_cands = [c for c in cands if c["recall10"] >= 0.950]
                best_c = max(r95_cands, key=lambda c: c["qps"]) if r95_cands else max(cands, key=lambda c: (c["recall10"], c["qps"]))
                w_dict["Official_ACORN"] = dict(best_c)
            save_cur(cur)

        # ------------------------------------------------------------------
        # 3. Official rutgers-db/SeRF (`SeRF::IndexSegmentGraph2D`, 48T) on W1 & W2 (<=1M) + W1 on deep10m
        # ------------------------------------------------------------------
        cur = load_cur()
        ds_e = cur["datasets"].setdefault(ds_name, {"N": ds.N, "D": ds.D, "build": {}, "workloads": {}})
        serf_targets = [("W1_range", ds.col0, True), ("W2_correlated", ds.col1, False)] if ds.N <= 1_000_000 else (
            [("W1_range", ds.col0, True)] if ds_name == "deep10m" else []
        )
        for wname, col_arr, is_c0 in serf_targets:
            if "Official_SeRF" in ds_e["workloads"].get(wname, {}):
                continue
            print(f"  [Building Official rutgers-db/SeRF on {ds_name} ({wname})] ...", flush=True)
            try:
                attr_f32 = np.ascontiguousarray(col_arr.astype(np.float32))
                t0 = time.perf_counter()
                serf = official_fanns_ext.OfficialSeRF2DIndex(xb_f32, attr_f32, 32, 120, 1200, 48)
                serf_build_s = time.perf_counter() - t0
                pb = wls[wname]
                lo_all = (pb.l0 if is_c0 else pb.l1).astype(np.float32)
                hi_all = (pb.r0 if is_c0 else pb.r1).astype(np.float32)
                gvi, gvd, gti, gtd = gts[wname]
                serf_knobs = [32, 64, 128, 256, 512, 800, 1200]
                best_ef, _ = calibrate_knob(
                    lambda ef: serf.search_range(xq_val, lo_all[:n_val], hi_all[:n_val], 10, ef, 48),
                    gvi, gvd, serf_knobs, 0.960
                )
                (si, sd), s_ms = trimmed_mean_eval(
                    lambda: serf.search_range(xq_test, lo_all[n_val:n_tot], hi_all[n_val:n_tot], 10, best_ef, 48),
                    warmup=1, repeats=5
                )
                rec = compute_filtered_recall_at_k(si, gti, gtd, sd, k=10)
                cur = load_cur()
                ds_e = cur["datasets"].setdefault(ds_name, {"N": ds.N, "D": ds.D, "build": {}, "workloads": {}})
                ds_e["build"][f"Official_SeRF_{wname}"] = {"build_s": round(serf_build_s, 3), "repo": "rutgers-db/SeRF (IndexSegmentGraph2D)"}
                ds_e["workloads"].setdefault(wname, {})["Official_SeRF"] = {
                    "qps": round((n_test * 1000.0) / s_ms, 2),
                    "recall10": round(rec, 4),
                    "latency_ms": round(s_ms, 3),
                    "efSearch": int(best_ef),
                    "build_s": round(serf_build_s, 3),
                }
                save_cur(cur)
                print(f"  [Official SeRF | {ds_name} | {wname}] build={serf_build_s:.1f}s, ef={best_ef}, QPS={ds_e['workloads'][wname]['Official_SeRF']['qps']:.1f}, R@10={rec:.4f}", flush=True)
                del serf
                gc.collect()
            except Exception as e_serf:
                print(f"  [Official SeRF | {ds_name} | {wname}] FAILED: {e_serf}", flush=True)
                cur = load_cur()
                ds_e = cur["datasets"].setdefault(ds_name, {"N": ds.N, "D": ds.D, "build": {}, "workloads": {}})
                ds_e["workloads"].setdefault(wname, {})["Official_SeRF"] = {"status": f"FAILED: {e_serf}"}
                save_cur(cur)

        cur = load_cur()
        ds_e = cur["datasets"].setdefault(ds_name, {"N": ds.N, "D": ds.D, "build": {}, "workloads": {}})
        for w_na in ["W3_compound", "W4_mixed", "WL1_single_label", "WL2_two_label_and"]:
            ds_e["workloads"].setdefault(w_na, {})["Official_SeRF"] = {"status": "N/A (1D range-dedicated index)"}
        save_cur(cur)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "e14_official_baselines.json")))
    ap.add_argument("--datasets", nargs="+", default=["modbench", "sift1m", "gist1m", "convlog1m", "deep10m", "sift10m"])
    args = ap.parse_args()
    run_cpu_suite(args.out, args.datasets)
