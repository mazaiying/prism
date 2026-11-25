#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Official Supercomputing-System-AI-Lab/VecFlow (`vecflow-cu12 0.2.0`, SIGMOD'26) Benchmark.
Evaluates Official `VecFlow` (`from vecflow import VecFlow`) on:
  - `WL1_single_label` (`vecflow.search`: single discrete label `col2 == lbl_a`)
  - `WL2_two_label_and` (`vecflow.search_multi`: 2-label AND `col2 == lbl_a AND bucket8(col1) == lbl_b`)
across `modbench`, `sift1m`, `gist1m`, `convlog1m`, `deep10m`, `sift10m`, `t2i10m`
using identical query splits (`Q_val=256`, `Q_test=256`), exact ground-truth Recall@10 >= 0.95
calibration (validation margin 0.960), CUDA-context-warmed pure index VRAM measurement,
and 7-run trimmed mean.
"""

from __future__ import annotations
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.hybrid_datasets import load_hybrid_dataset
from vecflow import VecFlow


def get_gpu_used_mib() -> float:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "1").split(",")[0]
    out = subprocess.check_output(
        ["nvidia-smi", "-i", str(vis), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return float(out.strip().splitlines()[0])


def warmup_cuda_context() -> float:
    """Initialize CUDA + libcuvs/RMM runtime context with a tiny 512-row dummy index and free it."""
    tmp_w = tempfile.mkdtemp(prefix="vf_warm_")
    try:
        xw = np.random.default_rng(0).standard_normal((512, 32), dtype=np.float32)
        lw = [[int(i % 4)] for i in range(512)]
        vfw = VecFlow()
        vfw.build(xw, lw, 16, 100, os.path.join(tmp_w, "g.bin"), os.path.join(tmp_w, "b.bin"), True, True)
        ql = np.zeros(8, dtype=np.int32)
        vfw.search(xw[:8], ql, 16, 10)
        del vfw
    finally:
        shutil.rmtree(tmp_w, ignore_errors=True)
        gc.collect()
    time.sleep(0.15)
    return get_gpu_used_mib()


def recall_at_10(pred_i: np.ndarray, gt_i: np.ndarray, gt_d: np.ndarray, pred_d: np.ndarray) -> float:
    B = pred_i.shape[0]
    recs = []
    for i in range(B):
        vg = gt_i[i, :10]
        vg = vg[vg >= 0]
        den = len(vg)
        if den == 0:
            recs.append(1.0)
            continue
        got = set(int(x) for x in pred_i[i, :den] if 0 <= int(x) < 2_000_000_000)
        hit = len(got.intersection(int(x) for x in vg))
        if hit < den:
            kth = float(gt_d[i, den - 1])
            tol = max(1e-4, abs(kth) * 2e-5)
            gset = set(int(x) for x in vg)
            for j in range(den):
                pid = int(pred_i[i, j])
                if 0 <= pid < 2_000_000_000 and pid not in gset and float(pred_d[i, j]) <= kth + tol:
                    hit += 1
            hit = min(den, hit)
        recs.append(hit / float(den))
    return float(np.mean(recs))


def trimmed_eval(fn, warmup=2, repeats=7):
    for _ in range(warmup):
        out = fn()
    lats = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        lats.append((time.perf_counter() - t0) * 1000.0)
    s = sorted(lats)
    return out, float(np.mean(s[1:-1]))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "e14_official_vecflow.json"))
    res = {
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "1"),
        "repo": "Supercomputing-System-AI-Lab/VecFlow (vecflow-cu12 0.2.0)",
        "n_val": 256,
        "n_test": 256,
        "datasets": {},
    }
    if os.path.exists(out_path):
        try:
            old = json.load(open(out_path))
            if old.get("n_val") == 256 and not args.force:
                res = old
            elif not args.force and args.dataset != "all":
                res["datasets"] = old.get("datasets", {})
                res["n_val"] = 256
                res["n_test"] = 256
        except Exception:
            pass

    n_val, n_test = 256, 256
    n_tot = n_val + n_test

    ds_list = ["modbench", "sift1m", "gist1m", "convlog1m", "deep10m", "sift10m", "t2i10m"] if args.dataset == "all" else [args.dataset]
    for ds_name in ds_list:
        if not args.force and ds_name in res["datasets"] and res["datasets"][ds_name].get("n_val") == 256:
            print(f"[Skip completed Official VecFlow (n_val=256)] {ds_name}", flush=True)
            continue
        print(f"\n==================== Official VecFlow: {ds_name} ====================", flush=True)
        ds = load_hybrid_dataset(ds_name, max_queries=n_tot)
        xb = np.ascontiguousarray(ds.base_vectors, dtype=np.float32)
        xq_val = np.ascontiguousarray(ds.query_vectors[:n_val], dtype=np.float32)
        xq_test = np.ascontiguousarray(ds.query_vectors[n_val:n_tot], dtype=np.float32)

        rng = np.random.default_rng(2027)
        vals, cnts = np.unique(ds.col2, return_counts=True)
        good_cats = vals[cnts >= min(500, max(50, ds.N // 400))]
        if len(good_cats) == 0:
            good_cats = vals
        q_lbl_a = rng.choice(good_cats, size=n_tot, replace=True).astype(np.int32)
        q_lbl_a2 = rng.choice(good_cats, size=n_tot, replace=True).astype(np.int32)
        q_bucket_b = rng.integers(0, 8, size=n_tot, dtype=np.int32)
        q_lbl_b2 = (64 + q_bucket_b).astype(np.int32)

        # Exact ground truth via inverted label lists
        b_col = (ds.col1 // 8192).astype(np.int32)
        gt1_i = np.full((n_tot, 10), -1, dtype=np.int64)
        gt1_d = np.full((n_tot, 10), 1e30, dtype=np.float32)
        for lbl in np.unique(q_lbl_a):
            q_idx = np.where(q_lbl_a == lbl)[0]
            cand_idx = np.where(ds.col2 == lbl)[0]
            if len(cand_idx) == 0:
                continue
            xc = xb[cand_idx]
            xq_sub = ds.query_vectors[:n_tot][q_idx]
            d2 = np.maximum(0.0, np.sum(xq_sub * xq_sub, axis=1, keepdims=True) + np.sum(xc * xc, axis=1)[None, :] - 2.0 * (xq_sub @ xc.T))
            m_take = min(10, len(cand_idx))
            part = np.argpartition(d2, m_take - 1, axis=1)[:, :m_take]
            pd = np.take_along_axis(d2, part, axis=1)
            ord_i = np.argsort(pd, axis=1)
            gt1_d[q_idx, :m_take] = np.take_along_axis(pd, ord_i, axis=1)
            gt1_i[q_idx, :m_take] = cand_idx[np.take_along_axis(part, ord_i, axis=1)]

        gt2_i = np.full((n_tot, 10), -1, dtype=np.int64)
        gt2_d = np.full((n_tot, 10), 1e30, dtype=np.float32)
        pairs = np.unique(np.stack([q_lbl_a2, q_bucket_b], axis=1), axis=0)
        for la, lb in pairs:
            q_idx = np.where((q_lbl_a2 == la) & (q_bucket_b == lb))[0]
            cand_idx = np.where((ds.col2 == la) & (b_col == lb))[0]
            if len(cand_idx) == 0:
                continue
            xc = xb[cand_idx]
            xq_sub = ds.query_vectors[:n_tot][q_idx]
            d2 = np.maximum(0.0, np.sum(xq_sub * xq_sub, axis=1, keepdims=True) + np.sum(xc * xc, axis=1)[None, :] - 2.0 * (xq_sub @ xc.T))
            m_take = min(10, len(cand_idx))
            part = np.argpartition(d2, m_take - 1, axis=1)[:, :m_take]
            pd = np.take_along_axis(d2, part, axis=1)
            ord_i = np.argsort(pd, axis=1)
            gt2_d[q_idx, :m_take] = np.take_along_axis(pd, ord_i, axis=1)
            gt2_i[q_idx, :m_take] = cand_idx[np.take_along_axis(part, ord_i, axis=1)]

        data_labels = [[int(ds.col2[i]), int(64 + b_col[i])] for i in range(ds.N)]
        tmp_d = tempfile.mkdtemp(prefix=f"vf_{ds_name}_")
        entry = {"N": ds.N, "D": ds.D, "n_val": n_val, "n_test": n_test}
        try:
            mem_ctx_mib = warmup_cuda_context()
            vf = VecFlow()
            t0 = time.perf_counter()
            vf.build(
                xb,
                data_labels,
                32,
                1000,
                os.path.join(tmp_d, "graph.bin"),
                os.path.join(tmp_d, "bfs.bin"),
                True,
                True,
            )
            build_s = time.perf_counter() - t0
            mem_after_mib = get_gpu_used_mib()
            vram_gib = max(0.05, (mem_after_mib - mem_ctx_mib) / 1024.0)
            entry["status"] = "OK"
            entry["build_s"] = round(build_s, 3)
            entry["vram_gib"] = round(vram_gib, 3)
            entry["cuda_ctx_mib"] = round(mem_ctx_mib, 1)
            entry["total_device_mib"] = round(mem_after_mib, 1)

            itopk_grid = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
            # WL1: single-label `search` (calibrate with 0.960 validation margin)
            best_it1, best_r1 = itopk_grid[-1], 0.0
            for itk in itopk_grid:
                ni, nd = vf.search(xq_val, q_lbl_a[:n_val], itk, 10)
                r = recall_at_10(ni, gt1_i[:n_val], gt1_d[:n_val], nd)
                if r >= best_r1:
                    best_r1, best_it1 = r, itk
                if r >= 0.960:
                    break
            (ni1, nd1), ms1 = trimmed_eval(lambda: vf.search(xq_test, q_lbl_a[n_val:n_tot], best_it1, 10))
            r1_test = recall_at_10(ni1, gt1_i[n_val:n_tot], gt1_d[n_val:n_tot], nd1)
            entry["WL1_single_label"] = {
                "qps": round((n_test * 1000.0) / ms1, 2),
                "recall10": round(r1_test, 4),
                "latency_ms": round(ms1, 3),
                "itopk_size": int(best_it1),
            }

            # WL2: two-label AND `search_multi` (isolated try block for D=1024 upstream bug)
            try:
                best_it2, best_r2 = itopk_grid[-1], 0.0
                for itk in itopk_grid:
                    ni, nd = vf.search_multi(xq_val, q_lbl_a2[:n_val], q_lbl_b2[:n_val], itk, 10)
                    r = recall_at_10(ni, gt2_i[:n_val], gt2_d[:n_val], nd)
                    if r >= best_r2:
                        best_r2, best_it2 = r, itk
                    if r >= 0.960:
                        break
                (ni2, nd2), ms2 = trimmed_eval(lambda: vf.search_multi(xq_test, q_lbl_a2[n_val:n_tot], q_lbl_b2[n_val:n_tot], best_it2, 10))
                r2_test = recall_at_10(ni2, gt2_i[n_val:n_tot], gt2_d[n_val:n_tot], nd2)
                entry["WL2_two_label_and"] = {
                    "qps": round((n_test * 1000.0) / ms2, 2),
                    "recall10": round(r2_test, 4),
                    "latency_ms": round(ms2, 3),
                    "itopk_size": int(best_it2),
                }
                wl2_msg = f"{entry['WL2_two_label_and']['qps']:.1f} QPS (R={r2_test:.4f})"
            except Exception as e_wl2:
                entry["WL2_two_label_and"] = {"status": f"CUDA_ERROR: {str(e_wl2).splitlines()[0]}"}
                wl2_msg = entry["WL2_two_label_and"]["status"]

            print(f"  [Official VecFlow | {ds_name}] pure_VRAM={vram_gib:.2f}GiB, build={build_s:.2f}s, WL1={entry['WL1_single_label']['qps']:.1f} QPS (R={r1_test:.4f}), WL2={wl2_msg}", flush=True)
            del vf
        except Exception as e:
            entry["status"] = f"FAILED/OOM: {e}"
            print(f"  [Official VecFlow | {ds_name}] {entry['status']}", flush=True)
        finally:
            shutil.rmtree(tmp_d, ignore_errors=True)
            gc.collect()

        res["datasets"][ds_name] = entry
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
