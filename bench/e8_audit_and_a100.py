#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase P3/P4: Comprehensive Lossless Audit Summary (`E8`) & Optional Idle-Verified
Cross-GPU Portability Check on `NVIDIA A100 80GB PCIe` (`E9`).
"""

from __future__ import annotations
import json
import os
import subprocess
import time
from typing import Dict, List
import numpy as np
import torch

from src.hybrid_datasets import load_hybrid_dataset, generate_sql_workload, get_paths
from src.oracle import compute_exact_filtered_gt, compute_filtered_recall_at_k, audit_lossless_agreement
from src.prism_engine import PrismEngine
from baselines.filtered_baselines import CuVSFilteredBaseline
from bench.e1_main_evaluation import free_gpu, timed_eval, calibrate_param


def build_e8_lossless_summary(out_dir: str) -> Dict:
    print("=== Building E8 Comprehensive Lossless & Exact-Predicate Audit Summary ===")
    suites = []
    grand_evals = 0
    grand_slots = 0
    grand_mismatches = 0
    grand_pred_viol = 0

    # Dedicated rigorous multi-regime audit across 8 datasets
    for ds_name, nl in [
        ("convlog1m", 1024), ("bge1024", 1024), ("gist1m", 1024), ("msong", 1024),
        ("sift1m", 1024), ("deep10m", 4096), ("sift10m", 4096), ("t2i10m", 4096),
    ]:
        ds = load_hybrid_dataset(ds_name, max_queries=128)
        eng = PrismEngine(nlist=nl, M_blocks=4, use_sign_bit=True, sketch_mode=2, host_pinned_exact=True)
        eng.build(ds, cache_kmeans=True)
        ora = PrismEngine(nlist=nl, M_blocks=4, sketch_mode=0, host_pinned_exact=True)
        ora.build(ds, cache_kmeans=True)

        ds_evals = 0
        ds_slots = 0
        ds_mis = 0
        ds_pv = 0
        max_rdiff = 0.0

        for wl_name, sel in [("W1_range", 0.05), ("W2_correlated", 0.02), ("W3_compound", 0.02), ("W4_mixed", 0.05)]:
            preds = generate_sql_workload(ds, wl_name, num_queries=ds.query_vectors.shape[0], selectivity=sel)
            oi, od, _ = ora.search(ds.query_vectors, preds, k=10, nprobe=32 if nl == 1024 else 48, regime_override=2)
            for reg in (2, 3):
                pi, pd, _ = eng.search(ds.query_vectors, preds, k=10, nprobe=32 if nl == 1024 else 48, regime_override=reg)
                aud = audit_lossless_agreement(ds, preds, pi, pd, oi, od, k=10)
                ds_evals += 1
                ds_slots += aud["total_slots"]
                ds_mis += aud["mismatched_slots"]
                ds_pv += aud["predicate_violations"]
                max_rdiff = max(max_rdiff, aud["max_rel_dist_diff"])

            # Audit Regime 1 against global exact brute-force GT on ultra-low selectivity (0.1%)
            preds_low = generate_sql_workload(ds, wl_name, num_queries=64, selectivity=0.001)
            gti, gtd, _ = compute_exact_filtered_gt(ds, preds_low, k=10, use_cache=True)
            r1i, r1d, _ = eng.search(ds.query_vectors[:64], preds_low, k=10, nprobe=32, regime_override=1)
            aud1 = audit_lossless_agreement(ds, preds_low, r1i, r1d, gti, gtd, k=10)
            ds_evals += 1
            ds_slots += aud1["total_slots"]
            ds_mis += aud1["mismatched_slots"]
            ds_pv += aud1["predicate_violations"]
            max_rdiff = max(max_rdiff, aud1["max_rel_dist_diff"])

        del eng, ora
        free_gpu()
        assert ds_mis == 0 and ds_pv == 0, f"Audit failure on {ds_name}!"
        suites.append({
            "dataset": ds_name,
            "evaluations_audited": ds_evals,
            "total_topk_slots": ds_slots,
            "mismatched_slots": ds_mis,
            "predicate_violations": ds_pv,
            "agreement_pct": 100.0,
            "max_rel_dist_diff": max_rdiff,
        })
        grand_evals += ds_evals
        grand_slots += ds_slots
        grand_mismatches += ds_mis
        grand_pred_viol += ds_pv
        print(f"  [E8 {ds_name:10s}] Evals={ds_evals}, Slots={ds_slots}, Mismatches={ds_mis}, PredViol={ds_pv} -> 100.000%")

    summary = {
        "gpu": torch.cuda.get_device_name(0),
        "grand_evaluations_audited": grand_evals,
        "grand_topk_slots_checked": grand_slots,
        "grand_mismatched_slots": grand_mismatches,
        "grand_predicate_violations": grand_pred_viol,
        "overall_agreement_pct": 100.0 if grand_mismatches == 0 else (100.0 * (grand_slots - grand_mismatches) / grand_slots),
        "datasets": suites,
    }
    with open(os.path.join(out_dir, "e8_lossless_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_a100_portability_worker(out_dir: str) -> Dict:
    """Executed when CUDA_VISIBLE_DEVICES=1 (A100 80GB verified idle)."""
    dev_name = torch.cuda.get_device_name(0)
    print(f"=== Running E9 Cross-GPU Portability on {dev_name} ===")
    rows = []
    for ds_name, nl in [("sift1m", 1024), ("gist1m", 1024), ("convlog1m", 1024), ("t2i10m", 4096)]:
        ds = load_hybrid_dataset(ds_name, max_queries=320)
        q_val, q_test = ds.query_vectors[:64], ds.query_vectors[64:320]
        n_test = q_test.shape[0]
        pf = generate_sql_workload(ds, "W3_compound", num_queries=64 + n_test, selectivity=0.02)
        pv, pt = pf.slice(0, 64), pf.slice(64, 64 + n_test)
        gti, gtd, _ = compute_exact_filtered_gt(ds, pf, k=10, use_cache=True)

        prism = PrismEngine(nlist=nl, M_blocks=4, use_sign_bit=True, sketch_mode=2, host_pinned_exact=True)
        prism.build(ds, cache_kmeans=True)
        ora = PrismEngine(nlist=nl, M_blocks=4, sketch_mode=0, host_pinned_exact=True)
        ora.build(ds, cache_kmeans=True)

        np_cal, _, _ = calibrate_param(
            lambda p: prism.search(q_val, pv, k=10, nprobe=p, regime_override=0)[:2],
            gti[:64], gtd[:64], [16, 32, 48, 64, 96, 128, 192, 256], 0.95, k=10
        )
        (pi, pd, _), pms = timed_eval(lambda: prism.search(q_test, pt, k=10, nprobe=np_cal, regime_override=0))
        prec = compute_filtered_recall_at_k(pi, gti[64:64+n_test], gtd[64:64+n_test], pd, k=10)

        p2i, p2d, _ = prism.search(q_test, pt, k=10, nprobe=np_cal, regime_override=2)
        oi, od, _ = ora.search(q_test, pt, k=10, nprobe=np_cal, regime_override=2)
        aud = audit_lossless_agreement(ds, pt, p2i, p2d, oi, od, k=10)
        assert aud["mismatched_slots"] == 0
        del prism, ora
        free_gpu()

        cb = CuVSFilteredBaseline(mode="ivf_flat", nlist=nl)
        cb.build(ds)
        np_cu, _, _ = calibrate_param(lambda p: cb.search(q_val[:24], pv.slice(0, 24), k=10, nprobe=p)[:2], gti[:24], gtd[:24], [16, 32, 64, 128, 256], 0.95, k=10)
        (ci, cd, _), cms = timed_eval(lambda: cb.search(q_test, pt, k=10, nprobe=np_cu), warmup=1, repeats=3)
        crec = compute_filtered_recall_at_k(ci, gti[64:64+n_test], gtd[64:64+n_test], cd, k=10)
        del cb
        free_gpu()

        rows.append({
            "dataset": ds_name,
            "prism_qps": (n_test * 1000.0) / pms,
            "prism_recall": prec,
            "cuvs_ivf_qps": (n_test * 1000.0) / cms,
            "cuvs_ivf_recall": crec,
            "speedup_vs_cuvs": cms / pms,
            "lossless_agreement_pct": aud["agreement_pct"],
        })
        print(f"  [A100 {ds_name}] PRISM={rows[-1]['prism_qps']:8.1f} QPS vs cuVS={rows[-1]['cuvs_ivf_qps']:7.1f} QPS ({cms/pms:.2f}x, 100% lossless)")

    res = {"gpu": dev_name, "rows": rows}
    with open(os.path.join(out_dir, "e9_a100_portability.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


def main():
    import sys
    prism_root, _, _ = get_paths()
    out_dir = os.path.join(prism_root, "results")
    os.makedirs(out_dir, exist_ok=True)

    if "--a100-worker" in sys.argv:
        run_a100_portability_worker(out_dir)
        return

    build_e8_lossless_summary(out_dir)

    # Check if GPU 1 (A100 80GB) is 100% idle (< 100 MiB used, 0% util) before running 90s portability check
    chk = subprocess.run(
        ["nvidia-smi", "-i", "1", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    if chk.returncode == 0:
        parts = [x.strip() for x in chk.stdout.strip().split(",")]
        if len(parts) == 2 and int(parts[0]) < 100 and int(parts[1]) == 0:
            print("GPU 1 (A100 80GB) verified 100% idle -> running 90s sm_80 portability verification...")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "1"
            subprocess.run([sys.executable, "-u", __file__, "--a100-worker"], env=env, check=True)
        else:
            print(f"GPU 1 is busy ({chk.stdout.strip()}) -> strictly skipping E9 per isolation rule.")


if __name__ == "__main__":
    main()
