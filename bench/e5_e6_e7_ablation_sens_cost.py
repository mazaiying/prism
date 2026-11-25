#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase P3/P4: Step-by-Step Ablation & Slice Pruning Breakdown (`E5`),
4-Axis Parameter Sensitivity (`E6`), and Analytical Cost Model Validation (`E7`).
"""

from __future__ import annotations
import gc
import json
import os
import time
from typing import Dict, List
import numpy as np
import torch
from scipy import stats

from src.hybrid_datasets import load_hybrid_dataset, generate_sql_workload, get_paths
from src.oracle import compute_exact_filtered_gt, compute_filtered_recall_at_k, audit_lossless_agreement
from src.prism_engine import PrismEngine
from baselines.filtered_baselines import CuVSFilteredBaseline
from bench.e1_main_evaluation import free_gpu, timed_eval, calibrate_param


def run_e5_ablation_and_slice_pruning() -> Dict:
    out_ablation = {}
    for ds_name, nlist in [("sift1m", 1024), ("gist1m", 1024), ("convlog1m", 1024), ("sift10m", 4096)]:
        print(f"\n=== E5 Step-by-Step Ablation on {ds_name} ===")
        ds = load_hybrid_dataset(ds_name, max_queries=320)
        q_val, q_test = ds.query_vectors[:64], ds.query_vectors[64:320]
        n_test = q_test.shape[0]
        # Evaluate on Mixed-LogUniform W4 + Compound W3
        pf_w3 = generate_sql_workload(ds, "W3_compound", num_queries=64 + n_test, selectivity=0.02)
        pf_w4 = generate_sql_workload(ds, "W4_mixed", num_queries=64 + n_test, selectivity=0.05)
        gti_w3, gtd_w3, _ = compute_exact_filtered_gt(ds, pf_w3, k=10, use_cache=True)
        gti_w4, gtd_w4, _ = compute_exact_filtered_gt(ds, pf_w4, k=10, use_cache=True)

        # Build the 3 engine variants:
        # 1) No-sketch Host-Offload (`sketch_mode=0`)
        # 2) INT4-only (`sketch_mode=1`, `use_sign_bit=False`)
        # 3) Full INT4+1b Sign-Refined (`sketch_mode=2`, `use_sign_bit=True`)
        e_nosketch = PrismEngine(nlist=nlist, M_blocks=4, sketch_mode=0, host_pinned_exact=True)
        e_nosketch.build(ds, cache_kmeans=True)

        e_int4 = PrismEngine(nlist=nlist, M_blocks=4, use_sign_bit=False, sketch_mode=1, host_pinned_exact=True)
        e_int4.build(ds, cache_kmeans=True)

        e_full = PrismEngine(nlist=nlist, M_blocks=4, use_sign_bit=True, sketch_mode=2, host_pinned_exact=True)
        e_full.build(ds, cache_kmeans=True)

        p_grid = [16, 24, 32, 48, 64, 96, 128, 192]
        np_w3, _, _ = calibrate_param(
            lambda p: e_full.search(q_val, pf_w3.slice(0, 64), k=10, nprobe=p, regime_override=0)[:2],
            gti_w3[:64], gtd_w3[:64], p_grid, 0.95, k=10
        )
        np_w4, _, _ = calibrate_param(
            lambda p: e_full.search(q_val, pf_w4.slice(0, 64), k=10, nprobe=p, regime_override=0)[:2],
            gti_w4[:64], gtd_w4[:64], p_grid, 0.95, k=10
        )

        pt_w3 = pf_w3.slice(64, 64 + n_test)
        pt_w4 = pf_w4.slice(64, 64 + n_test)
        gt3_i, gt3_d = gti_w3[64:64+n_test], gtd_w3[64:64+n_test]
        gt4_i, gt4_d = gti_w4[64:64+n_test], gtd_w4[64:64+n_test]

        variants = [
            ("V0_HostOffload_NoHeader", e_nosketch, dict(regime_override=2, enable_warp_compaction=False, enable_tile_header=False, enable_prewarm=False, adaptive_probe_expand=False)),
            ("V1_+TileHeader_Skip", e_nosketch, dict(regime_override=2, enable_warp_compaction=False, enable_tile_header=True, enable_prewarm=False, adaptive_probe_expand=False)),
            ("V2_+INT4_Ball_Uncomp", e_int4, dict(regime_override=2, enable_warp_compaction=False, enable_tile_header=True, enable_prewarm=False, adaptive_probe_expand=False)),
            ("V3_+1Bit_Sign_Refined", e_full, dict(regime_override=2, enable_warp_compaction=False, enable_tile_header=True, enable_prewarm=False, adaptive_probe_expand=False)),
            ("V4_+Warp_Compaction", e_full, dict(regime_override=2, enable_warp_compaction=True, enable_tile_header=True, enable_prewarm=False, adaptive_probe_expand=False)),
            ("V5_+Survivor_Prewarm", e_full, dict(regime_override=2, enable_warp_compaction=True, enable_tile_header=True, enable_prewarm=True, adaptive_probe_expand=True)),
            ("V6_Full_PRISM_M3", e_full, dict(regime_override=0, enable_warp_compaction=True, enable_tile_header=True, enable_prewarm=True, adaptive_probe_expand=True)),
        ]

        ds_res = {}
        for v_name, eng_v, kw in variants:
            (i3, d3, _), ms3 = timed_eval(lambda: eng_v.search(q_test, pt_w3, k=10, nprobe=np_w3, **kw), warmup=2, repeats=5)
            _, _, tel3 = eng_v.search(q_test, pt_w3, k=10, nprobe=np_w3, record_telemetry=True, **kw)
            (i4, d4, _), ms4 = timed_eval(lambda: eng_v.search(q_test, pt_w4, k=10, nprobe=np_w4, **kw), warmup=2, repeats=5)
            r3 = compute_filtered_recall_at_k(i3, gt3_i, gt3_d, d3, k=10)
            r4 = compute_filtered_recall_at_k(i4, gt4_i, gt4_d, d4, k=10)
            ds_res[v_name] = {
                "w3_qps": (n_test * 1000.0) / ms3,
                "w3_recall": r3,
                "w3_sketch_prune_pct": tel3["overall_sketch_prune_pct"],
                "w3_host_fetches": tel3["host_fp32_fetches"],
                "w4_qps": (n_test * 1000.0) / ms4,
                "w4_recall": r4,
            }
            print(f"  {v_name:24s} | W3={ds_res[v_name]['w3_qps']:8.1f} QPS (Prune={tel3['overall_sketch_prune_pct']:5.2f}%) | W4={ds_res[v_name]['w4_qps']:8.1f} QPS (R@10={r4:.3f})")

        del e_nosketch, e_int4, e_full
        free_gpu()
        out_ablation[ds_name] = ds_res

    # Table 3: Slice-by-Slice Cumulative Pruning Breakdown across 8 Datasets (`Block 1..4` + Total)
    slice_table = {}
    for ds_name, nlist in [
        ("convlog1m", 1024), ("bge1024", 1024), ("gist1m", 1024), ("msong", 1024),
        ("sift1m", 1024), ("deep10m", 4096), ("sift10m", 4096), ("t2i10m", 4096),
    ]:
        ds = load_hybrid_dataset(ds_name, max_queries=256)
        preds = generate_sql_workload(ds, "W3_compound", num_queries=ds.query_vectors.shape[0], selectivity=0.05)
        M_use = 3 if ds.D == 96 else 4
        eng = PrismEngine(nlist=nlist, M_blocks=M_use, use_sign_bit=True, sketch_mode=2, host_pinned_exact=True)
        eng.build(ds, cache_kmeans=True)
        _, _, tel = eng.search(ds.query_vectors, preds, k=10, nprobe=32 if nlist == 1024 else 48, regime_override=2, record_telemetry=True)
        slice_table[ds_name] = {
            "D": ds.D,
            "M": eng.M,
            "header_skip_tile_pct": round(100.0 * tel["tiles_skipped_header"] / max(1, tel["tiles_visited"]), 2),
            "slice1_prune_pct": round(tel["prune_ratio_b0_pct"], 2),
            "slice2_prune_pct": round(tel["prune_ratio_b1_pct"], 2),
            "slice3_prune_pct": round(tel["prune_ratio_b2_pct"], 2),
            "slice4_prune_pct": round(tel["prune_ratio_b3_pct"] if eng.M >= 4 else tel["overall_sketch_prune_pct"], 2),
            "total_sketch_prune_pct": round(tel["overall_sketch_prune_pct"], 2),
        }
        print(f"  [Slice Table {ds_name:10s}] S1={slice_table[ds_name]['slice1_prune_pct']:5.2f}% -> S2={slice_table[ds_name]['slice2_prune_pct']:5.2f}% -> S3={slice_table[ds_name]['slice3_prune_pct']:5.2f}% -> Total={slice_table[ds_name]['total_sketch_prune_pct']:5.2f}%")
        del eng
        free_gpu()

    return {"step_ablation": out_ablation, "slice_pruning_table": slice_table}


def run_e6_sensitivity() -> Dict:
    out = {"batch_size": {}, "top_k": {}, "m_blocks": {}, "quantile_clip": {}}

    for ds_name in ["sift1m", "gist1m"]:
        print(f"\n=== E6 Sensitivity on {ds_name} ===")
        ds = load_hybrid_dataset(ds_name, max_queries=1000)
        B_max = ds.query_vectors.shape[0]
        preds_all = generate_sql_workload(ds, "W3_compound", num_queries=B_max, selectivity=0.05)

        prism = PrismEngine(nlist=1024, M_blocks=4, use_sign_bit=True, sketch_mode=2, host_pinned_exact=True)
        prism.build(ds, cache_kmeans=True)
        cuvs_b = CuVSFilteredBaseline(mode="ivf_flat", nlist=1024)
        cuvs_b.build(ds)

        # 1. Batch Size B in {1, 4, 16, 64, 256, 1000}
        b_rows = []
        for b_sz in [1, 4, 16, 64, 256, min(1000, B_max)]:
            q_sub = ds.query_vectors[:b_sz]
            p_sub = preds_all.slice(0, b_sz)
            _, p_ms = timed_eval(lambda: prism.search(q_sub, p_sub, k=10, nprobe=32, regime_override=0), warmup=3, repeats=7)
            _, c_ms = timed_eval(lambda: cuvs_b.search(q_sub, p_sub, k=10, nprobe=32), warmup=2, repeats=5)
            b_rows.append({
                "batch_size": b_sz,
                "prism_qps": (b_sz * 1000.0) / p_ms,
                "prism_lat_ms": p_ms,
                "cuvs_qps": (b_sz * 1000.0) / c_ms,
                "cuvs_lat_ms": c_ms,
                "speedup": c_ms / p_ms,
            })
        out["batch_size"][ds_name] = b_rows

        # 2. Top-k in {1, 10, 20, 50, 100}
        k_rows = []
        q_256 = ds.query_vectors[:256]
        p_256 = preds_all.slice(0, 256)
        for kv in [1, 10, 20, 50, 100]:
            _, p_ms = timed_eval(lambda: prism.search(q_256, p_256, k=kv, nprobe=40, regime_override=0), warmup=2, repeats=5)
            _, c_ms = timed_eval(lambda: cuvs_b.search(q_256, p_256, k=kv, nprobe=40), warmup=1, repeats=3)
            k_rows.append({
                "k": kv,
                "prism_qps": (256 * 1000.0) / p_ms,
                "cuvs_qps": (256 * 1000.0) / c_ms,
                "speedup": c_ms / p_ms,
            })
        out["top_k"][ds_name] = k_rows
        del prism, cuvs_b
        free_gpu()

        # 3. Macro-Blocks M in {1, 2, 4, 8 (if D>=256)}
        m_list = [1, 2, 4] if ds.D < 256 else [1, 2, 4, 8]
        m_rows = []
        for mv in m_list:
            em = PrismEngine(nlist=1024, M_blocks=mv, use_sign_bit=True, sketch_mode=2, host_pinned_exact=True)
            b_m = em.build(ds, cache_kmeans=True)
            _, p_ms = timed_eval(lambda: em.search(q_256, p_256, k=10, nprobe=32, regime_override=2), warmup=2, repeats=5)
            _, _, tel_m = em.search(q_256, p_256, k=10, nprobe=32, regime_override=2, record_telemetry=True)
            m_rows.append({
                "M_blocks": mv,
                "qps": (256 * 1000.0) / p_ms,
                "vram_mb": b_m["total_gpu_vram_mb"],
                "sketch_prune_pct": tel_m["overall_sketch_prune_pct"],
            })
            del em
            free_gpu()
        out["m_blocks"][ds_name] = m_rows

        # 4. Quantile Clip in {0.0, 0.002, 0.005, 0.01, 0.02}
        qc_rows = []
        ora = PrismEngine(nlist=1024, M_blocks=4, sketch_mode=0, host_pinned_exact=True)
        ora.build(ds, cache_kmeans=True)
        oi, od, _ = ora.search(q_256, p_256, k=10, nprobe=32, regime_override=2)
        for qc in [0.0, 0.002, 0.005, 0.01, 0.02]:
            eq = PrismEngine(nlist=1024, M_blocks=4, use_sign_bit=True, sketch_mode=2, host_pinned_exact=True, quantile_clip=qc)
            eq.build(ds, cache_kmeans=True)
            (qi, qd, _), p_ms = timed_eval(lambda: eq.search(q_256, p_256, k=10, nprobe=32, regime_override=2), warmup=2, repeats=5)
            _, _, tel_q = eq.search(q_256, p_256, k=10, nprobe=32, regime_override=2, record_telemetry=True)
            aud = audit_lossless_agreement(ds, p_256, qi, qd, oi, od, k=10)
            assert aud["mismatched_slots"] == 0
            qc_rows.append({
                "quantile_clip": qc,
                "qps": (256 * 1000.0) / p_ms,
                "sketch_prune_pct": tel_q["overall_sketch_prune_pct"],
                "agreement_pct": aud["agreement_pct"],
            })
            del eq
            free_gpu()
        del ora
        free_gpu()
        out["quantile_clip"][ds_name] = qc_rows
    return out


def run_e7_cost_model_validation() -> Dict:
    print("\n=== E7 Analytical Three-Regime Cost Model Validation ===")
    sels = [0.0005, 0.002, 0.01, 0.05, 0.20, 0.60]
    points = []
    correct_picks = 0
    near_opt_picks = 0  # within 8% of empirical oracle best regime

    for ds_name, nlist in [("sift1m", 1024), ("gist1m", 1024), ("convlog1m", 1024)]:
        ds = load_hybrid_dataset(ds_name, max_queries=256)
        eng = PrismEngine(nlist=nlist, M_blocks=4, use_sign_bit=True, sketch_mode=2, host_pinned_exact=True)
        eng.build(ds, cache_kmeans=True)

        # Warm-up NLMS online calibrator on 6 calibration batches across regimes
        for cal_sel in [0.001, 0.03, 0.45]:
            p_cal = generate_sql_workload(ds, "W3_compound", num_queries=256, selectivity=cal_sel, seed=1001)
            for r_id in (1, 2, 3):
                for _ in range(4):
                    eng.search(ds.query_vectors, p_cal, k=10, nprobe=32, regime_override=r_id, update_cost_model=True)

        # Evaluate out-of-sample across unseen seeds & selectivities
        for sel in sels:
            preds = generate_sql_workload(ds, "W3_compound", num_queries=256, selectivity=sel, seed=3033)
            gt_i, gt_d, _ = compute_exact_filtered_gt(ds, preds, k=10, use_cache=True)

            reg_actual = {}
            reg_pred = {}
            reg_rec = {}
            for r_id in (1, 2, 3):
                (ri, rd, rinfo), rms = timed_eval(
                    lambda r=r_id: eng.search(ds.query_vectors, preds, k=10, nprobe=32, regime_override=r),
                    warmup=2, repeats=5
                )
                reg_actual[r_id] = rms
                reg_pred[r_id] = rinfo["pred_ms"]
                reg_rec[r_id] = compute_filtered_recall_at_k(ri, gt_i, gt_d, rd, k=10)
                points.append({
                    "dataset": ds_name,
                    "selectivity": sel,
                    "regime": r_id,
                    "pred_ms": rinfo["pred_ms"],
                    "actual_ms": rms,
                    "ape_pct": 100.0 * abs(rinfo["pred_ms"] - rms) / max(rms, 1e-6),
                    "recall": reg_rec[r_id],
                })

            # Run Auto M3 and check if picked regime matches quality-qualified fastest regime
            (_, _, auto_info), auto_ms = timed_eval(
                lambda: eng.search(ds.query_vectors, preds, k=10, nprobe=32, regime_override=0),
                warmup=2, repeats=5
            )
            # Best feasible regime (must preserve recall within 0.02 of best recall across regimes)
            max_r = max(reg_rec.values())
            feasible = [r for r in (1, 2, 3) if reg_rec[r] >= max_r - 0.02]
            best_r = min(feasible, key=lambda r: reg_actual[r])
            best_ms = reg_actual[best_r]
            dom_r = max((1, 2, 3), key=lambda r: auto_info[f"r{r}_queries"])
            if dom_r == best_r:
                correct_picks += 1
            if auto_ms <= best_ms * 1.10:
                near_opt_picks += 1

        del eng
        free_gpu()

    pred_arr = np.array([p["pred_ms"] for p in points])
    act_arr = np.array([p["actual_ms"] for p in points])
    apes = np.array([p["ape_pct"] for p in points])
    r_pearson = float(stats.pearsonr(pred_arr, act_arr)[0])
    r_spearman = float(stats.spearmanr(pred_arr, act_arr)[0])
    total_cases = len(sels) * 3

    summary = {
        "num_evaluations": len(points),
        "mape_pct": float(apes.mean()),
        "median_ape_pct": float(np.median(apes)),
        "pearson_r": r_pearson,
        "spearman_rho": r_spearman,
        "exact_regime_hit_rate_pct": 100.0 * correct_picks / total_cases,
        "within_10pct_oracle_rate_pct": 100.0 * near_opt_picks / total_cases,
        "points": points,
    }
    print(
        f"  [E7 Summary] MAPE={summary['mape_pct']:.2f}%, MedAPE={summary['median_ape_pct']:.2f}%, "
        f"Pearson r={r_pearson:.3f}, Regime Hit={summary['exact_regime_hit_rate_pct']:.1f}%"
    )
    return summary


def main():
    prism_root, _, _ = get_paths()
    out_dir = os.path.join(prism_root, "results")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(os.path.join(out_dir, "e5_ablation.json")):
        e5 = run_e5_ablation_and_slice_pruning()
        with open(os.path.join(out_dir, "e5_ablation.json"), "w") as f:
            json.dump(e5, f, indent=2)
    else:
        print("[Skip completed e5_ablation.json]")

    e6 = run_e6_sensitivity()
    with open(os.path.join(out_dir, "e6_sensitivity.json"), "w") as f:
        json.dump(e6, f, indent=2)

    e7 = run_e7_cost_model_validation()
    with open(os.path.join(out_dir, "e7_cost_model.json"), "w") as f:
        json.dump(e7, f, indent=2)

    print("\nCompleted E5, E6, and E7!")


if __name__ == "__main__":
    main()
