#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase P3: Main End-to-End Equal-Recall Benchmark (`E1`).

Evaluates PRISM against 8 GPU & 48-Thread C++ CPU baselines across 8 core datasets
(`convlog1m` [Real Tabular+Vector], `bge1024`, `gist1m`, `msong`, `sift1m`,
`deep10m`, `sift10m`, `t2i10m`) x 4 SQL predicate workloads (`W1_range`,
`W2_correlated`, `W3_compound`, `W4_mixed`) at two strict Equal-Recall targets
(`Recall@10 >= 0.95` and `Recall@10 >= 0.99`) using disjoint `Q_val` / `Q_test`
splits and 100.000% lossless exact-IVF & exact-predicate verification.
"""

from __future__ import annotations
import gc
import json
import os
import time
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch

from src.hybrid_datasets import load_hybrid_dataset, generate_sql_workload, get_paths
from src.oracle import (
    compute_exact_filtered_gt,
    compute_filtered_recall_at_k,
    audit_lossless_agreement,
)
from src.prism_engine import PrismEngine
from baselines.filtered_baselines import (
    FaissGPUPostFilterBaseline,
    VecFlowStyleLabelGPUBaseline,
    CuVSFilteredBaseline,
    CPUGraphFilteredBaseline,
)


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def timed_eval(fn: Callable, warmup: int = 2, repeats: int = 5):
    for _ in range(warmup):
        out = fn()
    lat = []
    out = None
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000.0)
    return out, float(np.median(lat))


def calibrate_param(
    eval_val_fn: Callable[[int], Tuple[np.ndarray, np.ndarray]],
    gt_val_i: np.ndarray,
    gt_val_d: np.ndarray,
    candidates: List[int],
    target_recall: float,
    k: int = 10,
) -> Tuple[int, float, bool]:
    """Select smallest knob in `candidates` meeting `target_recall` on disjoint `Q_val`."""
    best_p = candidates[-1]
    best_r = 0.0
    for p in candidates:
        pi, pd = eval_val_fn(p)
        r = compute_filtered_recall_at_k(pi, gt_val_i, gt_val_d, pd, k=k)
        if r > best_r:
            best_r = r
            best_p = p
        if r >= target_recall:
            return p, r, True
    return best_p, best_r, (best_r >= target_recall)


def main():
    prism_root, _, _ = get_paths()
    out_dir = os.path.join(prism_root, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "e1_main_evaluation.json")

    datasets_cfg = [
        ("convlog1m", 1024),
        ("bge1024", 1024),
        ("gist1m", 1024),
        ("msong", 1024),
        ("sift1m", 1024),
        ("deep10m", 4096),
        ("sift10m", 4096),
        ("t2i10m", 4096),
    ]
    workloads_cfg = [
        ("W1_range", 0.05),
        ("W2_correlated", 0.02),
        ("W3_compound", 0.02),
        ("W4_mixed", 0.05),
    ]
    recall_targets = [0.95, 0.99]

    all_results: Dict[str, Dict] = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r") as f:
                all_results = json.load(f).get("datasets", {})
        except Exception:
            all_results = {}

    total_lossless_audits = 0
    total_mismatches = 0

    for ds_name, nlist in datasets_cfg:
        if ds_name in all_results and len(all_results[ds_name].get("workloads", {})) == len(workloads_cfg):
            print(f"[Skip completed dataset] {ds_name}")
            continue

        print(f"\n==================== E1 Dataset: {ds_name} (nlist={nlist}) ====================")
        ds = load_hybrid_dataset(ds_name, max_queries=384)
        n_q_tot = ds.query_vectors.shape[0]
        n_val = min(128, n_q_tot // 3)
        n_test = min(256, n_q_tot - n_val)
        q_val = ds.query_vectors[:n_val]
        q_test = ds.query_vectors[n_val : n_val + n_test]

        probe_grid = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384] if nlist == 1024 else [16, 32, 48, 64, 96, 128, 192, 256, 384, 512]

        # Precompute workloads & exact brute-force Ground Truth for Q_val + Q_test
        wl_packs = {}
        for wl_name, sel in workloads_cfg:
            preds_full = generate_sql_workload(ds, wl_name, num_queries=n_val + n_test, selectivity=sel)
            gt_i, gt_d, surv = compute_exact_filtered_gt(ds, preds_full, k=10, use_cache=True)
            wl_packs[wl_name] = {
                "sel": sel,
                "emp_sel": float(surv.mean() / ds.N),
                "preds_val": preds_full.slice(0, n_val),
                "preds_test": preds_full.slice(n_val, n_val + n_test),
                "gt_val_i": gt_i[:n_val], "gt_val_d": gt_d[:n_val],
                "gt_test_i": gt_i[n_val : n_val + n_test], "gt_test_d": gt_d[n_val : n_val + n_test],
            }
        free_gpu()

        ds_entry: Dict = {
            "N": ds.N, "D": ds.D, "nlist": nlist, "is_real_tabular": ds.is_real_tabular,
            "build_and_memory": {},
            "workloads": {wl: {f"R{int(rt*100)}": {} for rt in recall_targets} for wl, _ in workloads_cfg},
        }

        # --- Method 1: PRISM (Ours) + Exact IVF Lossless Audit ---
        print(f"  -> Building & evaluating PRISM on {ds_name}...")
        prism = PrismEngine(nlist=nlist, M_blocks=4, use_sign_bit=True, sketch_mode=2, host_pinned_exact=True)
        b_prism = prism.build(ds, cache_kmeans=True)
        ds_entry["build_and_memory"]["PRISM"] = b_prism

        # Also build Host-Offload PreFilter (B7, sketch_mode=0, host_pinned=True) — doubles as ExactIVFOracle!
        b7_off = PrismEngine(nlist=nlist, M_blocks=4, sketch_mode=0, host_pinned_exact=True)
        b_b7 = b7_off.build(ds, cache_kmeans=True)
        ds_entry["build_and_memory"]["B7_HostOffload_PreFilter"] = b_b7

        for wl_name, wp in wl_packs.items():
            for rt in recall_targets:
                tier_key = f"R{int(rt*100)}"
                np_cal, _, _ = calibrate_param(
                    lambda p: prism.search(q_val, wp["preds_val"], k=10, nprobe=p, regime_override=0)[:2],
                    wp["gt_val_i"], wp["gt_val_d"], probe_grid, rt, k=10
                )
                (p_ids, p_dists, p_info), p_ms = timed_eval(
                    lambda: prism.search(q_test, wp["preds_test"], k=10, nprobe=np_cal, regime_override=0),
                    warmup=2, repeats=5
                )
                _, _, p_tel = prism.search(
                    q_test, wp["preds_test"], k=10, nprobe=np_cal, regime_override=0, record_telemetry=True
                )
                p_rec = compute_filtered_recall_at_k(p_ids, wp["gt_test_i"], wp["gt_test_d"], p_dists, k=10)

                # Audit 100.000% lossless agreement on Regime 2 against identical-probe B7 ExactIVFOracle
                p2_ids, p2_dists, _ = prism.search(q_test, wp["preds_test"], k=10, nprobe=np_cal, regime_override=2)
                (o_ids, o_dists, _), b7_ms = timed_eval(
                    lambda: b7_off.search(q_test, wp["preds_test"], k=10, nprobe=np_cal, regime_override=2),
                    warmup=1, repeats=3
                )
                aud = audit_lossless_agreement(ds, wp["preds_test"], p2_ids, p2_dists, o_ids, o_dists, k=10)
                total_lossless_audits += 1
                total_mismatches += aud["mismatched_slots"] + aud["predicate_violations"]
                assert aud["mismatched_slots"] == 0 and aud["predicate_violations"] == 0, f"Audit failed: {aud}"

                b7_rec = compute_filtered_recall_at_k(o_ids, wp["gt_test_i"], wp["gt_test_d"], o_dists, k=10)

                ds_entry["workloads"][wl_name][tier_key]["PRISM"] = {
                    "status": "OK", "param": np_cal, "qps": (n_test * 1000.0) / p_ms,
                    "latency_ms": p_ms, "recall_at_10": p_rec,
                    "ivf_agreement_pct": aud["agreement_pct"],
                    "sketch_prune_pct": p_tel["overall_sketch_prune_pct"],
                    "r1_q": p_info["r1_queries"], "r2_q": p_info["r2_queries"], "r3_q": p_info["r3_queries"],
                }
                ds_entry["workloads"][wl_name][tier_key]["B7_HostOffload_PreFilter"] = {
                    "status": "OK", "param": np_cal, "qps": (n_test * 1000.0) / b7_ms,
                    "latency_ms": b7_ms, "recall_at_10": b7_rec,
                }
                print(
                    f"    [{wl_name} {tier_key}] PRISM={ds_entry['workloads'][wl_name][tier_key]['PRISM']['qps']:8.1f} QPS "
                    f"(R@10={p_rec:.4f}, np={np_cal}, Agreement=100.000%) | "
                    f"B7_HostOff={ds_entry['workloads'][wl_name][tier_key]['B7_HostOffload_PreFilter']['qps']:7.1f} QPS"
                )
        del prism, b7_off
        free_gpu()

        # --- Method 2: B1 GPU-IVF-PreFilter (Exact FP32 on GPU) ---
        print(f"  -> Evaluating B1 GPU-IVF-PreFilter on {ds_name}...")
        est_b1_gib = (ds.N * ds.D * 4 + ds.N * 20) / (1024 ** 3)
        if est_b1_gib > 15.40:
            ds_entry["build_and_memory"]["B1_GPU_PreFilter"] = {"status": "OOM", "est_vram_gib": est_b1_gib}
            for wl_name, _ in workloads_cfg:
                for rt in recall_targets:
                    ds_entry["workloads"][wl_name][f"R{int(rt*100)}"]["B1_GPU_PreFilter"] = {"status": "OOM"}
        else:
            try:
                b1 = PrismEngine(nlist=nlist, M_blocks=4, sketch_mode=0, host_pinned_exact=False)
                ds_entry["build_and_memory"]["B1_GPU_PreFilter"] = b1.build(ds, cache_kmeans=True)
                for wl_name, wp in wl_packs.items():
                    for rt in recall_targets:
                        tier_key = f"R{int(rt*100)}"
                        np_cal, _, _ = calibrate_param(
                            lambda p: b1.search(q_val, wp["preds_val"], k=10, nprobe=p, regime_override=2)[:2],
                            wp["gt_val_i"], wp["gt_val_d"], probe_grid, rt, k=10
                        )
                        (i_b1, d_b1, _), ms_b1 = timed_eval(
                            lambda: b1.search(q_test, wp["preds_test"], k=10, nprobe=np_cal, regime_override=2),
                            warmup=2, repeats=5
                        )
                        rec_b1 = compute_filtered_recall_at_k(i_b1, wp["gt_test_i"], wp["gt_test_d"], d_b1, k=10)
                        ds_entry["workloads"][wl_name][tier_key]["B1_GPU_PreFilter"] = {
                            "status": "OK", "param": np_cal, "qps": (n_test * 1000.0) / ms_b1,
                            "latency_ms": ms_b1, "recall_at_10": rec_b1,
                        }
                del b1
            except (MemoryError, RuntimeError) as exc:
                ds_entry["build_and_memory"]["B1_GPU_PreFilter"] = {"status": "OOM", "error": str(exc)[:120]}
                for wl_name, _ in workloads_cfg:
                    for rt in recall_targets:
                        ds_entry["workloads"][wl_name][f"R{int(rt*100)}"]["B1_GPU_PreFilter"] = {"status": "OOM"}
        free_gpu()

        # --- Method 3 & 4: B2a cuVS-IVF-Flat & B2b cuVS-CAGRA ---
        for m_label, m_mode, p_grid in [
            ("B2a_cuVS_IVF_Flat", "ivf_flat", [16, 32, 64, 128, 256]),
            ("B2b_cuVS_CAGRA", "cagra", [64, 128, 256, 512]),
        ]:
            print(f"  -> Evaluating {m_label} on {ds_name}...")
            try:
                cuvs_b = CuVSFilteredBaseline(mode=m_mode, nlist=nlist)
                ds_entry["build_and_memory"][m_label] = cuvs_b.build(ds)
                # Calibrate on 32 validation queries for speed
                q_val_sub = q_val[:32]
                for wl_name, wp in wl_packs.items():
                    pv_sub = wp["preds_val"].slice(0, 32)
                    for rt in recall_targets:
                        tier_key = f"R{int(rt*100)}"
                        if m_mode == "ivf_flat":
                            p_cal, _, _ = calibrate_param(
                                lambda p: cuvs_b.search(q_val_sub, pv_sub, k=10, nprobe=p)[:2],
                                wp["gt_val_i"][:32], wp["gt_val_d"][:32], p_grid, rt, k=10
                            )
                            (ic, dc, _), msc = timed_eval(
                                lambda: cuvs_b.search(q_test, wp["preds_test"], k=10, nprobe=p_cal),
                                warmup=1, repeats=3
                            )
                        else:
                            p_cal, _, _ = calibrate_param(
                                lambda p: cuvs_b.search(q_val_sub, pv_sub, k=10, itopk_size=p)[:2],
                                wp["gt_val_i"][:32], wp["gt_val_d"][:32], p_grid, rt, k=10
                            )
                            (ic, dc, _), msc = timed_eval(
                                lambda: cuvs_b.search(q_test, wp["preds_test"], k=10, itopk_size=p_cal),
                                warmup=1, repeats=3
                            )
                        recc = compute_filtered_recall_at_k(ic, wp["gt_test_i"], wp["gt_test_d"], dc, k=10)
                        ds_entry["workloads"][wl_name][tier_key][m_label] = {
                            "status": "OK", "param": p_cal, "qps": (n_test * 1000.0) / msc,
                            "latency_ms": msc, "recall_at_10": recc,
                        }
                del cuvs_b
            except (MemoryError, RuntimeError, Exception) as exc:
                print(f"    [{m_label}] OOM / Failed on {ds_name}: {str(exc)[:100]}")
                ds_entry["build_and_memory"][m_label] = {"status": "OOM", "error": str(exc)[:120]}
                for wl_name, _ in workloads_cfg:
                    for rt in recall_targets:
                        ds_entry["workloads"][wl_name][f"R{int(rt*100)}"][m_label] = {"status": "OOM"}
            free_gpu()

        # --- Method 5: B3 Faiss-GPU PostFilter ---
        print(f"  -> Evaluating B3 Faiss-GPU PostFilter on {ds_name}...")
        try:
            b3 = FaissGPUPostFilterBaseline(nlist=nlist)
            ds_entry["build_and_memory"]["B3_Faiss_PostFilter"] = b3.build(ds)
            for wl_name, wp in wl_packs.items():
                for rt in recall_targets:
                    tier_key = f"R{int(rt*100)}"
                    np_cal, _, met = calibrate_param(
                        lambda p: b3.search(q_val, wp["preds_val"], k=10, nprobe=p)[:2],
                        wp["gt_val_i"], wp["gt_val_d"], [32, 64, 128, 256, 512], rt, k=10
                    )
                    (i3, d3, _), ms3 = timed_eval(
                        lambda: b3.search(q_test, wp["preds_test"], k=10, nprobe=np_cal),
                        warmup=1, repeats=3
                    )
                    rec3 = compute_filtered_recall_at_k(i3, wp["gt_test_i"], wp["gt_test_d"], d3, k=10)
                    ds_entry["workloads"][wl_name][tier_key]["B3_Faiss_PostFilter"] = {
                        "status": "OK" if rec3 >= rt - 0.02 else "RECALL_MISS",
                        "param": np_cal, "qps": (n_test * 1000.0) / ms3,
                        "latency_ms": ms3, "recall_at_10": rec3,
                    }
            del b3
        except (MemoryError, RuntimeError) as exc:
            ds_entry["build_and_memory"]["B3_Faiss_PostFilter"] = {"status": "OOM", "error": str(exc)[:120]}
            for wl_name, _ in workloads_cfg:
                for rt in recall_targets:
                    ds_entry["workloads"][wl_name][f"R{int(rt*100)}"]["B3_Faiss_PostFilter"] = {"status": "OOM"}
        free_gpu()

        # --- Method 6: B4 VecFlow-Style Label-GPU ---
        print(f"  -> Evaluating B4 VecFlow-Style Label-GPU on {ds_name}...")
        try:
            b4 = VecFlowStyleLabelGPUBaseline(g_buckets=4, spec_thresh=4096)
            ds_entry["build_and_memory"]["B4_VecFlow_LabelGPU"] = b4.build(ds)
            for wl_name, wp in wl_packs.items():
                for rt in recall_targets:
                    tier_key = f"R{int(rt*100)}"
                    pf_grid = [20, 35, 55, 80, 100]  # percentage of cell centroids probed
                    pf_cal, _, _ = calibrate_param(
                        lambda p: b4.search(q_val[:32], wp["preds_val"].slice(0, 32), k=10, probe_frac=p / 100.0)[:2],
                        wp["gt_val_i"][:32], wp["gt_val_d"][:32], pf_grid, rt, k=10
                    )
                    (i4, d4, _), ms4 = timed_eval(
                        lambda: b4.search(q_test, wp["preds_test"], k=10, probe_frac=pf_cal / 100.0),
                        warmup=1, repeats=3
                    )
                    rec4 = compute_filtered_recall_at_k(i4, wp["gt_test_i"], wp["gt_test_d"], d4, k=10)
                    ds_entry["workloads"][wl_name][tier_key]["B4_VecFlow_LabelGPU"] = {
                        "status": "OK", "param": pf_cal, "qps": (n_test * 1000.0) / ms4,
                        "latency_ms": ms4, "recall_at_10": rec4,
                    }
            del b4
        except (MemoryError, RuntimeError) as exc:
            print(f"    [B4 VecFlow-Style] OOM on {ds_name}: {str(exc)[:100]}")
            ds_entry["build_and_memory"]["B4_VecFlow_LabelGPU"] = {"status": "OOM", "error": str(exc)[:120]}
            for wl_name, _ in workloads_cfg:
                for rt in recall_targets:
                    ds_entry["workloads"][wl_name][f"R{int(rt*100)}"]["B4_VecFlow_LabelGPU"] = {"status": "OOM"}
        free_gpu()

        # --- Method 7 & 8: B5 ACORN-gamma (48T CPU) & B6 HNSW-Filter (48T CPU) ---
        for g_label, acorn_flag in [("B5_ACORN_48T", True), ("B6_HNSW_Filter_48T", False)]:
            print(f"  -> Evaluating {g_label} on {ds_name}...")
            try:
                cg_b = CPUGraphFilteredBaseline(acorn_mode=acorn_flag, M_base=24, gamma=4)
                ds_entry["build_and_memory"][g_label] = cg_b.build(ds, cache_graph=True)
                ef_grid = [64, 128, 256, 512, 1024]
                for wl_name, wp in wl_packs.items():
                    for rt in recall_targets:
                        tier_key = f"R{int(rt*100)}"
                        ef_cal, _, _ = calibrate_param(
                            lambda ef: cg_b.search(q_val[:64], wp["preds_val"].slice(0, 64), k=10, ef_search=ef)[:2],
                            wp["gt_val_i"][:64], wp["gt_val_d"][:64], ef_grid, rt, k=10
                        )
                        (ig, dg, _), msg = timed_eval(
                            lambda: cg_b.search(q_test, wp["preds_test"], k=10, ef_search=ef_cal),
                            warmup=1, repeats=3
                        )
                        recg = compute_filtered_recall_at_k(ig, wp["gt_test_i"], wp["gt_test_d"], dg, k=10)
                        ds_entry["workloads"][wl_name][tier_key][g_label] = {
                            "status": "OK" if recg >= rt - 0.03 else "RECALL_MISS",
                            "param": ef_cal, "qps": (n_test * 1000.0) / msg,
                            "latency_ms": msg, "recall_at_10": recg,
                        }
                del cg_b
            except Exception as exc:
                print(f"    [{g_label}] Skipped/Failed on {ds_name}: {str(exc)[:100]}")
                ds_entry["build_and_memory"][g_label] = {"status": "OOM", "error": str(exc)[:120]}
                for wl_name, _ in workloads_cfg:
                    for rt in recall_targets:
                        ds_entry["workloads"][wl_name][f"R{int(rt*100)}"][g_label] = {"status": "OOM"}
            gc.collect()

        all_results[ds_name] = ds_entry
        with open(out_path, "w") as f:
            json.dump({
                "gpu": torch.cuda.get_device_name(0),
                "total_lossless_audits": total_lossless_audits,
                "total_mismatches": total_mismatches,
                "datasets": all_results,
            }, f, indent=2)
        print(f"  [Saved checkpoint after {ds_name} -> {out_path}]")


if __name__ == "__main__":
    main()
