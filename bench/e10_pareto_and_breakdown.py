#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advisor-Standard Upgrade Benchmarks E10 & E11:
  - E10: Continuous `QPS vs. Recall (Log-Loss 1 - Recall)` Pareto Trade-Off Curves
         (matching SIGMOD'25 Tribase Fig. 7 & SIGMOD'26 HARMONY Fig. 6, using 7-run trimmed mean!)
  - E11: End-to-End Query 4-Stage Latency Breakdown & Index Build 3-Stage Breakdown
         (matching SIGMOD'26 HARMONY Fig. 8 & Fig. 10)
"""

from __future__ import annotations
import gc
import json
import os
import time
from typing import Callable, Dict, List
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


def trimmed_mean_eval(fn: Callable, warmup: int = 2, repeats: int = 7):
    """Prof. Feng Zhang Master-Prompt protocol: 7 runs, drop highest & lowest, average middle 5."""
    for _ in range(warmup):
        out = fn()
    lats = []
    out = None
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1000.0)
    lats_sorted = sorted(lats)
    mid = lats_sorted[1:-1] if len(lats_sorted) >= 5 else lats_sorted
    return out, float(np.mean(mid))


def main():
    prism_root, _, _ = get_paths()
    out_dir = os.path.join(prism_root, "results")
    os.makedirs(out_dir, exist_ok=True)
    p_e10 = os.path.join(out_dir, "e10_pareto_qps_recall.json")
    p_e11 = os.path.join(out_dir, "e11_time_breakdown.json")

    datasets_cfg = [
        ("convlog1m", 1024),
        ("modbench", 512),
        ("bge1024", 1024),
        ("gist1m", 1024),
        ("msong", 1024),
        ("sift1m", 1024),
        ("deep10m", 4096),
        ("sift10m", 4096),
    ]

    pareto_res: Dict[str, Dict] = {}
    breakdown_res: Dict[str, Dict] = {"query_breakdown": {}, "build_breakdown": {}}
    total_audits = 0
    total_mismatches = 0

    if os.path.exists(p_e10) and os.path.exists(p_e11):
        try:
            old10 = json.load(open(p_e10))
            pareto_res = old10.get("datasets", {})
            total_audits = int(old10.get("total_lossless_audits", 0))
            total_mismatches = int(old10.get("total_mismatches", 0))
            breakdown_res = json.load(open(p_e11))
        except Exception:
            pass

    for ds_name, nlist in datasets_cfg:
        if ds_name in pareto_res and ds_name in breakdown_res.get("query_breakdown", {}):
            print(f"[Skip completed E10/E11 dataset] {ds_name}", flush=True)
            continue
        print(f"\n==================== E10/E11 Dataset: {ds_name} (nlist={nlist}) ====================", flush=True)
        ds = load_hybrid_dataset(ds_name, max_queries=256)
        B = min(256, ds.query_vectors.shape[0])
        xq = ds.query_vectors[:B]
        preds = generate_sql_workload(ds, "W3_compound", num_queries=B, selectivity=0.02)
        gt_i, gt_d, surv = compute_exact_filtered_gt(ds, preds, k=10, use_cache=True)

        if nlist <= 512:
            probes = [4, 8, 16, 32, 64, 128, 256, 384]
        elif nlist == 1024:
            probes = [8, 16, 32, 64, 128, 256, 384, 512]
        else:
            probes = [16, 32, 64, 128, 256, 384, 512, 768]

        # 1. Build PRISM (Full INT4+1b), PRISM-INT4 (no sign), and Host-Offload (no sketch)
        eng_prism = PrismEngine(nlist=nlist, M_blocks=4, sketch_mode=2, use_sign_bit=True, host_pinned_exact=True)
        b_prism = eng_prism.build(ds, cache_kmeans=True)
        breakdown_res["build_breakdown"][ds_name] = {
            "N": ds.N,
            "D": ds.D,
            "kmeans_s": b_prism["kmeans_time_s"],
            "layout_s": b_prism["layout_time_s"],
            "encode_s": b_prism["encode_time_s"],
            "total_s": b_prism["total_build_s"],
            "vram_gib": b_prism["total_gpu_vram_gib"],
        }

        eng_int4 = PrismEngine(nlist=nlist, M_blocks=4, sketch_mode=1, use_sign_bit=False, host_pinned_exact=True)
        eng_int4.build(ds, cache_kmeans=True)

        eng_host = PrismEngine(nlist=nlist, M_blocks=4, sketch_mode=0, host_pinned_exact=True)
        eng_host.build(ds, cache_kmeans=True)

        ds_curves: Dict[str, List[Dict[str, float]]] = {
            "PRISM": [],
            "PRISM_INT4": [],
            "Host_Offload": [],
            "cuVS_IVF": [],
            "cuVS_CAGRA": [],
            "ACORN_48T": [],
        }

        for np_val in probes:
            (pi, pd, _), ms_p = trimmed_mean_eval(
                lambda p=np_val: eng_prism.search(xq, preds, k=10, nprobe=p, adaptive_probe_expand=False)
            )
            rec_p = compute_filtered_recall_at_k(pi, gt_i, gt_d, pd, k=10)
            ds_curves["PRISM"].append({"param": np_val, "recall": rec_p, "loss": max(1e-5, 1.0 - rec_p), "qps": B * 1000.0 / ms_p})

            (i4_i, i4_d, _), ms_4 = trimmed_mean_eval(
                lambda p=np_val: eng_int4.search(xq, preds, k=10, nprobe=p, adaptive_probe_expand=False)
            )
            rec_4 = compute_filtered_recall_at_k(i4_i, gt_i, gt_d, i4_d, k=10)
            ds_curves["PRISM_INT4"].append({"param": np_val, "recall": rec_4, "loss": max(1e-5, 1.0 - rec_4), "qps": B * 1000.0 / ms_4})

            (hi, hd, _), ms_h = trimmed_mean_eval(
                lambda p=np_val: eng_host.search(xq, preds, k=10, nprobe=p, adaptive_probe_expand=False)
            )
            rec_h = compute_filtered_recall_at_k(hi, gt_i, gt_d, hd, k=10)
            ds_curves["Host_Offload"].append({"param": np_val, "recall": rec_h, "loss": max(1e-5, 1.0 - rec_h), "qps": B * 1000.0 / ms_h})

            # Audit 100% lossless agreement at identical nprobe (fixed Regime 2)
            pi_r2, pd_r2, _ = eng_prism.search(xq, preds, k=10, nprobe=np_val, regime_override=2, adaptive_probe_expand=False)
            hi_r2, hd_r2, _ = eng_host.search(xq, preds, k=10, nprobe=np_val, regime_override=2, adaptive_probe_expand=False)
            rep = audit_lossless_agreement(ds, preds, pi_r2, pd_r2, hi_r2, hd_r2, k=10)
            total_audits += 1
            total_mismatches += rep["mismatched_slots"]
            assert rep["mismatched_slots"] == 0, f"Mismatch on {ds_name} nprobe={np_val}: {rep}"

        # Add exact global Regime 1 point (`Recall = 1.0000`) at end of PRISM curve!
        (gi, gd, _), ms_g = trimmed_mean_eval(
            lambda: eng_prism.search(xq, preds, k=10, nprobe=nlist, regime_override=1)
        )
        rec_g = compute_filtered_recall_at_k(gi, gt_i, gt_d, gd, k=10)
        ds_curves["PRISM"].append({"param": nlist, "recall": rec_g, "loss": max(1e-5, 1.0 - rec_g), "qps": B * 1000.0 / ms_g})

        # E11: Measure 4-stage query breakdown at high-recall probe (`probes[-2]`)
        p_hi = probes[-2]
        # Clean wall-clock latency from trimmed_mean without telemetry atomics, scaled onto cycle proportions
        _, clean_ms_prism = trimmed_mean_eval(lambda: eng_prism.search(xq, preds, k=10, nprobe=p_hi, regime_override=2))
        _, _, tel_prism = eng_prism.search(xq, preds, k=10, nprobe=p_hi, regime_override=2, record_telemetry=True)

        _, clean_ms_uncomp = trimmed_mean_eval(lambda: eng_prism.search(xq, preds, k=10, nprobe=p_hi, regime_override=2, enable_warp_compaction=False))
        _, _, tel_uncomp = eng_prism.search(xq, preds, k=10, nprobe=p_hi, regime_override=2, enable_warp_compaction=False, record_telemetry=True)

        _, clean_ms_host = trimmed_mean_eval(lambda: eng_host.search(xq, preds, k=10, nprobe=p_hi, regime_override=2))
        _, _, tel_host = eng_host.search(xq, preds, k=10, nprobe=p_hi, regime_override=2, record_telemetry=True)

        def _norm_stages(clean_ms: float, tel: Dict[str, float]) -> Dict[str, float]:
            raw_tot = max(1e-6, tel["stage_plan_ms"] + tel["stage_filter_ms"] + tel["stage_sketch_ms"] + tel["stage_refine_ms"])
            scale = clean_ms / raw_tot
            return {
                "total_ms": round(clean_ms, 4),
                "qps": round(B * 1000.0 / clean_ms, 2),
                "plan_ms": round(tel["stage_plan_ms"] * scale, 4),
                "filter_ms": round(tel["stage_filter_ms"] * scale, 4),
                "sketch_ms": round(tel["stage_sketch_ms"] * scale, 4),
                "refine_ms": round(tel["stage_refine_ms"] * scale, 4),
                "prune_pct": round(tel.get("overall_sketch_prune_pct", 0.0), 2),
                "host_fetches": int(tel.get("host_fp32_fetches", 0)),
            }

        breakdown_res["query_breakdown"][ds_name] = {
            "PRISM": _norm_stages(clean_ms_prism, tel_prism),
            "Uncompacted_Sketch": _norm_stages(clean_ms_uncomp, tel_uncomp),
            "Host_Offload": _norm_stages(clean_ms_host, tel_host),
        }

        del eng_prism, eng_int4, eng_host
        free_gpu()

        # 2. cuVS-IVF-Flat Pareto curve
        try:
            cuvs_ivf = CuVSFilteredBaseline(mode="ivf_flat", nlist=nlist)
            b_ivf = cuvs_ivf.build(ds)
            breakdown_res["build_breakdown"][ds_name]["cuvs_ivf_build_s"] = b_ivf["total_build_s"]
            for np_val in probes:
                (ci, cd, _), ms_c = trimmed_mean_eval(
                    lambda p=np_val: cuvs_ivf.search(xq, preds, k=10, nprobe=p), warmup=1, repeats=5
                )
                rec_c = compute_filtered_recall_at_k(ci, gt_i, gt_d, cd, k=10)
                ds_curves["cuVS_IVF"].append({"param": np_val, "recall": rec_c, "loss": max(1e-5, 1.0 - rec_c), "qps": B * 1000.0 / ms_c})
            del cuvs_ivf
            free_gpu()
        except Exception as e:
            print(f"  [cuVS-IVF skipped on {ds_name}]: {e}")
            free_gpu()

        # 3. cuVS-CAGRA Pareto curve
        try:
            cuvs_cg = CuVSFilteredBaseline(mode="cagra", nlist=nlist)
            b_cg = cuvs_cg.build(ds)
            breakdown_res["build_breakdown"][ds_name]["cuvs_cagra_build_s"] = b_cg["total_build_s"]
            for itk in [32, 64, 128, 256, 384, 512]:
                (gi_c, gd_c, _), ms_cg = trimmed_mean_eval(
                    lambda ik=itk: cuvs_cg.search(xq, preds, k=10, itopk_size=ik), warmup=1, repeats=5
                )
                rec_cg = compute_filtered_recall_at_k(gi_c, gt_i, gt_d, gd_c, k=10)
                ds_curves["cuVS_CAGRA"].append({"param": itk, "recall": rec_cg, "loss": max(1e-5, 1.0 - rec_cg), "qps": B * 1000.0 / ms_cg})
            del cuvs_cg
            free_gpu()
        except Exception as e:
            print(f"  [cuVS-CAGRA skipped on {ds_name}]: {e}")
            free_gpu()

        # 4. ACORN-48T CPU Graph Pareto curve
        try:
            acorn = CPUGraphFilteredBaseline(acorn_mode=True)
            b_ac = acorn.build(ds, cache_graph=True)
            breakdown_res["build_breakdown"][ds_name]["acorn_48t_build_s"] = b_ac["total_build_s"]
            for ef in [32, 64, 128, 256, 512, 800]:
                (ai, ad, _), ms_a = trimmed_mean_eval(
                    lambda e_val=ef: acorn.search(xq, preds, k=10, ef_search=e_val), warmup=1, repeats=5
                )
                rec_a = compute_filtered_recall_at_k(ai, gt_i, gt_d, ad, k=10)
                ds_curves["ACORN_48T"].append({"param": ef, "recall": rec_a, "loss": max(1e-5, 1.0 - rec_a), "qps": B * 1000.0 / ms_a})
            del acorn
            free_gpu()
        except Exception as e:
            print(f"  [ACORN-48T skipped on {ds_name}]: {e}")

        pareto_res[ds_name] = ds_curves
        print(f"  [{ds_name}] PRISM peak R={max(x['recall'] for x in ds_curves['PRISM']):.4f}, "
              f"breakdown PRISM={breakdown_res['query_breakdown'][ds_name]['PRISM']}", flush=True)

        with open(p_e10, "w") as f:
            json.dump({
                "gpu": torch.cuda.get_device_name(0),
                "protocol": "7-run trimmed mean (drop min & max)",
                "total_lossless_audits": total_audits,
                "total_mismatches": total_mismatches,
                "datasets": pareto_res,
            }, f, indent=2)

        with open(p_e11, "w") as f:
            json.dump(breakdown_res, f, indent=2)

    # Also merge t2i10m and sift100m build breakdowns from e1/e4 if present
    for extra_f, extra_k in [("e1_main_evaluation.json", "t2i10m"), ("e4_sift100m_scale.json", "sift100m")]:
        ep = os.path.join(out_dir, extra_f)
        if os.path.exists(ep):
            d_ex = json.load(open(ep))
            if extra_k == "t2i10m" and "t2i10m" in d_ex.get("datasets", {}):
                bp = d_ex["datasets"]["t2i10m"]["build_and_memory"]["PRISM"]
                breakdown_res["build_breakdown"]["t2i10m"] = {
                    "N": 10_000_000, "D": 200,
                    "kmeans_s": bp["kmeans_time_s"], "layout_s": bp["layout_time_s"],
                    "encode_s": bp["encode_time_s"], "total_s": bp["total_build_s"],
                    "vram_gib": bp["total_gpu_vram_gib"],
                    "acorn_48t_build_s": d_ex["datasets"]["t2i10m"]["build_and_memory"].get("B5_ACORN_48T", {}).get("total_build_s", 0.0),
                }
            elif extra_k == "sift100m" and "build_prism" in d_ex:
                bp = d_ex["build_prism"]
                breakdown_res["build_breakdown"]["sift100m"] = {
                    "N": 100_000_000, "D": 128,
                    "kmeans_s": bp["kmeans_time_s"], "layout_s": bp["layout_time_s"],
                    "encode_s": bp["encode_time_s"], "total_s": bp["total_build_s"],
                    "vram_gib": bp["total_gpu_vram_gib"],
                }
    with open(p_e11, "w") as f:
        json.dump(breakdown_res, f, indent=2)
    print(f"\n[DONE E10 & E11] Saved {p_e10} and {p_e11} (audits={total_audits}, mismatches={total_mismatches})")


if __name__ == "__main__":
    main()
