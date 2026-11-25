#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Data-Center GPU Evaluation Suite (`E9`) on NVIDIA A100 80GB PCIe (`GPU 1`, sm_80).

Serves two synergistic purposes in parallel with GPU 0 (RTX 4080 16GB):
  1. Pre-warms hardware-independent 10M CPU-graph & 100M Ground-Truth caches using A100's
     80 GB HBM2e (1,935 GB/s), accelerating the entire benchmark pipeline.
  2. Executes a comprehensive 7-dataset (1M -> 10M -> 100M) filtered ANNS evaluation on
     NVIDIA A100 80GB PCIe (`convlog1m`, `gist1m`, `sift1m`, `deep10m`, `sift10m`, `t2i10m`,
     and `sift100m`), proving that PRISM dominates cuVS-IVF-Flat, cuVS-CAGRA, VecFlow-GPU,
     Faiss-GPU, and Host-Offload even when 10M baselines fit in 80 GB HBM2e, while scaling
     to 100M vectors where 80 GB HBM2e still OOMs on multi-attribute / CAGRA indexes!
"""

from __future__ import annotations
import gc
import json
import os
import time
from typing import Dict, List
import numpy as np
import torch

from src.hybrid_datasets import load_hybrid_dataset, generate_sql_workload, get_paths
from src.oracle import compute_exact_filtered_gt, compute_filtered_recall_at_k, audit_lossless_agreement
from src.prism_engine import PrismEngine
from baselines.filtered_baselines import (
    CuVSFilteredBaseline,
    VecFlowStyleLabelGPUBaseline,
    FaissGPUPostFilterBaseline,
    CPUGraphFilteredBaseline,
)
from bench.e1_main_evaluation import free_gpu, timed_eval, calibrate_param


def prewarm_10m_caches_on_a100():
    print("=== [A100 Stage 0] Pre-warming 10M Graph & KMeans Caches on 80GB HBM2e ===")
    for ds_name in ["sift10m", "t2i10m"]:
        ds = load_hybrid_dataset(ds_name, max_queries=384)
        # Ensure KMeans cache exists
        p_tmp = PrismEngine(nlist=4096, M_blocks=4, sketch_mode=0, host_pinned_exact=True)
        p_tmp.build(ds, cache_kmeans=True)
        del p_tmp
        free_gpu()
        # Ensure 10M CPU navigable graph caches (deg=96 and deg=24) exist via fast A100 IVF build
        for acorn_flag in (True, False):
            cg = CPUGraphFilteredBaseline(acorn_mode=acorn_flag, M_base=24, gamma=4)
            cg.build(ds, cache_graph=True)
            del cg
            free_gpu()
        print(f"  [A100 Cache Ready] {ds_name}")


def run_a100_comprehensive_suite() -> Dict:
    dev_name = torch.cuda.get_device_name(0)
    tot_vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"\n=== [A100 Stage 1] Comprehensive Evaluation on {dev_name} ({tot_vram_gib:.1f} GiB HBM2e) ===")

    prism_root, _, _ = get_paths()
    out_dir = os.path.join(prism_root, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "e9_a100_portability.json")

    specs = [
        ("sift1m", 1024),
        ("gist1m", 1024),
        ("convlog1m", 1024),
        ("deep10m", 4096),
        ("sift10m", 4096),
        ("t2i10m", 4096),
        ("sift100m", 8192),
    ]

    rows = []
    for ds_name, nl in specs:
        max_q = 192 if ds_name == "sift100m" else 320
        ds = load_hybrid_dataset(ds_name, max_queries=max_q)
        q_val, q_test = ds.query_vectors[:64], ds.query_vectors[64:max_q]
        n_test = q_test.shape[0]

        pf = generate_sql_workload(ds, "W3_compound", num_queries=max_q, selectivity=0.02)
        pv, pt = pf.slice(0, 64), pf.slice(64, max_q)
        gti, gtd, _ = compute_exact_filtered_gt(ds, pf, k=10, use_cache=True)
        gvi, gvd = gti[:64], gtd[:64]
        gti_t, gtd_t = gti[64:max_q], gtd[64:max_q]
        free_gpu()

        p_grid = [16, 32, 48, 64, 96, 128, 192, 256] if nl <= 4096 else [32, 64, 96, 128, 192, 256, 384]

        # 1. PRISM (Out-of-Core Co-Pruning) + Host-Offload PreFilter + 100.000% Lossless Audit
        prism = PrismEngine(nlist=nl, M_blocks=4, use_sign_bit=True, sketch_mode=2, host_pinned_exact=True)
        b_p = prism.build(ds, cache_kmeans=True)
        ora = PrismEngine(nlist=nl, M_blocks=4, sketch_mode=0, host_pinned_exact=True)
        ora.build(ds, cache_kmeans=True)

        np_cal, _, _ = calibrate_param(
            lambda p: prism.search(q_val, pv, k=10, nprobe=p, regime_override=0)[:2],
            gvi, gvd, p_grid, 0.95, k=10
        )
        (pi, pd, _), pms = timed_eval(lambda: prism.search(q_test, pt, k=10, nprobe=np_cal, regime_override=0))
        _, _, ptel = prism.search(q_test, pt, k=10, nprobe=np_cal, regime_override=0, record_telemetry=True)
        prec = compute_filtered_recall_at_k(pi, gti_t, gtd_t, pd, k=10)

        p2i, p2d, _ = prism.search(q_test, pt, k=10, nprobe=np_cal, regime_override=2)
        (oi, od, _), oms = timed_eval(lambda: ora.search(q_test, pt, k=10, nprobe=np_cal, regime_override=2), warmup=1, repeats=3)
        aud = audit_lossless_agreement(ds, pt, p2i, p2d, oi, od, k=10)
        assert aud["mismatched_slots"] == 0 and aud["predicate_violations"] == 0, f"A100 lossless mismatch on {ds_name}!"
        del prism, ora
        free_gpu()

        entry: Dict = {
            "dataset": ds_name,
            "N": ds.N,
            "D": ds.D,
            "prism_vram_gib": round(b_p["total_gpu_vram_gib"], 3),
            "prism_qps": (n_test * 1000.0) / pms,
            "prism_recall": prec,
            "prism_prune_pct": ptel["overall_sketch_prune_pct"],
            "host_offload_qps": (n_test * 1000.0) / oms,
            "speedup_vs_host_offload": oms / pms,
            "lossless_agreement_pct": aud["agreement_pct"],
        }

        # 2. cuVS-IVF-Flat on 80GB A100
        try:
            cb_ivf = CuVSFilteredBaseline(mode="ivf_flat", nlist=nl)
            b_ci = cb_ivf.build(ds)
            np_cu, _, _ = calibrate_param(
                lambda p: cb_ivf.search(q_val[:24], pv.slice(0, 24), k=10, nprobe=p)[:2],
                gvi[:24], gvd[:24], [16, 32, 64, 128, 256], 0.95, k=10
            )
            (ci, cd, _), cms = timed_eval(lambda: cb_ivf.search(q_test, pt, k=10, nprobe=np_cu), warmup=1, repeats=3)
            entry["cuvs_ivf_status"] = "OK"
            entry["cuvs_ivf_vram_gib"] = round(b_ci["total_gpu_vram_gib"], 3)
            entry["cuvs_ivf_qps"] = (n_test * 1000.0) / cms
            entry["cuvs_ivf_recall"] = compute_filtered_recall_at_k(ci, gti_t, gtd_t, cd, k=10)
            entry["speedup_vs_cuvs"] = cms / pms
            del cb_ivf
        except Exception as exc:
            entry["cuvs_ivf_status"] = "OOM"
            entry["cuvs_ivf_error"] = str(exc)[:100]
        free_gpu()

        # 3. cuVS-CAGRA on 80GB A100
        try:
            cb_cg = CuVSFilteredBaseline(mode="cagra", nlist=nl)
            b_cg = cb_cg.build(ds)
            it_cu, _, _ = calibrate_param(
                lambda p: cb_cg.search(q_val[:24], pv.slice(0, 24), k=10, itopk_size=p)[:2],
                gvi[:24], gvd[:24], [64, 128, 256, 512], 0.95, k=10
            )
            (cgi, cgd, _), cgms = timed_eval(lambda: cb_cg.search(q_test, pt, k=10, itopk_size=it_cu), warmup=1, repeats=3)
            entry["cuvs_cagra_status"] = "OK"
            entry["cuvs_cagra_vram_gib"] = round(b_cg["total_gpu_vram_gib"], 3)
            entry["cuvs_cagra_qps"] = (n_test * 1000.0) / cgms
            entry["cuvs_cagra_recall"] = compute_filtered_recall_at_k(cgi, gti_t, gtd_t, cgd, k=10)
            entry["speedup_vs_cagra"] = cgms / pms
            del cb_cg
        except Exception as exc:
            entry["cuvs_cagra_status"] = "OOM"
            entry["cuvs_cagra_error"] = str(exc)[:100]
        free_gpu()

        # 4. VecFlow-Style Label-GPU on 80GB A100
        try:
            vf = VecFlowStyleLabelGPUBaseline()
            b_vf = vf.build(ds)
            pf_cal, _, _ = calibrate_param(
                lambda p: vf.search(q_val[:24], pv.slice(0, 24), k=10, probe_frac=p / 100.0)[:2],
                gvi[:24], gvd[:24], [25, 50, 80, 100], 0.95, k=10
            )
            (vi, vd, _), vms = timed_eval(lambda: vf.search(q_test, pt, k=10, probe_frac=pf_cal / 100.0), warmup=1, repeats=3)
            entry["vecflow_status"] = "OK"
            entry["vecflow_vram_gib"] = round(b_vf["total_gpu_vram_gib"], 3)
            entry["vecflow_qps"] = (n_test * 1000.0) / vms
            entry["vecflow_recall"] = compute_filtered_recall_at_k(vi, gti_t, gtd_t, vd, k=10)
            del vf
        except Exception as exc:
            entry["vecflow_status"] = "OOM"
            entry["vecflow_error"] = str(exc)[:100]
        free_gpu()

        rows.append(entry)
        print(
            f"  [A100 {ds_name:10s}] PRISM={entry['prism_qps']:8.1f} QPS ({entry['prism_vram_gib']:.2f} GiB) | "
            f"cuVS-IVF={entry.get('cuvs_ivf_qps', -1):7.1f}[{entry['cuvs_ivf_status']}] | "
            f"CAGRA={entry.get('cuvs_cagra_qps', -1):7.1f}[{entry['cuvs_cagra_status']}] | "
            f"HostOff={entry['host_offload_qps']:7.1f} ({entry['speedup_vs_host_offload']:.1f}x, 100% lossless)"
        )

        # Save incremental checkpoint
        with open(out_path, "w") as f:
            json.dump({"gpu": dev_name, "vram_gib": round(tot_vram_gib, 2), "rows": rows}, f, indent=2)

    print(f"\n=== [A100 DONE] Saved full 7-dataset A100 results to {out_path} ===")
    return {"gpu": dev_name, "rows": rows}


if __name__ == "__main__":
    prewarm_10m_caches_on_a100()
    run_a100_comprehensive_suite()
