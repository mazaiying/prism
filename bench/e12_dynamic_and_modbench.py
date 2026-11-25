#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advisor-Standard Upgrade Benchmarks E12 & E13:
  - E12: Dynamic Incremental Insert & Tombstone Delete Microbenchmark (`Section 5.4` verification)
         Evaluates 80% initial build -> 5% O(1) GPU tombstone delete -> 4 streaming +5% insert waves
         (+20% total) -> +5% Out-of-Distribution (+1.5 sigma) shifted insert wave across
         `convlog1m`, `modbench`, `gist1m`, and `sift10m`, verifying 100.000% lossless agreement!
  - E13: Full Equal-Recall (`R95` & `R99`) Benchmark of 2nd Real Organic Tabular Dataset `ModBench`
         (`134,556 x 384D` UC Berkeley D-Lab human-annotated moderation records) across all 9 methods.
"""

from __future__ import annotations
import gc
import json
import os
import time
from typing import Callable, Dict, List, Tuple
import numpy as np
import torch

from src.hybrid_datasets import HybridDataset, load_hybrid_dataset, generate_sql_workload, get_paths
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


def trimmed_mean_eval(fn: Callable, warmup: int = 2, repeats: int = 7):
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
    s = sorted(lats)
    mid = s[1:-1] if len(s) >= 5 else s
    return out, float(np.mean(mid))


def run_e12_dynamic_updates(out_path: str) -> Dict:
    dyn_datasets = [
        ("convlog1m", 1024, 256),
        ("modbench", 512, 192),
        ("gist1m", 1024, 160),
        ("sift10m", 4096, 128),
    ]
    results: Dict[str, Dict] = {}
    grand_audits = 0
    grand_mismatches = 0
    grand_tombstone_leaks = 0

    if os.path.exists(out_path):
        try:
            old = json.load(open(out_path))
            results = old.get("datasets", {})
            grand_audits = int(old.get("grand_audits", 0))
            grand_mismatches = int(old.get("grand_mismatches", 0))
            grand_tombstone_leaks = int(old.get("grand_tombstone_leaks", 0))
        except Exception:
            pass

    for ds_name, nlist, nprobe in dyn_datasets:
        if ds_name in results and len(results[ds_name].get("checkpoints", [])) >= 7:
            print(f"[Skip completed E12 dataset] {ds_name}", flush=True)
            continue
        print(f"\n==================== E12 Dynamic Updates: {ds_name} ====================", flush=True)
        ds = load_hybrid_dataset(ds_name, max_queries=256)
        B = min(256, ds.query_vectors.shape[0])
        xq = ds.query_vectors[:B]
        N_full = ds.N
        D = ds.D

        # Deterministically permute rows so 80% initial / 20% streaming split is unbiased
        rng = np.random.default_rng(2027)
        perm_all = rng.permutation(N_full)
        xb_perm = np.asarray(ds.base_vectors, dtype=np.float32)[perm_all]
        c0_perm = ds.col0[perm_all]
        c1_perm = ds.col1[perm_all]
        c2_perm = ds.col2[perm_all]
        c3_perm = ds.col3[perm_all]

        N_init = int(N_full * 0.80)
        N_step = (N_full - N_init) // 4  # Four +5% streaming waves

        # Also prepare an additional +5% Out-of-Distribution (+1.5 sigma) shifted wave!
        N_ood = max(2000, int(N_full * 0.05))
        xb_std = float(xb_perm[: min(N_init, 32768)].std())
        xb_ood = (
            xb_perm[:N_ood]
            + rng.standard_normal((N_ood, D)).astype(np.float32) * (1.5 * xb_std)
        ).astype(np.float32)
        c0_ood = c0_perm[:N_ood].copy()
        c1_ood = c1_perm[:N_ood].copy()
        c2_ood = c2_perm[:N_ood].copy()
        c3_ood = c3_perm[:N_ood].copy()

        xb_ext = np.concatenate([xb_perm, xb_ood], axis=0)
        c0_ext = np.concatenate([c0_perm, c0_ood])
        c1_ext = np.concatenate([c1_perm, c1_ood])
        c2_ext = np.concatenate([c2_perm, c2_ood])
        c3_ext = np.concatenate([c3_perm, c3_ood])

        ds_ext = HybridDataset(
            name=f"{ds_name}_ext", N=N_full + N_ood, D=D,
            base_vectors=xb_ext, query_vectors=xq,
            col0=c0_ext, col1=c1_ext, col2=c2_ext, col3=c3_ext,
        )
        preds = generate_sql_workload(ds, "W3_compound", num_queries=B, selectivity=0.02)

        ds_init = HybridDataset(
            name=f"{ds_name}_init80p", N=N_init, D=D,
            base_vectors=xb_ext[:N_init], query_vectors=xq,
            col0=c0_ext[:N_init], col1=c1_ext[:N_init], col2=c2_ext[:N_init], col3=c3_ext[:N_init],
        )

        # Build PRISM & identical-cluster Host-Offload Oracle with mutable headroom
        eng_p = PrismEngine(nlist=nlist, M_blocks=4, sketch_mode=2, host_pinned_exact=True)
        b_p = eng_p.build(ds_init, cache_kmeans=True, mutable_headroom=0.65)

        eng_o = PrismEngine(nlist=nlist, M_blocks=4, sketch_mode=0, host_pinned_exact=True)
        eng_o.build(ds_init, cache_kmeans=True, mutable_headroom=0.65)

        # Measure full CAGRA rebuild cost on ds_init for amortization comparison (skip if >VRAM)
        cagra_rebuild_s = None
        try:
            cg = CuVSFilteredBaseline(mode="cagra", nlist=nlist)
            cagra_rebuild_s = cg.build(ds_init)["total_build_s"]
            del cg
            free_gpu()
        except Exception:
            free_gpu()

        del_set: set[int] = set()
        checkpoints: List[Dict] = []

        def _audit_checkpoint(tag: str, cur_limit: int, op_ms: float = 0.0, op_ips: float = 0.0):
            nonlocal grand_audits, grand_mismatches, grand_tombstone_leaks
            (pi, pd, _), ms_q = trimmed_mean_eval(
                lambda: eng_p.search(xq, preds, k=10, nprobe=nprobe, regime_override=2, adaptive_probe_expand=False)
            )
            _, _, tel = eng_p.search(xq, preds, k=10, nprobe=nprobe, regime_override=2, adaptive_probe_expand=False, record_telemetry=True)
            oi, od, _ = eng_o.search(xq, preds, k=10, nprobe=nprobe, regime_override=2, adaptive_probe_expand=False)

            leaked = sum(1 for x in pi.ravel() if int(x) in del_set)
            grand_tombstone_leaks += leaked
            assert leaked == 0, f"Tombstone leak at {tag}: {leaked}"

            rep = audit_lossless_agreement(ds_ext, preds, pi, pd, oi, od, k=10)
            grand_audits += 1
            grand_mismatches += rep["mismatched_slots"]
            assert rep["mismatched_slots"] == 0, f"Mismatch at {tag}: {rep}"
            assert rep["predicate_violations"] == 0, f"Violation at {tag}: {rep}"

            # Compute true Recall@10 against live exact set (for N <= 1.2M; sample 64 queries on 10M for speed)
            q_eval_n = B if cur_limit <= 1_500_000 else 64
            live_mask = np.ones(cur_limit, dtype=bool)
            if del_set:
                live_mask[np.array(list(del_set), dtype=np.int64)] = False
            live_idx = np.nonzero(live_mask)[0]
            ds_live = HybridDataset(
                name=f"{ds_name}_{tag}", N=int(live_idx.size), D=D,
                base_vectors=xb_ext[live_idx], query_vectors=xq[:q_eval_n],
                col0=c0_ext[live_idx], col1=c1_ext[live_idx], col2=c2_ext[live_idx], col3=c3_ext[live_idx],
            )
            gt_loc, gt_d_live, _ = compute_exact_filtered_gt(ds_live, preds.slice(0, q_eval_n), k=10, use_cache=False)
            gt_orig = np.where(gt_loc >= 0, live_idx[np.maximum(0, gt_loc)], -1)
            rec = compute_filtered_recall_at_k(pi[:q_eval_n], gt_orig, gt_d_live, pd[:q_eval_n], k=10)

            entry = {
                "stage": tag,
                "live_rows": int(live_idx.size),
                "update_ms": round(op_ms, 3),
                "update_rows_per_s": round(op_ips, 1),
                "search_qps": round(B * 1000.0 / ms_q, 2),
                "recall_at_10": round(rec, 4),
                "sketch_prune_pct": round(tel["overall_sketch_prune_pct"], 2),
                "lossless_agreement_pct": rep["agreement_pct"],
                "mismatched_slots": rep["mismatched_slots"],
                "tombstone_leaks": leaked,
            }
            checkpoints.append(entry)
            print(f"  [{ds_name} | {tag}] QPS={entry['search_qps']}, Prune={entry['sketch_prune_pct']}%, "
                  f"R@10={entry['recall_at_10']}, op_ms={op_ms:.2f}ms, ips={op_ips:,.0f}/s, agree=100.0%")

        # Checkpoint 0: Initial 80%
        _audit_checkpoint("0_Init_80pct", N_init, op_ms=b_p["total_build_s"] * 1000.0, op_ips=N_init / b_p["total_build_s"])

        # Checkpoint 1: O(1) GPU Tombstone Delete of 5% rows
        n_del = int(N_full * 0.05)
        del_ids = rng.choice(N_init, size=n_del, replace=False).astype(np.int64)
        del_set.update(del_ids.tolist())
        d_res = eng_p.delete_rows(del_ids)
        eng_o.delete_rows(del_ids)
        _audit_checkpoint("1_Del_5pct", N_init, op_ms=d_res["delete_ms"], op_ips=d_res["delete_dps"])

        # Checkpoints 2..5: Four +5% Streaming Insertions (up to 100% N_full)
        cur = N_init
        for w_idx in range(1, 5):
            nxt = N_full if w_idx == 4 else (cur + N_step)
            oids = np.arange(cur, nxt, dtype=np.int64)
            ins_res = eng_p.insert_batch(xb_ext[cur:nxt], c0_ext[cur:nxt], c1_ext[cur:nxt], c2_ext[cur:nxt], c3_ext[cur:nxt], oids)
            eng_o.insert_batch(xb_ext[cur:nxt], c0_ext[cur:nxt], c1_ext[cur:nxt], c2_ext[cur:nxt], c3_ext[cur:nxt], oids)
            cur = nxt
            _audit_checkpoint(f"{w_idx + 1}_Ins_+{w_idx * 5}pct", cur, op_ms=ins_res["insert_ms"], op_ips=ins_res["insert_ips"])

        # Checkpoint 6: Out-of-Distribution (+1.5 sigma drift) +5% Insertion
        oids_ood = np.arange(N_full, N_full + N_ood, dtype=np.int64)
        ins_ood = eng_p.insert_batch(xb_ext[N_full:], c0_ext[N_full:], c1_ext[N_full:], c2_ext[N_full:], c3_ext[N_full:], oids_ood)
        eng_o.insert_batch(xb_ext[N_full:], c0_ext[N_full:], c1_ext[N_full:], c2_ext[N_full:], c3_ext[N_full:], oids_ood)
        _audit_checkpoint("6_Ins_OOD_Drift", N_full + N_ood, op_ms=ins_ood["insert_ms"], op_ips=ins_ood["insert_ips"])

        results[ds_name] = {
            "N_initial": N_init,
            "N_deleted": n_del,
            "N_inserted_in_dist": N_full - N_init,
            "N_inserted_ood": N_ood,
            "initial_build_s": round(b_p["total_build_s"], 3),
            "cagra_single_rebuild_s": round(cagra_rebuild_s, 3) if cagra_rebuild_s else None,
            "total_incremental_update_s": round(sum(c["update_ms"] for c in checkpoints[1:]) / 1000.0, 4),
            "checkpoints": checkpoints,
        }
        with open(out_path, "w") as f:
            json.dump({
                "gpu": torch.cuda.get_device_name(0),
                "grand_audits": grand_audits,
                "grand_mismatches": grand_mismatches,
                "grand_tombstone_leaks": grand_tombstone_leaks,
                "datasets": results,
            }, f, indent=2)
        del eng_p, eng_o
        free_gpu()

    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "grand_audits": grand_audits,
        "grand_mismatches": grand_mismatches,
        "grand_tombstone_leaks": grand_tombstone_leaks,
        "datasets": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[DONE E12] Saved {out_path} (audits={grand_audits}, mismatches={grand_mismatches}, leaks={grand_tombstone_leaks})")
    return payload


def run_e13_modbench_eval(out_path: str) -> Dict:
    print("\n==================== E13 2nd Real Dataset: ModBench (135K x 384D) ====================", flush=True)
    ds = load_hybrid_dataset("modbench", max_queries=384)
    nlist = 512
    n_val, n_test = 128, 256
    q_val = ds.query_vectors[:n_val]
    q_test = ds.query_vectors[n_val : n_val + n_test]
    probe_grid = [4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]

    workloads_cfg = [
        ("W1_range", 0.05),
        ("W2_correlated", 0.02),
        ("W3_compound", 0.02),
        ("W4_mixed", 0.05),
    ]
    wl_packs = {}
    for wl_name, sel in workloads_cfg:
        pf = generate_sql_workload(ds, wl_name, num_queries=n_val + n_test, selectivity=sel)
        gt_i, gt_d, surv = compute_exact_filtered_gt(ds, pf, k=10, use_cache=True)
        wl_packs[wl_name] = {
            "emp_sel": float(surv.mean() / ds.N),
            "preds_val": pf.slice(0, n_val),
            "preds_test": pf.slice(n_val, n_val + n_test),
            "gt_val_i": gt_i[:n_val], "gt_val_d": gt_d[:n_val],
            "gt_test_i": gt_i[n_val:], "gt_test_d": gt_d[n_val:],
        }

    def _cal(fn_val, gt_vi, gt_vd, cands, target_r):
        best_p, best_r = cands[-1], 0.0
        for p in cands:
            pi, pd = fn_val(p)
            r = compute_filtered_recall_at_k(pi, gt_vi, gt_vd, pd, k=10)
            if r > best_r:
                best_r, best_p = r, p
            if r >= target_r:
                return p, r
        return best_p, best_r

    eng_p = PrismEngine(nlist=nlist, M_blocks=4, sketch_mode=2, host_pinned_exact=True)
    bm_p = eng_p.build(ds, cache_kmeans=True)
    eng_h = PrismEngine(nlist=nlist, M_blocks=4, sketch_mode=0, host_pinned_exact=True)
    bm_h = eng_h.build(ds, cache_kmeans=True)
    eng_b1 = PrismEngine(nlist=nlist, M_blocks=4, sketch_mode=0, host_pinned_exact=False)
    bm_b1 = eng_b1.build(ds, cache_kmeans=True)

    cuvs_ivf = CuVSFilteredBaseline(mode="ivf_flat", nlist=nlist)
    bm_ivf = cuvs_ivf.build(ds)
    cuvs_cg = CuVSFilteredBaseline(mode="cagra", nlist=nlist)
    bm_cg = cuvs_cg.build(ds)
    faiss_pf = FaissGPUPostFilterBaseline(nlist=nlist)
    bm_fp = faiss_pf.build(ds)
    vecflow = VecFlowStyleLabelGPUBaseline()
    bm_vf = vecflow.build(ds)
    acorn = CPUGraphFilteredBaseline(acorn_mode=True)
    bm_ac = acorn.build(ds, cache_graph=True)
    hnsw = CPUGraphFilteredBaseline(acorn_mode=False)
    bm_hn = hnsw.build(ds, cache_graph=True)

    res_wls = {}
    for wl_name, _ in workloads_cfg:
        wp = wl_packs[wl_name]
        res_wls[wl_name] = {"empirical_sel": wp["emp_sel"]}
        for r_tgt, r_key in [(0.95, "R95"), (0.99, "R99")]:
            tier = {}
            p_opt, _ = _cal(lambda p: eng_p.search(q_val, wp["preds_val"], k=10, nprobe=p)[:2], wp["gt_val_i"], wp["gt_val_d"], probe_grid, r_tgt)
            (pi, pd, _), ms_p = trimmed_mean_eval(lambda: eng_p.search(q_test, wp["preds_test"], k=10, nprobe=p_opt))
            _, _, tel_p = eng_p.search(q_test, wp["preds_test"], k=10, nprobe=p_opt, record_telemetry=True)
            (hi, hd, _), ms_h = trimmed_mean_eval(lambda: eng_h.search(q_test, wp["preds_test"], k=10, nprobe=p_opt))
            (b1i, b1d, _), ms_b1 = trimmed_mean_eval(lambda: eng_b1.search(q_test, wp["preds_test"], k=10, nprobe=p_opt))

            # Lossless agreement check
            pi2, pd2, _ = eng_p.search(q_test, wp["preds_test"], k=10, nprobe=p_opt, regime_override=2)
            hi2, hd2, _ = eng_h.search(q_test, wp["preds_test"], k=10, nprobe=p_opt, regime_override=2)
            rep = audit_lossless_agreement(ds, wp["preds_test"], pi2, pd2, hi2, hd2, k=10)
            assert rep["mismatched_slots"] == 0

            tier["PRISM"] = {
                "param": p_opt, "qps": n_test * 1000.0 / ms_p,
                "recall_at_10": compute_filtered_recall_at_k(pi, wp["gt_test_i"], wp["gt_test_d"], pd, k=10),
                "sketch_prune_pct": tel_p["overall_sketch_prune_pct"],
                "ivf_agreement_pct": rep["agreement_pct"],
            }
            tier["B7_HostOffload"] = {
                "param": p_opt, "qps": n_test * 1000.0 / ms_h,
                "recall_at_10": compute_filtered_recall_at_k(hi, wp["gt_test_i"], wp["gt_test_d"], hd, k=10),
            }
            tier["B1_InVRAM"] = {
                "param": p_opt, "qps": n_test * 1000.0 / ms_b1,
                "recall_at_10": compute_filtered_recall_at_k(b1i, wp["gt_test_i"], wp["gt_test_d"], b1d, k=10),
            }

            p_ivf, _ = _cal(lambda p: cuvs_ivf.search(q_val, wp["preds_val"], k=10, nprobe=p)[:2], wp["gt_val_i"], wp["gt_val_d"], probe_grid, r_tgt)
            (ci, cd, _), ms_ivf = trimmed_mean_eval(lambda: cuvs_ivf.search(q_test, wp["preds_test"], k=10, nprobe=p_ivf), warmup=1, repeats=5)
            tier["B2a_cuVS_IVF"] = {
                "param": p_ivf, "qps": n_test * 1000.0 / ms_ivf,
                "recall_at_10": compute_filtered_recall_at_k(ci, wp["gt_test_i"], wp["gt_test_d"], cd, k=10),
            }

            p_cg, _ = _cal(lambda p: cuvs_cg.search(q_val, wp["preds_val"], k=10, itopk_size=p)[:2], wp["gt_val_i"], wp["gt_val_d"], [64, 128, 256, 384, 512], r_tgt)
            (cgi, cgd, _), ms_cg = trimmed_mean_eval(lambda: cuvs_cg.search(q_test, wp["preds_test"], k=10, itopk_size=p_cg), warmup=1, repeats=5)
            tier["B2b_cuVS_CAGRA"] = {
                "param": p_cg, "qps": n_test * 1000.0 / ms_cg,
                "recall_at_10": compute_filtered_recall_at_k(cgi, wp["gt_test_i"], wp["gt_test_d"], cgd, k=10),
            }

            p_fp, _ = _cal(lambda p: faiss_pf.search(q_val, wp["preds_val"], k=10, nprobe=p)[:2], wp["gt_val_i"], wp["gt_val_d"], probe_grid, r_tgt)
            (fpi, fpd, _), ms_fp = trimmed_mean_eval(lambda: faiss_pf.search(q_test, wp["preds_test"], k=10, nprobe=p_fp), warmup=1, repeats=5)
            tier["B3_Faiss_PostFilter"] = {
                "param": p_fp, "qps": n_test * 1000.0 / ms_fp,
                "recall_at_10": compute_filtered_recall_at_k(fpi, wp["gt_test_i"], wp["gt_test_d"], fpd, k=10),
            }

            p_vf, _ = _cal(lambda p: vecflow.search(q_val, wp["preds_val"], k=10, probe_frac=p / 100.0)[:2], wp["gt_val_i"], wp["gt_val_d"], [15, 30, 50, 75, 100], r_tgt)
            (vfi, vfd, _), ms_vf = trimmed_mean_eval(lambda: vecflow.search(q_test, wp["preds_test"], k=10, probe_frac=p_vf / 100.0), warmup=1, repeats=5)
            tier["B4_VecFlow_GPU"] = {
                "param": p_vf, "qps": n_test * 1000.0 / ms_vf,
                "recall_at_10": compute_filtered_recall_at_k(vfi, wp["gt_test_i"], wp["gt_test_d"], vfd, k=10),
            }

            p_ac, _ = _cal(lambda p: acorn.search(q_val, wp["preds_val"], k=10, ef_search=p)[:2], wp["gt_val_i"], wp["gt_val_d"], [64, 128, 256, 512, 800], r_tgt)
            (aci, acd, _), ms_ac = trimmed_mean_eval(lambda: acorn.search(q_test, wp["preds_test"], k=10, ef_search=p_ac), warmup=1, repeats=5)
            tier["B5_ACORN_48T"] = {
                "param": p_ac, "qps": n_test * 1000.0 / ms_ac,
                "recall_at_10": compute_filtered_recall_at_k(aci, wp["gt_test_i"], wp["gt_test_d"], acd, k=10),
            }

            p_hn, _ = _cal(lambda p: hnsw.search(q_val, wp["preds_val"], k=10, ef_search=p)[:2], wp["gt_val_i"], wp["gt_val_d"], [64, 128, 256, 512, 800], r_tgt)
            (hni, hnd, _), ms_hn = trimmed_mean_eval(lambda: hnsw.search(q_test, wp["preds_test"], k=10, ef_search=p_hn), warmup=1, repeats=5)
            tier["B6_HNSW_48T"] = {
                "param": p_hn, "qps": n_test * 1000.0 / ms_hn,
                "recall_at_10": compute_filtered_recall_at_k(hni, wp["gt_test_i"], wp["gt_test_d"], hnd, k=10),
            }

            res_wls[wl_name][r_key] = tier
            print(f"  [ModBench | {wl_name} | {r_key}] PRISM={tier['PRISM']['qps']:.1f} QPS "
                  f"vs cuVS-IVF={tier['B2a_cuVS_IVF']['qps']:.1f}, CAGRA={tier['B2b_cuVS_CAGRA']['qps']:.1f}, "
                  f"HostOff={tier['B7_HostOffload']['qps']:.1f}, ACORN={tier['B5_ACORN_48T']['qps']:.1f}", flush=True)

    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "N": ds.N,
        "D": ds.D,
        "nlist": nlist,
        "is_real_tabular": True,
        "build_and_memory": {
            "PRISM": bm_p, "B7_HostOffload": bm_h, "B1_InVRAM": bm_b1,
            "B2a_cuVS_IVF": bm_ivf, "B2b_cuVS_CAGRA": bm_cg,
            "B3_Faiss_PostFilter": bm_fp, "B4_VecFlow_GPU": bm_vf,
            "B5_ACORN_48T": bm_ac, "B6_HNSW_48T": bm_hn,
        },
        "workloads": res_wls,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[DONE E13] Saved {out_path}")
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["all", "e12", "e13"], default="all")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    prism_root, _, _ = get_paths()
    out_dir = os.path.join(prism_root, "results")
    os.makedirs(out_dir, exist_ok=True)
    sfx = f"_{args.tag}" if args.tag else ""

    if args.only in ("all", "e13"):
        run_e13_modbench_eval(os.path.join(out_dir, f"e13_modbench_eval{sfx}.json"))
    if args.only in ("all", "e12"):
        run_e12_dynamic_updates(os.path.join(out_dir, f"e12_dynamic_updates{sfx}.json"))
