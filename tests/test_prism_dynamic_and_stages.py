#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit & Adversarial Lossless Verification for PRISM Incremental Updates (`insert_batch` & `delete_rows`)
and 4-Stage Hardware Cycle / CUDA Event Latency Breakdown.
"""

from __future__ import annotations
import numpy as np

from src.hybrid_datasets import HybridDataset, generate_sql_workload
from src.oracle import compute_exact_filtered_gt, audit_lossless_agreement
from src.prism_engine import PrismEngine


def test_incremental_insert_delete_lossless():
    rng = np.random.default_rng(2027)
    N_tot = 24_000
    D = 128
    Q = 64
    centers = rng.standard_normal((64, D)).astype(np.float32) * 3.0
    c_ids = rng.integers(0, 64, size=N_tot)
    xb_all = (centers[c_ids] + rng.standard_normal((N_tot, D)).astype(np.float32) * 0.5).astype(np.float32)
    # Inject +2.5 sigma out-of-distribution drift into the last 2,000 inserted rows
    xb_all[22_000:] += rng.standard_normal((2_000, D)).astype(np.float32) * 2.5

    xq = (centers[rng.integers(0, 64, size=Q)] + rng.standard_normal((Q, D)).astype(np.float32) * 0.5).astype(np.float32)
    col0 = rng.integers(0, 65536, size=N_tot, dtype=np.uint16)
    col1 = rng.integers(0, 65536, size=N_tot, dtype=np.uint16)
    col2 = rng.integers(0, 64, size=N_tot, dtype=np.uint16)
    col3 = rng.integers(0, 65536, size=N_tot, dtype=np.uint16)

    N_init = 18_000
    ds_init = HybridDataset(
        name="unit_dyn_v2", N=N_init, D=D,
        base_vectors=xb_all[:N_init], query_vectors=xq,
        col0=col0[:N_init], col1=col1[:N_init], col2=col2[:N_init], col3=col3[:N_init],
    )

    eng_prism = PrismEngine(nlist=128, M_blocks=4, sketch_mode=2, host_pinned_exact=True)
    eng_prism.build(ds_init, cache_kmeans=True, mutable_headroom=0.60)
    eng_oracle = PrismEngine(nlist=128, M_blocks=4, sketch_mode=0, host_pinned_exact=True)
    eng_oracle.build(ds_init, cache_kmeans=True, mutable_headroom=0.60)

    # 1. Delete 1,200 random rows
    del_ids = rng.choice(N_init, size=1200, replace=False).astype(np.int64)
    del_set = set(del_ids.tolist())
    d_stat = eng_prism.delete_rows(del_ids)
    eng_oracle.delete_rows(del_ids)
    assert d_stat["deleted_count"] == 1200

    # 2. Insert remaining 6,000 rows (including 2,000 OOD drifted rows) in 2 waves
    for s, e in [(18_000, 21_000), (21_000, 24_000)]:
        oids = np.arange(s, e, dtype=np.int64)
        eng_prism.insert_batch(xb_all[s:e], col0[s:e], col1[s:e], col2[s:e], col3[s:e], oids)
        eng_oracle.insert_batch(xb_all[s:e], col0[s:e], col1[s:e], col2[s:e], col3[s:e], oids)

    ds_full = HybridDataset(
        name="unit_dyn_full", N=N_tot, D=D,
        base_vectors=xb_all, query_vectors=xq,
        col0=col0, col1=col1, col2=col2, col3=col3,
    )
    preds = generate_sql_workload(ds_full, "W3_compound", num_queries=Q, selectivity=0.05)

    # Verify Regime 1, Regime 2, and Regime 3 against identical-cluster exact oracle!
    for reg in (1, 2, 3):
        ids_p, d_p, info_p = eng_prism.search(xq, preds, k=10, nprobe=64, regime_override=reg, record_telemetry=True)
        ids_o, d_o, _ = eng_oracle.search(xq, preds, k=10, nprobe=64, regime_override=1 if reg == 1 else 2, record_telemetry=False)

        leaked = [int(x) for x in ids_p.ravel() if int(x) in del_set]
        assert len(leaked) == 0, f"Regime {reg} tombstone leak: {leaked}"

        tot_stages = (
            info_p["stage_plan_ms"] + info_p["stage_filter_ms"]
            + info_p["stage_sketch_ms"] + info_p["stage_refine_ms"]
        )
        assert abs(tot_stages - info_p["latency_ms"]) < 1e-2, f"Stage sum mismatch: {tot_stages} vs {info_p['latency_ms']}"

        rep = audit_lossless_agreement(ds_full, preds, ids_p, d_p, ids_o, d_o, k=10)
        assert rep["mismatched_slots"] == 0, f"Regime {reg} mismatch: {rep}"
        assert rep["predicate_violations"] == 0, f"Regime {reg} pred violation: {rep}"

    # Also verify Regime 1 against brute-force CPU oracle over live non-deleted rows
    live_mask = np.ones(N_tot, dtype=bool)
    live_mask[del_ids] = False
    live_idx = np.nonzero(live_mask)[0]
    ds_live = HybridDataset(
        name="unit_dyn_live", N=int(live_idx.size), D=D,
        base_vectors=xb_all[live_idx], query_vectors=xq,
        col0=col0[live_idx], col1=col1[live_idx], col2=col2[live_idx], col3=col3[live_idx],
    )
    gt_local_i, gt_d, _ = compute_exact_filtered_gt(ds_live, preds, k=10, use_cache=False)
    gt_orig_i = np.where(gt_local_i >= 0, live_idx[np.maximum(0, gt_local_i)], -1)
    ids_r1, d_r1, _ = eng_prism.search(xq, preds, k=10, nprobe=64, regime_override=1)
    rep_bf = audit_lossless_agreement(ds_full, preds, ids_r1, d_r1, gt_orig_i, gt_d, k=10)
    assert rep_bf["mismatched_slots"] == 0
    assert rep_bf["predicate_violations"] == 0
    print("[PASS] All 3 regimes + OOD drift + tombstone delete + 4-stage breakdown 100.000% lossless!", rep_bf)


if __name__ == "__main__":
    test_incremental_insert_delete_lossless()
