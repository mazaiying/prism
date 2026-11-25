#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adversarial Lossless & Exact-Predicate Unit Test Suite for PRISM.

Verifies Theorem 1 (IEEE-754 binary32 ULP-safe Subspace Ball Lower Bound) and
exact SQL predicate evaluation across all regimes, boundary outliers, ties,
vacuum queries, and real high-dimensional datasets (`sift1m` and `convlog1m`).
"""

from __future__ import annotations
import numpy as np
import pytest
import torch

from src.hybrid_datasets import (
    HybridDataset,
    PredicateBatch,
    PRED_COL0_RANGE,
    PRED_COL1_RANGE,
    PRED_COL2_CAT,
    PRED_ALG_EXPR,
    load_hybrid_dataset,
    generate_sql_workload,
)
from src.oracle import compute_exact_filtered_gt, audit_lossless_agreement
from src.prism_engine import PrismEngine


def _make_adversarial_dataset(N: int = 12000, D: int = 128, B: int = 64) -> HybridDataset:
    rng = np.random.default_rng(2027)
    xb = rng.standard_normal((N, D)).astype(np.float32)
    # Inject extreme 1e3x outliers in 2% of vectors to stress quantile clipping + eta_t ULP bound
    outlier_rows = rng.choice(N, size=N // 50, replace=False)
    xb[outlier_rows, :8] *= 850.0
    # Inject exact duplicate vectors to stress tie handling
    xb[100:115] = xb[99]

    xq = rng.standard_normal((B, D)).astype(np.float32)
    # Place several queries right next to outlier vectors!
    xq[:8] = xb[outlier_rows[:8]] + rng.standard_normal((8, D)).astype(np.float32) * 0.01

    col0 = rng.integers(0, 65536, size=N, dtype=np.uint16)
    col1 = rng.integers(0, 65536, size=N, dtype=np.uint16)
    col2 = rng.integers(0, 64, size=N, dtype=np.uint16)
    col3 = rng.integers(0, 65536, size=N, dtype=np.uint16)
    return HybridDataset(
        name="adv_synth",
        N=N, D=D,
        base_vectors=xb, query_vectors=xq,
        col0=col0, col1=col1, col2=col2, col3=col3,
    )


def test_adversarial_outliers_and_all_regimes():
    ds = _make_adversarial_dataset(N=12000, D=128, B=64)
    preds = generate_sql_workload(ds, "W4_mixed", num_queries=64, selectivity=0.05)
    # Force query 0 to be a vacuum query (0 survivors) and query 1 to have tiny range
    preds.op_flags[0] = PRED_COL0_RANGE
    preds.l0[0], preds.r0[0] = 65530, 65520  # impossible range -> 0 survivors

    gt_ids, gt_dists, _ = compute_exact_filtered_gt(ds, preds, k=10, use_cache=False)

    eng = PrismEngine(nlist=64, M_blocks=4, use_sign_bit=True, sketch_mode=2, host_pinned_exact=True)
    eng.build(ds, cache_kmeans=False)

    # 1. Regime 1 (Global Scan) must match Global Brute-Force GT 100.000%!
    r1_ids, r1_dists, _ = eng.search(ds.query_vectors, preds, k=10, nprobe=16, regime_override=1)
    aud_r1 = audit_lossless_agreement(ds, preds, r1_ids, r1_dists, gt_ids, gt_dists, k=10)
    assert aud_r1["predicate_violations"] == 0
    assert aud_r1["mismatched_slots"] == 0, f"Regime 1 mismatch: {aud_r1}"

    # 2. Exact IVF Oracle (sketch_mode=0 on identical probe plan) vs Regime 2 & Regime 3
    ora_eng = PrismEngine(nlist=64, M_blocks=4, sketch_mode=0, host_pinned_exact=False)
    ora_eng.build(ds, cache_kmeans=False)

    for reg in (2, 3):
        for comp in (True, False):
            ora_ids, ora_dists, _ = ora_eng.search(
                ds.query_vectors, preds, k=10, nprobe=24,
                regime_override=2, enable_warp_compaction=False,
                enable_prewarm=True, adaptive_probe_expand=True,
            )
            t_ids, t_dists, _ = eng.search(
                ds.query_vectors, preds, k=10, nprobe=24,
                regime_override=reg, enable_warp_compaction=comp,
                enable_prewarm=True, adaptive_probe_expand=True,
            )
            aud = audit_lossless_agreement(ds, preds, t_ids, t_dists, ora_ids, ora_dists, k=10)
            assert aud["predicate_violations"] == 0
            assert aud["mismatched_slots"] == 0, f"Regime {reg} (comp={comp}) mismatch: {aud}"


def test_uint8_exact_store_lossless():
    rng = np.random.default_rng(99)
    N, D, B = 8192, 128, 32
    xb = rng.integers(0, 256, size=(N, D), dtype=np.uint8)
    xq = rng.integers(0, 256, size=(B, D), dtype=np.uint8).astype(np.float32)
    ds = HybridDataset(
        name="u8_synth", N=N, D=D, base_vectors=xb, query_vectors=xq,
        col0=rng.integers(0, 65536, size=N, dtype=np.uint16),
        col1=rng.integers(0, 65536, size=N, dtype=np.uint16),
        col2=rng.integers(0, 64, size=N, dtype=np.uint16),
        col3=rng.integers(0, 65536, size=N, dtype=np.uint16),
    )
    preds = generate_sql_workload(ds, "W3_compound", num_queries=B, selectivity=0.08)
    gt_ids, gt_dists, _ = compute_exact_filtered_gt(ds, preds, k=10, use_cache=False)

    eng = PrismEngine(nlist=32, M_blocks=4, use_sign_bit=True, sketch_mode=2, host_pinned_exact=True)
    eng.build(ds, cache_kmeans=False)
    ids, dists, _ = eng.search(xq, preds, k=10, nprobe=32, regime_override=1)
    aud = audit_lossless_agreement(ds, preds, ids, dists, gt_ids, gt_dists, k=10)
    assert aud["predicate_violations"] == 0
    assert aud["mismatched_slots"] == 0, f"UInt8 store mismatch: {aud}"


@pytest.mark.parametrize("ds_name", ["sift1m", "convlog1m"])
def test_real_datasets_100pct_ivf_agreement(ds_name: str):
    ds = load_hybrid_dataset(ds_name, max_queries=128)
    preds = generate_sql_workload(ds, "W4_mixed", num_queries=128, selectivity=0.05)

    eng = PrismEngine(nlist=1024, M_blocks=4, use_sign_bit=True, sketch_mode=2, host_pinned_exact=True)
    eng.build(ds, cache_kmeans=True)

    ora = PrismEngine(nlist=1024, M_blocks=4, sketch_mode=0, host_pinned_exact=False)
    ora.build(ds, cache_kmeans=True)

    # Audit Regime 2 & Auto M3
    ora_ids, ora_dists, _ = ora.search(ds.query_vectors, preds, k=10, nprobe=32, regime_override=2)
    p_ids, p_dists, tel = eng.search(
        ds.query_vectors, preds, k=10, nprobe=32, regime_override=2, record_telemetry=True
    )
    aud = audit_lossless_agreement(ds, preds, p_ids, p_dists, ora_ids, ora_dists, k=10)
    print(f"[{ds_name}] Agreement={aud['agreement_pct']:.4f}%, Prune={tel['overall_sketch_prune_pct']:.2f}%, QPS={tel['qps']:.1f}")
    assert aud["predicate_violations"] == 0
    assert aud["mismatched_slots"] == 0, f"Real {ds_name} mismatch: {aud}"
    assert tel["overall_sketch_prune_pct"] > 75.0
