#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark E15 (Prof. Feng Zhang HARMONY Table 5 + TensorSlim + Relational DBMS Audit):
  1. Static Index VRAM vs. Query-Time Runtime Peak GPU Memory (`Peak VRAM`) & Exact PCIe
     Vector Transfer Volume (`MiB / 256-query batch`) across 135K-100M datasets on RTX 4080.
  2. Real 48-Thread SQL Relational/Vector DBMS Evaluation (`DuckDB 1.5.6` Columnar SQL Engine +
     `DuckDB vss` HNSW Vector Index) executing arbitrary 4-column SQL + Top-10 L2 similarity
     queries across `modbench`, `sift1m`, `gist1m`, `convlog1m`, and `deep10m` (10M).
"""

from __future__ import annotations
import gc
import json
import os
import sys
import time
from typing import Dict
import numpy as np
import pyarrow as pa
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.hybrid_datasets import load_hybrid_dataset, generate_sql_workload, get_paths
from src.oracle import compute_exact_filtered_gt, compute_filtered_recall_at_k
from src.prism_engine import PrismEngine
from baselines.filtered_baselines import CuVSFilteredBaseline, FaissGPUPostFilterBaseline


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def run_peak_vram_and_pcie_suite(res_dir: str) -> Dict:
    specs = [
        ("modbench", 512),
        ("sift1m", 1024),
        ("gist1m", 1024),
        ("convlog1m", 1024),
        ("deep10m", 4096),
        ("sift10m", 4096),
        ("t2i10m", 4096),
        ("sift100m", 8192),
    ]
    e14_vf_path = os.path.join(res_dir, "e14_official_vecflow.json")
    vf_db = json.load(open(e14_vf_path)).get("datasets", {}) if os.path.exists(e14_vf_path) else {}

    out_rows = {}
    for ds_name, nl in specs:
        print(f"\n=== [E15 Peak VRAM & PCIe Bytes] {ds_name} ===", flush=True)
        max_q = 192 if ds_name == "sift100m" else 384
        n_val = 64 if ds_name == "sift100m" else 128
        n_test = max_q - n_val
        ds = load_hybrid_dataset(ds_name, max_queries=max_q)
        q_val, q_test = ds.query_vectors[:n_val], ds.query_vectors[n_val:max_q]
        pf = generate_sql_workload(ds, "W3_compound", num_queries=max_q, selectivity=0.02)
        pv, pt = pf.slice(0, n_val), pf.slice(n_val, max_q)
        gti, gtd, _ = compute_exact_filtered_gt(ds, pf, k=10, use_cache=True)
        free_gpu()

        elem_bytes = 1 if ds.base_vectors.dtype == np.uint8 else 4
        raw_knobs = [16, 32, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512, nl]
        p_knobs = sorted(set(k for k in raw_knobs if k <= nl))

        # 1. PRISM: Static VRAM, Runtime Peak VRAM, Workspace MiB, Exact PCIe Fetch MiB
        free_gpu()
        prism = PrismEngine(nlist=nl, M_blocks=4, use_sign_bit=True, sketch_mode=2, host_pinned_exact=True)
        b_p = prism.build(ds, cache_kmeans=True)
        static_p_gib = float(b_p["total_gpu_vram_gib"])
        best_np = p_knobs[-1]
        for np_k in p_knobs:
            vi, vd, _ = prism.search(q_val, pv, k=10, nprobe=np_k)
            if compute_filtered_recall_at_k(vi, gti[:n_val], gtd[:n_val], vd, k=10) >= 0.962:
                best_np = np_k
                break

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_base_alloc = torch.cuda.memory_allocated()
        prism.search(q_test, pt, k=10, nprobe=best_np)
        torch.cuda.synchronize()
        peak_delta_gib = max(0.0, (torch.cuda.max_memory_allocated() - mem_base_alloc) / (1024 ** 3))
        peak_p_gib = static_p_gib + peak_delta_gib
        _, _, tel_p = prism.search(q_test, pt, k=10, nprobe=best_np, record_telemetry=True)
        fetches_p = int(tel_p.get("host_fp32_fetches", 0))
        pcie_p_mib = (fetches_p * ds.D * elem_bytes) / (1024 ** 2)
        del prism
        free_gpu()

        # 2. Host-Offload PreFilter: Static VRAM, Peak VRAM, Exact PCIe Fetch MiB
        b7 = PrismEngine(nlist=nl, M_blocks=4, sketch_mode=0, host_pinned_exact=True)
        b_h = b7.build(ds, cache_kmeans=True)
        static_h_gib = float(b_h["total_gpu_vram_gib"])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_h_base = torch.cuda.memory_allocated()
        b7.search(q_test, pt, k=10, nprobe=best_np, regime_override=2)
        torch.cuda.synchronize()
        peak_h_gib = static_h_gib + max(0.0, (torch.cuda.max_memory_allocated() - mem_h_base) / (1024 ** 3))
        _, _, tel_h = b7.search(q_test, pt, k=10, nprobe=best_np, regime_override=2, record_telemetry=True)
        fetches_h = int(tel_h.get("host_fp32_fetches", 0))
        pcie_h_mib = (fetches_h * ds.D * elem_bytes) / (1024 ** 2)
        del b7
        free_gpu()

        row = {
            "N": ds.N,
            "D": ds.D,
            "batch_size": n_test,
            "PRISM": {
                "static_vram_gib": round(static_p_gib, 4),
                "peak_vram_gib": round(peak_p_gib, 4),
                "runtime_workspace_mib": round(peak_delta_gib * 1024.0, 2),
                "host_pinned_gib": round(float(b_p["host_pinned_gib"]), 4),
                "pcie_vector_fetches": fetches_p,
                "pcie_transfer_mib": round(pcie_p_mib, 2),
            },
            "Host_Offload": {
                "static_vram_gib": round(static_h_gib, 4),
                "peak_vram_gib": round(peak_h_gib, 4),
                "pcie_vector_fetches": fetches_h,
                "pcie_transfer_mib": round(pcie_h_mib, 2),
                "pcie_reduction_x": round(pcie_h_mib / max(1e-4, pcie_p_mib), 2),
            },
        }

        # 3. cuVS-IVF-Flat & cuVS-CAGRA Runtime Peak VRAM on 16GB RTX 4080
        for m_mode, k_lbl in [("ivf_flat", "cuVS_IVF_Flat"), ("cagra", "cuVS_CAGRA")]:
            try:
                cb = CuVSFilteredBaseline(mode=m_mode, nlist=nl)
                b_c = cb.build(ds)
                _, _, cinfo = cb.search(q_test, pt, k=10, nprobe=64, itopk_size=128)
                row[k_lbl] = {
                    "status": "OK",
                    "static_vram_gib": round(float(b_c["total_gpu_vram_gib"]), 4),
                    "peak_vram_gib": round(float(cinfo["peak_vram_gib"]), 4),
                }
                del cb
            except Exception as exc:
                row[k_lbl] = {"status": "OOM", "error": str(exc)[:90]}
            free_gpu()

        # 4. Faiss-GPU PostFilter Runtime Peak VRAM
        try:
            fb = FaissGPUPostFilterBaseline(nlist=nl)
            b_f = fb.build(ds)
            torch.cuda.reset_peak_memory_stats()
            fb.search(q_test, pt, k=10, nprobe=64)
            torch.cuda.synchronize()
            row["Faiss_GPU"] = {
                "status": "OK",
                "static_vram_gib": round(float(b_f["total_gpu_vram_gib"]), 4),
                "peak_vram_gib": round(max(float(b_f["total_gpu_vram_gib"]), torch.cuda.max_memory_allocated() / (1024 ** 3)), 4),
            }
            del fb
        except Exception as exc:
            row["Faiss_GPU"] = {"status": "OOM", "error": str(exc)[:90]}
        free_gpu()

        if ds_name in vf_db and "vram_gib" in vf_db[ds_name]:
            row["Official_VecFlow_A100"] = {
                "pure_index_vram_gib": vf_db[ds_name]["vram_gib"],
                "total_device_vram_gib": round(vf_db[ds_name].get("total_device_mib", vf_db[ds_name]["vram_gib"] * 1024) / 1024.0, 3),
                "fits_16gb_4080": bool(vf_db[ds_name]["vram_gib"] <= 15.57),
            }

        out_rows[ds_name] = row
        print(
            f"  [{ds_name}] PRISM static={row['PRISM']['static_vram_gib']:.3f}G, peak={row['PRISM']['peak_vram_gib']:.3f}G "
            f"(+{row['PRISM']['runtime_workspace_mib']:.1f} MiB), PCIe={row['PRISM']['pcie_transfer_mib']:.1f} MiB vs "
            f"HostOff {row['Host_Offload']['pcie_transfer_mib']:.1f} MiB ({row['Host_Offload']['pcie_reduction_x']:.1f}x cut!)",
            flush=True,
        )
    return out_rows


def run_duckdb_sql_baseline(res_dir: str) -> Dict:
    import duckdb
    out_dbms = {}
    for ds_name in ["modbench", "sift1m", "gist1m", "convlog1m", "deep10m"]:
        print(f"\n=== [E15 DuckDB 1.5.6 48T SQL + VSS HNSW Baseline] {ds_name} ===", flush=True)
        n_val, n_test = 128, 256
        ds = load_hybrid_dataset(ds_name, max_queries=n_val + n_test)
        D = ds.D
        xb = np.ascontiguousarray(ds.base_vectors, dtype=np.float32)
        xq = np.ascontiguousarray(ds.query_vectors[n_val : n_val + n_test], dtype=np.float32)

        con = duckdb.connect(":memory:")
        con.execute("SET threads TO 48;")
        con.execute("LOAD vss;")

        t0 = time.perf_counter()
        vec_arr = pa.FixedSizeListArray.from_arrays(pa.array(xb.reshape(-1), type=pa.float32()), D)
        tbl = pa.Table.from_arrays(
            [
                pa.array(np.arange(ds.N, dtype=np.int64)),
                pa.array(ds.col0.astype(np.int32)),
                pa.array(ds.col1.astype(np.int32)),
                pa.array(ds.col2.astype(np.int32)),
                pa.array(ds.col3.astype(np.int32)),
                vec_arr,
            ],
            names=["id", "c0", "c1", "c2", "c3", "vec"],
        )
        con.register("src_arrow", tbl)
        con.execute(f"CREATE TABLE items AS SELECT id, c0, c1, c2, c3, vec::FLOAT[{D}] AS vec FROM src_arrow;")
        con.unregister("src_arrow")
        del tbl, vec_arr
        gc.collect()
        ingest_s = time.perf_counter() - t0

        ds_res = {"N": ds.N, "D": D, "ingest_s": round(ingest_s, 3), "workloads": {}}

        # Evaluate 48-thread DuckDB Vectorized Columnar SQL Pushdown + SIMD array_distance on W1 & W3
        # Note: DuckDB parallelizes intra-query scan across 48 threads per SQL query!
        # Evaluate on 64 representative test queries (or 256 on <=1M, 64 on 10M) and compute exact QPS & Recall@10!
        nq_eval = 64 if ds.N >= 5_000_000 else 128
        for wl_name, sel in [("W1_range", 0.05), ("W3_compound", 0.02)]:
            pf = generate_sql_workload(ds, wl_name, num_queries=n_val + n_test, selectivity=sel)
            gi, gd, _ = compute_exact_filtered_gt(ds, pf, k=10, use_cache=True)
            pt = pf.slice(n_val, n_val + nq_eval)
            gti = gi[n_val : n_val + nq_eval]
            gtd = gd[n_val : n_val + nq_eval]

            pred_i = np.full((nq_eval, 10), -1, dtype=np.int64)
            pred_d = np.full((nq_eval, 10), 1e30, dtype=np.float32)

            if wl_name == "W1_range":
                sql = f"SELECT id, array_distance(vec, $1::FLOAT[{D}]) AS dist FROM items WHERE c0 BETWEEN $2 AND $3 ORDER BY dist LIMIT 10"
                # Warmup 2 queries
                for j in range(2):
                    con.execute(sql, [xq[j].tolist(), int(pt.l0[j]), int(pt.r0[j])]).fetchall()
                t1 = time.perf_counter()
                for j in range(nq_eval):
                    rows_q = con.execute(sql, [xq[j].tolist(), int(pt.l0[j]), int(pt.r0[j])]).fetchall()
                    for r_idx, (rid, rdist) in enumerate(rows_q[:10]):
                        pred_i[j, r_idx] = int(rid)
                        pred_d[j, r_idx] = float(rdist) ** 2
                elap_ms = (time.perf_counter() - t1) * 1000.0
            else:
                sql = (
                    f"SELECT id, array_distance(vec, $1::FLOAT[{D}]) AS dist FROM items "
                    f"WHERE c1 BETWEEN $2 AND $3 AND ((1::UBIGINT << c2) & $4::UBIGINT) != 0 "
                    f"AND ($5 * c0 + $6 * c3) <= $7 ORDER BY dist LIMIT 10"
                )
                for j in range(2):
                    con.execute(sql, [xq[j].tolist(), int(pt.l1[j]), int(pt.r1[j]), int(np.uint64(pt.cat_mask[j])), int(pt.w0[j]), int(pt.w3[j]), int(pt.alg_bound[j])]).fetchall()
                t1 = time.perf_counter()
                for j in range(nq_eval):
                    rows_q = con.execute(sql, [xq[j].tolist(), int(pt.l1[j]), int(pt.r1[j]), int(np.uint64(pt.cat_mask[j])), int(pt.w0[j]), int(pt.w3[j]), int(pt.alg_bound[j])]).fetchall()
                    for r_idx, (rid, rdist) in enumerate(rows_q[:10]):
                        pred_i[j, r_idx] = int(rid)
                        pred_d[j, r_idx] = float(rdist) ** 2
                elap_ms = (time.perf_counter() - t1) * 1000.0

            rec = compute_filtered_recall_at_k(pred_i, gti, gtd, pred_d, k=10)
            qps = (nq_eval * 1000.0) / elap_ms
            ds_res["workloads"][wl_name] = {
                "DuckDB_48T_SQL": {
                    "qps": round(qps, 2),
                    "recall10": round(rec, 4),
                    "latency_ms_per_256": round(elap_ms * (256.0 / nq_eval), 2),
                }
            }
            print(f"  [DuckDB-48T Exact SQL | {ds_name} | {wl_name}] QPS={qps:.2f}, R@10={rec:.4f}", flush=True)

        # Also build DuckDB VSS HNSW index on <= 1M datasets (`CREATE INDEX ... USING HNSW`) to test SQL + HNSW planner!
        if ds.N <= 1_000_000:
            try:
                t_h = time.perf_counter()
                con.execute("CREATE INDEX idx_hnsw ON items USING HNSW (vec) WITH (metric = 'l2sq', M = 16, ef_construction = 64);")
                hnsw_build_s = time.perf_counter() - t_h
                con.execute("SET hnsw_ef_search = 256;")
                ds_res["hnsw_build_s"] = round(hnsw_build_s, 3)

                pf3 = generate_sql_workload(ds, "W3_compound", num_queries=n_val + n_test, selectivity=0.02)
                gi3, gd3, _ = compute_exact_filtered_gt(ds, pf3, k=10, use_cache=True)
                pt3 = pf3.slice(n_val, n_val + nq_eval)
                pi_h = np.full((nq_eval, 10), -1, dtype=np.int64)
                pd_h = np.full((nq_eval, 10), 1e30, dtype=np.float32)
                sql_h = (
                    f"SELECT id, array_distance(vec, $1::FLOAT[{D}]) AS dist FROM items "
                    f"WHERE c1 BETWEEN $2 AND $3 AND ((1::UBIGINT << c2) & $4::UBIGINT) != 0 "
                    f"AND ($5 * c0 + $6 * c3) <= $7 ORDER BY vec <-> $1::FLOAT[{D}] LIMIT 10"
                )
                t2 = time.perf_counter()
                for j in range(nq_eval):
                    rows_q = con.execute(sql_h, [xq[j].tolist(), int(pt3.l1[j]), int(pt3.r1[j]), int(np.uint64(pt3.cat_mask[j])), int(pt3.w0[j]), int(pt3.w3[j]), int(pt3.alg_bound[j])]).fetchall()
                    for r_idx, (rid, rdist) in enumerate(rows_q[:10]):
                        pi_h[j, r_idx] = int(rid)
                        pd_h[j, r_idx] = float(rdist) ** 2
                ms_h = (time.perf_counter() - t2) * 1000.0
                rec_h = compute_filtered_recall_at_k(pi_h, gi3[n_val : n_val + nq_eval], gd3[n_val : n_val + nq_eval], pd_h, k=10)
                ds_res["workloads"]["W3_compound"]["DuckDB_VSS_HNSW"] = {
                    "qps": round((nq_eval * 1000.0) / ms_h, 2),
                    "recall10": round(rec_h, 4),
                    "hnsw_build_s": round(hnsw_build_s, 3),
                }
                print(f"  [DuckDB-VSS HNSW | {ds_name} | W3] build={hnsw_build_s:.1f}s, QPS={(nq_eval*1000.0)/ms_h:.2f}, R@10={rec_h:.4f}", flush=True)
            except Exception as exc:
                print(f"  [DuckDB-VSS HNSW | {ds_name}] note: {exc}", flush=True)

        con.close()
        out_dbms[ds_name] = ds_res
        gc.collect()
    return out_dbms


def main():
    prism_root, _, _ = get_paths()
    res_dir = os.path.join(prism_root, "results")
    out_path = os.path.join(res_dir, "e15_peak_vram_pcie_and_dbms.json")

    res = {}
    if os.path.exists(out_path):
        try:
            res = json.load(open(out_path))
        except Exception:
            res = {}

    if "peak_vram_and_pcie" not in res:
        res["peak_vram_and_pcie"] = run_peak_vram_and_pcie_suite(res_dir)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)

    if "duckdb_sql_dbms" not in res:
        res["duckdb_sql_dbms"] = run_duckdb_sql_baseline(res_dir)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)

    print(f"\n=== [E15 COMPLETED] Saved to {out_path} ===", flush=True)


if __name__ == "__main__":
    main()
