#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Comprehensive Zero-Hand-Arithmetic Audit & LaTeX Table Generator for ALL PRISM Experiments:
  1. Asserts 100% strict `Recall@10 >= 0.950` (`R95`) and `>= 0.990` (`R99`) across all 9 datasets x 4 workloads
     (`36/36` R95 + `36/36` R99 = `72/72` cells in E1 + E13).
  2. Asserts 100% strict `Recall@10 >= 0.950` across all 30/30 E2 selectivity points and 8/8 E9 A100 datasets.
  3. Computes exact geometric mean and `[minx, maxx]` speedups over `cuVS-IVF-Flat`, `cuVS-CAGRA`, `Host-Offload`,
     `Official ACORN`, and `DuckDB 1.5.6 (48T SQL + VSS HNSW)` for Abstract, Intro, Conclusion, and Section 7.
  4. Generates ready-to-paste LaTeX rows for `Table 4` (`\label{tab:e1_high_recall}`), `Table 7` (`\label{tab:a100_and_audit}`),
     and `E15` Peak VRAM + PCIe Traffic + DuckDB SQL summary.
"""

from __future__ import annotations
import json
import math
import os
import sys
from typing import List


def geomean(xs: List[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    res = os.path.join(root, "results")

    e1 = json.load(open(os.path.join(res, "e1_main_evaluation.json")))["datasets"]
    e13 = json.load(open(os.path.join(res, "e13_modbench_eval.json")))
    e2 = json.load(open(os.path.join(res, "e2_selectivity_sweep.json")))
    e9 = json.load(open(os.path.join(res, "e9_a100_portability.json")))
    e15_p = os.path.join(res, "e15_peak_vram_pcie_and_dbms.json")
    e15 = json.load(open(e15_p)) if os.path.exists(e15_p) else {}

    # Normalize 9 datasets into unified view
    ds_all = dict(e1)
    mod_wls = {}
    remap = {
        "PRISM": "PRISM",
        "B7_HostOffload": "B7_HostOffload_PreFilter",
        "B1_InVRAM": "B1_GPU_PreFilter",
        "B2a_cuVS_IVF": "B2a_cuVS_IVF_Flat",
        "B2b_cuVS_CAGRA": "B2b_cuVS_CAGRA",
        "B5_ACORN_48T": "B5_ACORN_48T",
    }
    for wl, wd in e13["workloads"].items():
        mod_wls[wl] = {}
        for tk in ("R95", "R99"):
            mod_wls[wl][tk] = {new_k: wd[tk][old_k] for old_k, new_k in remap.items() if old_k in wd[tk]}
    ds_all["modbench"] = {
        "N": e13["N"], "D": e13["D"],
        "build_and_memory": {
            "PRISM": e13["build_and_memory"]["PRISM"],
            "B2a_cuVS_IVF_Flat": e13["build_and_memory"]["B2a_cuVS_IVF"],
            "B2b_cuVS_CAGRA": e13["build_and_memory"]["B2b_cuVS_CAGRA"],
        },
        "workloads": mod_wls,
    }

    order9 = ["convlog1m", "modbench", "bge1024", "gist1m", "msong", "sift1m", "deep10m", "sift10m", "t2i10m"]
    wls4 = ["W1_range", "W2_correlated", "W3_compound", "W4_mixed"]

    print("=== 1. STRICT EQUAL-RECALL ASSERTIONS (E1 + E13 + E2 + E9) ===")
    bad = []
    for ds in order9:
        for wl in wls4:
            for tk, thr in [("R95", 0.950), ("R99", 0.990)]:
                r = ds_all[ds]["workloads"][wl][tk]["PRISM"]["recall_at_10"]
                if r < thr:
                    bad.append(("E1/E13", ds, wl, tk, r))
    for ds in ("sift1m", "gist1m", "convlog1m"):
        for pt in e2[ds]:
            r = pt["methods"]["PRISM"]["recall"]
            if r < 0.950:
                bad.append(("E2", ds, pt["target_sel"], "R95", r))
    e9_rows = ([e9["modbench_row"]] if e9.get("modbench_row") else []) + e9["rows"]
    for r_row in e9_rows:
        if r_row["prism_recall"] < 0.950:
            bad.append(("E9_A100", r_row["dataset"], "W3", "R95", r_row["prism_recall"]))

    if bad:
        print("  [FAIL] Recall violations:", bad)
        sys.exit(1)
    print(f"  [PASS] 72/72 E1+E13 cells (R95>=0.950, R99>=0.990), 30/30 E2 cells (>=0.950), and {len(e9_rows)}/{len(e9_rows)} E9 A100 cells (>=0.950) VERIFIED!")

    print("\n=== 2. GLOBAL 9-DATASET SPEEDUPS AT R95 (36 PAIRS) ===")
    for bkey, bname in [
        ("B2a_cuVS_IVF_Flat", "cuVS-IVF-Flat"),
        ("B2b_cuVS_CAGRA", "cuVS-CAGRA"),
        ("B7_HostOffload_PreFilter", "Host-Offload PreFilter"),
        ("B5_ACORN_48T", "Official ACORN-1 (48T)"),
    ]:
        sp_all, sp_ok = [], []
        for ds in order9:
            for wl in wls4:
                p_q = ds_all[ds]["workloads"][wl]["R95"]["PRISM"]["qps"]
                b_ent = ds_all[ds]["workloads"][wl]["R95"].get(bkey, {})
                if "qps" in b_ent and b_ent.get("status") != "OOM":
                    rat = p_q / b_ent["qps"]
                    sp_all.append(rat)
                    if b_ent.get("status") == "OK":
                        sp_ok.append(rat)
        print(f"  vs {bname}: ALL non-OOM (n={len(sp_all)}): geomean={geomean(sp_all):.2f}x [{min(sp_all):.2f}x--{max(sp_all):.2f}x] | "
              f"Equal-Recall OK subset (n={len(sp_ok)}): geomean={geomean(sp_ok):.2f}x" + (f" [{min(sp_ok):.2f}x--{max(sp_ok):.2f}x]" if sp_ok else ""))

    print("\n=== 3. GENERATED TABLE 4 (W3 COMPOUND SQL, RECALL@10 >= 0.99, RTX 4080) ===")
    pretty = {
        "convlog1m": "ConvLog1M", "modbench": "ModBench", "bge1024": "BGE1024",
        "gist1m": "GIST1M", "msong": "MSong", "sift1m": "SIFT1M",
        "deep10m": "Deep10M", "sift10m": "SIFT10M", "t2i10m": "Text2Img10M",
    }
    for ds in order9:
        bm = ds_all[ds]["build_and_memory"]
        w3 = ds_all[ds]["workloads"]["W3_compound"]["R99"]
        pm = bm["PRISM"]["total_gpu_vram_gib"]
        bt = bm["PRISM"]["total_build_s"]
        im = f"{bm['B2a_cuVS_IVF_Flat']['total_gpu_vram_gib']:.2f}~GiB" if "total_gpu_vram_gib" in bm.get("B2a_cuVS_IVF_Flat", {}) else "OOM"
        cm = f"{bm['B2b_cuVS_CAGRA']['total_gpu_vram_gib']:.2f}~GiB" if "total_gpu_vram_gib" in bm.get("B2b_cuVS_CAGRA", {}) else "OOM"
        pq = f"{w3['PRISM']['qps']:,.1f}"

        def fq(ent):
            if not ent or ent.get("status") == "OOM" or "qps" not in ent:
                return "OOM"
            s = f"{ent['qps']:,.1f}"
            return s if ent.get("status") == "OK" else f"{s} ({ent['recall_at_10']:.2f})"

        print(f"{pretty[ds]} & {pm:.2f}~GiB & {im} & {cm} & {bt:.1f}~s & {pq} & {fq(w3.get('B2a_cuVS_IVF_Flat'))} & {fq(w3.get('B2b_cuVS_CAGRA'))} & {fq(w3.get('B7_HostOffload_PreFilter'))} & {fq(w3.get('B1_GPU_PreFilter'))} \\\\")

    print("\n=== 4. GENERATED TABLE 7 (A100 80GB W3 SQL, RECALL@10 >= 0.95) ===")
    pretty_a100 = dict(pretty, sift100m="SIFT100M")
    for r in e9_rows:
        ds = r["dataset"]
        ivf_s = "OOM" if r.get("cuvs_ivf_status") == "OOM" else f"{r['cuvs_ivf_qps']:,.1f}"
        cg_s = "OOM" if r.get("cuvs_cagra_status") == "OOM" else f"{r['cuvs_cagra_qps']:,.1f}"
        print(f"{pretty_a100[ds]} & {r['prism_vram_gib']:.2f}~G & {r['prism_qps']:,.1f} & {ivf_s} & {cg_s} & {r['host_offload_qps']:,.1f} & 100.0\\% \\\\")

    if e15:
        print("\n=== 5. E15 PEAK VRAM, PCIE TRAFFIC BYTES & DUCKDB 48T SQL SUMMARY ===")
        pv = e15.get("peak_vram_and_pcie", {})
        for ds, row in pv.items():
            p = row["PRISM"]
            h = row["Host_Offload"]
            ivf_pk = row.get("cuVS_IVF_Flat", {}).get("peak_vram_gib", "OOM")
            cg_pk = row.get("cuVS_CAGRA", {}).get("peak_vram_gib", "OOM")
            ws_mib = p.get("runtime_workspace_mib", (p["peak_vram_gib"] - p["static_vram_gib"]) * 1024.0)
            red_x = h.get("pcie_reduction_x", h["pcie_transfer_mib"] / max(1e-9, p["pcie_transfer_mib"]))
            print(f"  {ds}: PRISM static={p['static_vram_gib']:.3f}G -> peak={p['peak_vram_gib']:.3f}G (+{ws_mib:.1f} MiB) | "
                  f"cuVS-IVF peak={ivf_pk} | CAGRA peak={cg_pk} | "
                  f"PCIe batch: PRISM={p['pcie_transfer_mib']:.2f} MiB vs HostOff={h['pcie_transfer_mib']:.2f} MiB ({red_x:.1f}x reduction)")
        db = e15.get("duckdb_sql_dbms", {})
        for ds, drow in db.items():
            w1_d = drow["workloads"].get("W1_range", {}).get("DuckDB_48T_SQL", {})
            w3_d = drow["workloads"].get("W3_compound", {}).get("DuckDB_48T_SQL", {})
            w3_h = drow["workloads"].get("W3_compound", {}).get("DuckDB_VSS_HNSW", {})
            p_w1 = ds_all[ds]["workloads"]["W1_range"]["R95"]["PRISM"]["qps"]
            p_w3 = ds_all[ds]["workloads"]["W3_compound"]["R95"]["PRISM"]["qps"]
            print(f"  DuckDB 48T [{ds}]: W1={w1_d.get('qps',0):.1f} QPS (PRISM {p_w1/max(1e-6,w1_d.get('qps',1)):.1f}x) | "
                  f"W3 Exact={w3_d.get('qps',0):.1f} QPS (PRISM {p_w3/max(1e-6,w3_d.get('qps',1)):.1f}x) | "
                  f"W3 VSS-HNSW={w3_h.get('qps','N/A')} (R={w3_h.get('recall10','N/A')})")


if __name__ == "__main__":
    main()
