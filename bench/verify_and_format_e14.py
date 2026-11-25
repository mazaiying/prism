#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Deterministic Verification & LaTeX Formatter for E14 (`Table 5` + Section 7.2 Prose):
1. Audits `results/e14_official_baselines.json` and `results/e14_official_vecflow.json`.
2. Asserts strict Equal-Recall (`Recall@10 >= 0.950`) across ALL PRISM (RTX 4080) and PRISM_A100 cells.
3. Computes exact per-workload (`W1`, `W2`, `W3`, `W4`, `WL1`, `WL2`) and build-time `minx-maxx` + `geomeanx`
   ratios with zero manual arithmetic, distinguishing Equal-Recall (`>=0.95`) vs recall-capped (`<0.95`) cells
   and hardware-matched (`RTX 4080` vs `48T CPU`, `A100 80GB` vs `A100 80GB`) comparisons.
4. Emits ready-to-paste LaTeX rows for `Table 5` (`\label{tab:official_fanns}`).
"""

from __future__ import annotations
import json
import math
import os
import sys
from typing import List, Tuple


def geomean(xs: List[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def fmt_qps(q: float) -> str:
    if q >= 100_000:
        return f"{q / 1000.0:.0f}K"
    if q >= 10_000:
        return f"{q / 1000.0:.1f}K"
    return f"{q / 1000.0:.2f}K" if q >= 1_000 else f"{q:.0f}"


def fmt_qr(cell: dict, force_rec: bool = False) -> str:
    if not cell or "qps" not in cell:
        return "N/A"
    q = fmt_qps(cell["qps"])
    r = cell["recall10"]
    if r >= 0.950 and not force_rec:
        return f"{q}/.{int(round(r * 100)):02d}" if r < 0.995 else f"{q}/1.0"
    return f"{q}(.{int(round(r * 100)):02d})"


def summarize_ratios(name: str, pairs: List[Tuple[str, float, float, float]]):
    """pairs: `(label, prism_val, base_val, base_recall)` where ratio = prism_val / base_val."""
    if not pairs:
        print(f"  {name}: <empty>")
        return
    ratios = [p / b for _, p, b, _ in pairs]
    r95 = [(lbl, p / b, br) for lbl, p, b, br in pairs if br >= 0.950]
    rcap = [(lbl, p / b, br) for lbl, p, b, br in pairs if br < 0.950]
    print(f"  {name} [ALL n={len(pairs)}]: min={min(ratios):.2f}x, max={max(ratios):.2f}x, geomean={geomean(ratios):.2f}x")
    for lbl, p, b, br in pairs:
        print(f"    - {lbl}: {p:.1f} vs {b:.1f} -> {p/b:.2f}x (base_R={br:.4f})")
    if r95:
        vals = [x[1] for x in r95]
        print(f"    -> [R>=0.95 subset n={len(r95)}]: min={min(vals):.2f}x, max={max(vals):.2f}x, geomean={geomean(vals):.2f}x")
    if rcap:
        vals = [x[1] for x in rcap]
        print(f"    -> [R<0.95 capped subset n={len(rcap)}]: min={min(vals):.2f}x, max={max(vals):.2f}x, br_range=[{min(x[2] for x in rcap):.3f}, {max(x[2] for x in rcap):.3f}]")


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    b_path = os.path.join(root, "results", "e14_official_baselines.json")
    v_path = os.path.join(root, "results", "e14_official_vecflow.json")

    b_data = json.load(open(b_path))["datasets"]
    v_data = json.load(open(v_path))["datasets"]

    print("=== 1. STRICT EQUAL-RECALL AUDIT (PRISM & PRISM_A100 >= 0.950) ===")
    violations = []
    for tag in ["PRISM", "PRISM_A100"]:
        recs = []
        for ds, d in b_data.items():
            for wl, wd in d.get("workloads", {}).items():
                if tag in wd:
                    r = wd[tag]["recall10"]
                    recs.append((ds, wl, r))
                    if r < 0.950:
                        violations.append((tag, ds, wl, r))
        if recs:
            r_vals = [x[2] for x in recs]
            print(f"  {tag}: {len(recs)} cells, min_R@10={min(r_vals):.4f}, median={sorted(r_vals)[len(r_vals)//2]:.4f}, max={max(r_vals):.4f}")
    if violations:
        print("  [VIOLATIONS FOUND]:", violations)
        sys.exit(1)
    else:
        print("  [PASS] 100% of PRISM & PRISM_A100 cells satisfy Recall@10 >= 0.950!")

    print("\n=== 2. EXACT PER-WORKLOAD SPEEDUP RATIOS ===")
    ds_order = [ds for ds in ["modbench", "sift1m", "gist1m", "convlog1m", "deep10m", "sift10m", "t2i10m"] if ds in b_data]

    for wl in ["W1_range", "W2_correlated", "W3_compound", "W4_mixed", "WL1_single_label", "WL2_two_label_and"]:
        ac_pairs = []
        for ds in ds_order:
            wd = b_data[ds]["workloads"].get(wl, {})
            if "PRISM" in wd and "Official_ACORN" in wd and "qps" in wd["Official_ACORN"]:
                ac_pairs.append((ds, wd["PRISM"]["qps"], wd["Official_ACORN"]["qps"], wd["Official_ACORN"]["recall10"]))
        summarize_ratios(f"PRISM(4080) vs Official_ACORN on {wl}", ac_pairs)

    for wl in ["W1_range", "W2_correlated"]:
        sf_pairs = []
        for ds in ds_order:
            wd = b_data[ds]["workloads"].get(wl, {})
            if "PRISM" in wd and "Official_SeRF" in wd and "qps" in wd["Official_SeRF"]:
                sf_pairs.append((ds, wd["PRISM"]["qps"], wd["Official_SeRF"]["qps"], wd["Official_SeRF"]["recall10"]))
        summarize_ratios(f"PRISM(4080) vs Official_SeRF on {wl}", sf_pairs)

    da_4080, da_a100 = [], []
    for ds in ds_order:
        wd = b_data[ds]["workloads"].get("WL1_single_label", {})
        if "Official_DiskANN" in wd and "qps" in wd["Official_DiskANN"]:
            if "PRISM" in wd:
                da_4080.append((ds, wd["PRISM"]["qps"], wd["Official_DiskANN"]["qps"], wd["Official_DiskANN"]["recall10"]))
            if "PRISM_A100" in wd:
                da_a100.append((ds, wd["PRISM_A100"]["qps"], wd["Official_DiskANN"]["qps"], wd["Official_DiskANN"]["recall10"]))
    summarize_ratios("PRISM(4080) vs Official_DiskANN(StitchedVamana) on WL1", da_4080)
    summarize_ratios("PRISM_A100 vs Official_DiskANN(StitchedVamana) on WL1", da_a100)

    print("\n=== 3. A100 SAME-GPU PARITY: PRISM_A100 vs OFFICIAL VECFLOW (A100) ===")
    for ds in ds_order:
        if ds in v_data and v_data[ds].get("status") == "OK":
            vf = v_data[ds]
            pb_a100 = b_data[ds]["build"].get("PRISM_A100", b_data[ds]["build"].get("PRISM", {}))
            p_wl1 = b_data[ds]["workloads"]["WL1_single_label"].get("PRISM_A100", {})
            p_wl2 = b_data[ds]["workloads"]["WL2_two_label_and"].get("PRISM_A100", {})
            v_wl1 = vf.get("WL1_single_label", {})
            v_wl2 = vf.get("WL2_two_label_and", {})
            vram_rat = vf["vram_gib"] / pb_a100["vram_gib"] if pb_a100 else 0.0
            b_rat = vf["build_s"] / pb_a100["build_s"] if pb_a100 else 0.0
            print(f"  {ds}: VecFlow VRAM={vf['vram_gib']:.2f}G vs PRISM={pb_a100.get('vram_gib',0):.2f}G ({vram_rat:.1f}x), "
                  f"Build={vf['build_s']:.1f}s vs {pb_a100.get('build_s',0):.1f}s ({b_rat:.1f}x) | "
                  f"WL1: VF={v_wl1.get('qps',0):.1f}(R={v_wl1.get('recall10',0):.4f}) vs PRISM_A100={p_wl1.get('qps',0):.1f}(R={p_wl1.get('recall10',0):.4f}) | "
                  f"WL2: VF={v_wl2.get('qps','ERR')}(R={v_wl2.get('recall10','ERR')}) vs PRISM_A100={p_wl2.get('qps',0):.1f}(R={p_wl2.get('recall10',0):.4f})")

    print("\n=== 4. BUILD TIME SPEEDUP RATIOS (Baseline Build / PRISM 4080 Build) ===")
    for bkey in ["Official_ACORN", "Official_ACORN_1", "Official_SeRF_W1_range", "Official_DiskANN"]:
        bp = []
        for ds in ds_order:
            bd = b_data[ds].get("build", {})
            if "PRISM" in bd and bkey in bd:
                bp.append((ds, bd[bkey]["build_s"], bd["PRISM"]["build_s"], 1.0))
        summarize_ratios(f"{bkey}_build / PRISM_build", bp)

    print("\n=== 5. GENERATED LATEX TABLE 5 ROWS ===")
    pretty = {
        "modbench": "ModBench",
        "sift1m": "SIFT1M",
        "gist1m": "GIST1M",
        "convlog1m": "ConvLog1M",
        "deep10m": "Deep10M",
        "sift10m": "SIFT10M",
    }
    for ds in ["modbench", "sift1m", "gist1m", "convlog1m", "deep10m", "sift10m"]:
        if ds not in b_data:
            continue
        d = b_data[ds]
        bd = d["build"]
        wd = d["workloads"]
        vram_p = bd["PRISM"]["vram_gib"]
        print(f"% --- {pretty[ds]} ---")
        # PRISM 4080
        p_w1 = fmt_qr(wd["W1_range"].get("PRISM"))
        p_w3 = fmt_qr(wd["W3_compound"].get("PRISM"))
        p_wl1 = fmt_qps(wd["WL1_single_label"]["PRISM"]["qps"])
        p_wl2 = fmt_qr(wd["WL2_two_label_and"]["PRISM"])
        print(f"\\multirow{{6}}{{*}}{{{pretty[ds]}}} & PRISM ({vram_p:.2f}G) & {bd['PRISM']['build_s']:.1f}~s & {p_w1} & {p_w3} & {p_wl1} / {p_wl2} \\\\")
        # PRISM A100
        if "PRISM_A100" in bd:
            pa_w1 = fmt_qr(wd["W1_range"].get("PRISM_A100"))
            pa_w3 = fmt_qr(wd["W3_compound"].get("PRISM_A100"))
            pa_wl1 = fmt_qps(wd["WL1_single_label"]["PRISM_A100"]["qps"])
            pa_wl2 = fmt_qr(wd["WL2_two_label_and"]["PRISM_A100"])
            print(f" & PRISM-A100\\textsuperscript{{*}} & {bd['PRISM_A100']['build_s']:.1f}~s & {pa_w1} & {pa_w3} & {pa_wl1} / {pa_wl2} \\\\")
        # Official ACORN
        if "Official_ACORN" in bd:
            ac_w1 = fmt_qr(wd["W1_range"].get("Official_ACORN"))
            ac_w3 = fmt_qr(wd["W3_compound"].get("Official_ACORN"))
            ac_wl1_c = wd["WL1_single_label"].get("Official_ACORN", {})
            ac_wl1 = fmt_qps(ac_wl1_c["qps"]) if ac_wl1_c.get("recall10", 0) >= 0.95 else fmt_qr(ac_wl1_c, True)
            ac_wl2 = fmt_qr(wd["WL2_two_label_and"].get("Official_ACORN"), True)
            print(f" & Off.-ACORN & {bd['Official_ACORN']['build_s']:.1f}~s & {ac_w1} & {ac_w3} & {ac_wl1} / {ac_wl2} \\\\")
        # Official SeRF
        if "Official_SeRF_W1_range" in bd:
            sf_w1 = fmt_qr(wd["W1_range"].get("Official_SeRF"))
            print(f" & Off.-SeRF & {bd['Official_SeRF_W1_range']['build_s']:.1f}~s & {sf_w1} & N/A & N/A \\\\")
        else:
            print(" & Off.-SeRF & -- & -- & N/A & N/A \\\\")
        # Official DiskANN (StitchedVamana)
        if "Official_DiskANN" in bd:
            da_c = wd["WL1_single_label"]["Official_DiskANN"]
            da_s = fmt_qr(da_c)
            print(f" & Off.-DiskANN & {bd['Official_DiskANN']['build_s']:.1f}~s & N/A & N/A & {da_s} / N/A \\\\")
        # Official VecFlow
        if ds in v_data and v_data[ds].get("status") == "OK":
            vf = v_data[ds]
            v1 = fmt_qr(vf["WL1_single_label"])
            v2_c = vf.get("WL2_two_label_and", {})
            v2 = fmt_qr(v2_c, True) if "qps" in v2_c else "Err"
            oom_tag = "\\textsuperscript{\\dag}" if vf["vram_gib"] > 15.57 else ""
            print(f" & Off.-VecFlow\\textsuperscript{{*}}{oom_tag} ({vf['vram_gib']:.1f}G) & {vf['build_s']:.1f}~s & N/A & N/A & {v1} / {v2} \\\\")
        print("\\midrule")


if __name__ == "__main__":
    main()
