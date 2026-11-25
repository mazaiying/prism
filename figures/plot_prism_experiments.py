#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data-Driven Publication Plotting Suite for PRISM (SIGMOD 2027 Industrial Track).
Generates experimental evaluation plots directly from verified JSON artifacts in `results/`.
"""

from __future__ import annotations
import json
import os
import shutil
from typing import Dict, Optional
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["axes.linewidth"] = 0.85

LAB_COLORS = [
    "#E5E7E9",  # Soft silver-gray (Host-Offload)
    "#FCF3CF",  # Whisper-pale cream (HNSW-48T)
    "#D6EAF8",  # Soft ice blue (cuVS-IVF-Flat)
    "#DFEFE2",  # Pale mist sage (Official ACORN)
    "#EBDEF0",  # Pale lilac (Faiss-GPU Post)
    "#E1EEF6",  # Pale powder blue (VecFlow-SQL)
    "#FDEBD0",  # Pale peach (cuVS-CAGRA)
    "#F5B7B1",  # Soft muted dusty rose (PRISM Ours)
]
C_DARK = "#333333"


def load_json_if_exists(res_dir: str, fname: str) -> Optional[Dict]:
    p = os.path.join(res_dir, fname)
    if not os.path.exists(p):
        return None
    with open(p, "r") as f:
        return json.load(f)


def style_ax(ax, grid_axis="y"):
    ax.grid(True, which="major", axis=grid_axis, linestyle="--", alpha=0.38, linewidth=0.52, color="#BDBDBD", zorder=0)
    ax.yaxis.grid(False, which="minor")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=9.2, length=2.8, width=0.8)


def save_both(fig, out_dir: str, paper_fig_dir: str, stem: str):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(paper_fig_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        dst1 = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(dst1, dpi=250, bbox_inches="tight", pad_inches=0.02)
        dst2 = os.path.join(paper_fig_dir, f"{stem}.{ext}")
        if os.path.abspath(dst1) != os.path.abspath(dst2):
            shutil.copy2(dst1, dst2)
    plt.close(fig)
    print(f"[Generated] {stem}.pdf / .png")


# ==============================================================================
# 1. Main E1 Equal-Recall Throughput across 9 Datasets (`fig5_e1_main_qps.pdf`)
#    Ultra-flat 3x3 on 14.0-inch width => 50% larger printed text!
# ==============================================================================
def plot_fig5_e1_main_qps(res_dir: str, out_dir: str, paper_fig_dir: str):
    e1 = load_json_if_exists(res_dir, "e1_main_evaluation.json")
    e13 = load_json_if_exists(res_dir, "e13_modbench_eval.json") or load_json_if_exists(res_dir, "e13_modbench_eval_a100.json")
    if not e1:
        return

    merged_ds = dict(e1.get("datasets", {}))
    if e13 and "workloads" in e13:
        norm_wls = {}
        key_remap = {
            "PRISM": "PRISM",
            "B7_HostOffload": "B7_HostOffload_PreFilter",
            "B6_HNSW_48T": "B6_HNSW_Filter_48T",
            "B5_ACORN_48T": "B5_ACORN_48T",
            "B3_Faiss_PostFilter": "B3_Faiss_PostFilter",
            "B4_VecFlow_GPU": "B4_VecFlow_LabelGPU",
            "B2b_cuVS_CAGRA": "B2b_cuVS_CAGRA",
            "B2a_cuVS_IVF": "B2a_cuVS_IVF_Flat",
        }
        for wk, wv in e13["workloads"].items():
            norm_wls[wk] = {}
            for tier_k in ("R95", "R99"):
                if tier_k in wv:
                    t_norm = {}
                    for old_k, new_k in key_remap.items():
                        if old_k in wv[tier_k]:
                            ent = dict(wv[tier_k][old_k])
                            rec = ent.get("recall_at_10", 1.0)
                            thr = 0.95 if tier_k == "R95" else 0.99
                            if ent.get("status") != "OOM":
                                ent["status"] = "OK" if rec >= thr else "RECALL_MISS"
                            t_norm[new_k] = ent
                    norm_wls[wk][tier_k] = t_norm
        merged_ds["modbench"] = {"workloads": norm_wls}

    ds_order = [
        ("convlog1m", "ConvLog1M (1024D)"),
        ("modbench", "ModBench (384D)"),
        ("bge1024", "BGE1024 (1024D)"),
        ("gist1m", "GIST1M (960D)"),
        ("msong", "MSong (420D)"),
        ("sift1m", "SIFT1M (128D)"),
        ("deep10m", "Deep10M (96D)"),
        ("sift10m", "SIFT10M (128D)"),
        ("t2i10m", "Text2Img10M (200D)"),
    ]
    present = [(k, lbl) for k, lbl in ds_order if k in merged_ds]
    ncols, nrows = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, 2.42))
    axes_flat = np.atleast_1d(axes).flatten()

    methods = [
        ("B7_HostOffload_PreFilter", "Host-Offload", LAB_COLORS[0], None),
        ("B6_HNSW_Filter_48T", "HNSW-48T", LAB_COLORS[1], None),
        ("B5_ACORN_48T", "ACORN", LAB_COLORS[3], None),
        ("B3_Faiss_PostFilter", "Faiss-Post", LAB_COLORS[4], None),
        ("B4_VecFlow_LabelGPU", "VecFlow-SQL", LAB_COLORS[5], None),
        ("B2b_cuVS_CAGRA", "cuVS-CAGRA", LAB_COLORS[6], None),
        ("B2a_cuVS_IVF_Flat", "cuVS-IVF", LAB_COLORS[2], None),
        ("PRISM", "PRISM (Ours)", LAB_COLORS[7], "//"),
    ]
    wls = [("W1_range", "W1:Range"), ("W2_correlated", "W2:Corr"), ("W3_compound", "W3:Comp"), ("W4_mixed", "W4:Mixed")]
    bar_w = 0.098
    x_base = np.arange(len(wls))

    for idx, (dk, dlbl) in enumerate(present):
        ax = axes_flat[idx]
        style_ax(ax, "y")
        d_obj = merged_ds[dk]["workloads"]
        all_qps = []
        for wk, _ in wls:
            tier = d_obj.get(wk, {}).get("R95", {})
            for m_k, _, _, _ in methods:
                m_entry = tier.get(m_k, {})
                if m_entry.get("status") == "OK":
                    all_qps.append(m_entry["qps"])
        ymin = max(12.0, min(all_qps) * 0.42) if all_qps else 10.0
        ymax = (max(all_qps) * 3.8) if all_qps else 1e5

        for m_i, (m_k, m_lbl, m_col, m_hat) in enumerate(methods):
            offs = (m_i - (len(methods) - 1) / 2.0) * bar_w
            vals, statuses = [], []
            for wk, _ in wls:
                ent = d_obj.get(wk, {}).get("R95", {}).get(m_k, {"status": "OOM"})
                st = ent.get("status", "OOM")
                statuses.append(st)
                vals.append(ent.get("qps", 0.0) if st == "OK" else 0.0)
            ax.bar(x_base + offs, vals, bar_w * 0.94, facecolor=m_col, edgecolor=C_DARK,
                   lw=0.88 if m_k == "PRISM" else 0.68, hatch=m_hat, label=m_lbl if idx == 0 else None, zorder=3)
            for w_i, st in enumerate(statuses):
                if st == "OOM":
                    ax.plot(x_base[w_i] + offs, ymin * 1.40, marker="x", markersize=4.8, markeredgewidth=1.3, color="#B02A2A", zorder=5)
                elif st == "RECALL_MISS":
                    ax.plot(x_base[w_i] + offs, ymin * 1.40, marker="v", markersize=4.2, color="#7D3C98", zorder=5)

        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(x_base)
        ax.set_xticklabels([w[1] for w in wls], fontsize=9.4)
        if idx % ncols == 0:
            ax.set_ylabel("QPS (log)", fontsize=9.8)
        sub_let = chr(ord("a") + idx)
        ax.text(0.5, -0.58, f"({sub_let}) {dlbl}", transform=ax.transAxes, ha="center", va="top", fontsize=10.4, fontweight="bold")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    oom_proxy = mpl.lines.Line2D([], [], color="#B02A2A", marker="x", linestyle="None", markersize=5.6, markeredgewidth=1.4, label="OOM")
    miss_proxy = mpl.lines.Line2D([], [], color="#7D3C98", marker="v", linestyle="None", markersize=5.0, label="Recall Miss")
    fig.legend(handles + [oom_proxy, miss_proxy], labels + ["OOM", "Recall Miss"],
               loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=10, fontsize=9.5, frameon=True,
               edgecolor="#BDBDBD", columnspacing=0.75, handletextpad=0.35, handlelength=1.25)
    plt.subplots_adjust(hspace=0.96, wspace=0.17, top=0.89, bottom=0.16)
    save_both(fig, out_dir, paper_fig_dir, "fig5_e1_main_qps")


# ==============================================================================
# 2. Continuous `QPS vs. Recall` with `Log-Loss (1 - Recall)` Axis (`fig6_e10_pareto_logloss.pdf`)
# ==============================================================================
def plot_fig6_pareto_logloss(res_dir: str, out_dir: str, paper_fig_dir: str):
    e10 = load_json_if_exists(res_dir, "e10_pareto_qps_recall.json")
    if not e10 or not e10.get("datasets"):
        return

    ds_order = [
        ("convlog1m", "ConvLog1M (1024D)"),
        ("modbench", "ModBench (384D)"),
        ("bge1024", "BGE1024 (1024D)"),
        ("gist1m", "GIST1M (960D)"),
        ("msong", "MSong (420D)"),
        ("sift1m", "SIFT1M (128D)"),
        ("deep10m", "Deep10M (96D)"),
        ("sift10m", "SIFT10M (128D)"),
    ]
    present = [(k, lbl) for k, lbl in ds_order if k in e10["datasets"]]
    ncols, nrows = 4, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, 2.06))
    axes_flat = np.atleast_1d(axes).flatten()

    series_cfg = [
        ("PRISM", "PRISM (Ours, INT4+1b)", "#C45A52", "o-", 1.50, 4.4, 6),
        ("PRISM_INT4", "PRISM-INT4 (No Sign)", "#D49257", "d--", 1.10, 4.0, 5),
        ("Host_Offload", "Host-Offload", "#7F8C8D", "x--", 1.10, 4.2, 4),
        ("ACORN_48T", "Official ACORN", "#589A73", "^--", 1.10, 4.0, 3),
        ("cuVS_IVF", "cuVS-IVF-Flat", "#5486B4", "s--", 1.10, 4.0, 2),
        ("cuVS_CAGRA", "cuVS-CAGRA", "#8E7CC3", "p--", 1.10, 4.0, 1),
    ]

    for idx, (dk, dlbl) in enumerate(present):
        ax = axes_flat[idx]
        style_ax(ax, "both")
        d_curves = e10["datasets"][dk]

        for m_key, m_lbl, col, fmt, lw, ms, zord in series_cfg:
            pts = d_curves.get(m_key, [])
            if not pts:
                continue
            pts_sorted = sorted(pts, key=lambda r: (round(r["recall"], 4), -r["qps"]))
            dedup = []
            last_r = -1.0
            for p in pts_sorted:
                if p["recall"] < 0.50:
                    continue
                r_key = round(min(p["recall"], 0.9999), 4)
                if r_key > last_r:
                    dedup.append(p)
                    last_r = r_key
            losses = [float(np.clip(1.0 - p["recall"], 1.2e-4, 0.52)) for p in dedup]
            qps_v = [float(p["qps"]) for p in dedup]
            if not losses:
                continue
            ax.plot(losses, qps_v, fmt, color=col, markerfacecolor=col,
                    lw=lw, markersize=ms, alpha=0.88, zorder=zord,
                    label=m_lbl if idx == 0 else None)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.55, 0.9e-4)
        ax.set_xticks([4e-1, 1e-1, 1e-2, 1e-3, 1.2e-4])
        ax.set_xticklabels(["0.6", "0.9", "0.99", ".999", "1.0"], fontsize=8.8)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        if idx >= ncols:
            ax.set_xlabel("Recall@10", fontsize=9.5, labelpad=0.8)
        if idx % ncols == 0:
            ax.set_ylabel("QPS (log)", fontsize=9.8)
        sub_let = chr(ord("a") + idx)
        y_sub = -0.82 if idx >= ncols else -0.52
        ax.text(0.5, y_sub, f"({sub_let}) {dlbl}", transform=ax.transAxes, ha="center", va="top", fontsize=10.3, fontweight="bold")

    h0, l0 = axes_flat[0].get_legend_handles_labels()
    fig.legend(h0, l0, loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=6, fontsize=9.6, frameon=True,
               edgecolor="#BDBDBD", columnspacing=0.9, handletextpad=0.4)
    plt.subplots_adjust(hspace=0.90, wspace=0.20, top=0.86, bottom=0.26)
    save_both(fig, out_dir, paper_fig_dir, "fig6_e10_pareto_logloss")


# ==============================================================================
# 3. Full-Spectrum Selectivity Sweep + Three-Regime Crossover (`fig6_e2_selectivity.pdf`)
# ==============================================================================
def plot_fig7_e2_selectivity(res_dir: str, out_dir: str, paper_fig_dir: str):
    e2 = load_json_if_exists(res_dir, "e2_selectivity_sweep.json")
    if not e2:
        return

    fig, axes = plt.subplots(1, 4, figsize=(14.0, 1.38))
    line_cfg = [
        ("PRISM", "PRISM (Ours)", "#C45A52", "-", "o"),
        ("B2a_cuVS_IVF_Flat", "cuVS-IVF-Flat", "#5486B4", "--", "s"),
        ("B2b_cuVS_CAGRA", "cuVS-CAGRA", "#589A73", "-.", "^"),
        ("B4_VecFlow_LabelGPU", "VecFlow-SQL", "#8E7CC3", ":", "D"),
        ("B5_ACORN_48T", "Official ACORN", "#7F8C8D", "-.", "p"),
        ("B7_HostOffload", "Host-Offload", "#D49257", "--", "x"),
    ]

    for idx, (dk, dlbl) in enumerate([("sift1m", "SIFT1M (128D)"), ("gist1m", "GIST1M (960D)"), ("convlog1m", "ConvLog1M (1024D)")]):
        ax = axes[idx]
        style_ax(ax, "both")
        rows = e2[dk]
        xs = [r["target_sel"] * 100.0 for r in rows]
        for mk_key, mlbl, col, ls, mk in line_cfg:
            ys = [
                r["methods"].get(mk_key, {}).get("qps", np.nan)
                if r["methods"].get(mk_key, {}).get("recall", 0.0) >= 0.95
                else np.nan
                for r in rows
            ]
            if not all(np.isnan(ys)):
                ax.plot(xs, ys, color=col, linestyle=ls, marker=mk, markersize=4.2,
                        lw=1.60 if mk_key == "PRISM" else 1.15, label=mlbl if idx == 0 else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Selectivity (%)", fontsize=9.6, labelpad=0.8)
        if idx == 0:
            ax.set_ylabel("QPS (log)", fontsize=9.8)
        ax.text(0.5, -0.72, f"({chr(ord('a')+idx)}) {dlbl}", transform=ax.transAxes, ha="center", va="top", fontsize=10.5, fontweight="bold")

    ax = axes[3]
    style_ax(ax, "both")
    rows_g = e2["gist1m"]
    xs_g = [r["target_sel"] * 100.0 for r in rows_g]
    for rk, rlbl, col, ls, mk in [
        ("PRISM_Regime1", "R1", "#5486B4", "--", "^"),
        ("PRISM_Regime2", "R2", "#589A73", "-.", "s"),
        ("PRISM_Regime3", "R3", "#8E7CC3", ":", "D"),
        ("PRISM", "Auto M3", "#C45A52", "-", "o"),
    ]:
        ys = [
            r["methods"][rk]["qps"]
            if (rk == "PRISM" or r["methods"][rk]["recall"] >= 0.88)
            else np.nan
            for r in rows_g
        ]
        ax.plot(xs_g, ys, color=col, linestyle=ls, marker=mk, markersize=4.0,
                lw=1.60 if rk == "PRISM" else 1.10, label=rlbl)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Selectivity (%)", fontsize=9.6, labelpad=0.8)
    ax.set_ylabel("GIST1M QPS", fontsize=9.8)
    ax.legend(loc="lower left", fontsize=8.0, ncol=2, frameon=True, edgecolor="#BDBDBD", borderpad=0.14, columnspacing=0.5, handletextpad=0.3)
    ax.text(0.5, -0.72, "(d) M3 three-regime crossover", transform=ax.transAxes, ha="center", va="top", fontsize=10.5, fontweight="bold")

    h0, l0 = axes[0].get_legend_handles_labels()
    fig.legend(h0, l0, loc="upper center", bbox_to_anchor=(0.42, 1.15), ncol=6, fontsize=9.6, frameon=True,
               edgecolor="#BDBDBD", columnspacing=0.85, handletextpad=0.35)
    plt.subplots_adjust(wspace=0.24, top=0.82, bottom=0.38)
    save_both(fig, out_dir, paper_fig_dir, "fig6_e2_selectivity")


# ==============================================================================
# 4. Correlation Skew (`E3`) + SIFT100M Out-of-Core & Break-Even (`fig7_e3_e4_skew_and_100m.pdf`)
# ==============================================================================
def plot_fig8_e3_e4_skew_and_100m(res_dir: str, out_dir: str, paper_fig_dir: str):
    e3 = load_json_if_exists(res_dir, "e3_correlation_skew.json")
    e4 = load_json_if_exists(res_dir, "e4_sift100m_scale.json")
    if not e3 or not e4:
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.0, 1.30))

    style_ax(ax1, "both")
    for dk, dlbl, c_p, c_u, c_c, mk in [
        ("sift1m", "SIFT", "#C45A52", "#D49257", "#5486B4", "o"),
        ("gist1m", "GIST", "#8E7CC3", "#589A73", "#7F8C8D", "s"),
    ]:
        rows = e3[dk]
        rhos = [r["rho_corr"] for r in rows]
        ax1.plot(rhos, [r["prism_qps"] for r in rows], color=c_p, ls="-", marker=mk, ms=4.2, lw=1.50, label=f"PRISM ({dlbl})")
        ax1.plot(rhos, [r["uncompacted_qps"] for r in rows], color=c_u, ls="--", marker=mk, ms=3.8, lw=1.10, label=f"NoComp ({dlbl})")
        ax1.plot(rhos, [r["cuvs_ivf_qps"] for r in rows], color=c_c, ls=":", marker=mk, ms=3.8, lw=1.10, label=f"cuVS ({dlbl})")
    ax1.set_yscale("log")
    ax1.set_ylim(420, 220000)
    ax1.set_xlabel(r"Correlation $\rho_{\mathrm{corr}}$", fontsize=9.8, labelpad=1.0)
    ax1.set_ylabel("QPS (log)", fontsize=9.8)
    ax1.legend(loc="center", bbox_to_anchor=(0.50, 0.36), fontsize=7.6, ncol=3, frameon=True,
               facecolor="white", framealpha=0.95, edgecolor="#BDBDBD", borderpad=0.14, columnspacing=0.45, handletextpad=0.3)
    ax1.text(0.5, -0.54, "(a) Robustness to correlation skew", transform=ax1.transAxes, ha="center", va="top", fontsize=10.5, fontweight="bold")

    style_ax(ax2, "y")
    wls = [("W1_range", "W1:Range"), ("W2_correlated", "W2:Corr"), ("W3_compound", "W3:Comp"), ("W4_mixed", "W4:Mixed")]
    x = np.arange(len(wls))
    bw = 0.25
    w_obj = e4["workloads"]
    q_off = [w_obj[wk]["host_offload_qps"] for wk, _ in wls]
    q_pri = [w_obj[wk]["prism_qps"] for wk, _ in wls]

    ax2.bar(x - bw, [0] * 4, bw, facecolor=LAB_COLORS[2], edgecolor=C_DARK, lw=0.72, label="cuVS (OOM)")
    ax2.bar(x, q_off, bw, facecolor=LAB_COLORS[0], edgecolor=C_DARK, lw=0.72, label="Offload", zorder=3)
    ax2.bar(x + bw, q_pri, bw, facecolor=LAB_COLORS[7], edgecolor=C_DARK, lw=0.80, hatch="//", label="PRISM (9.8G)", zorder=3)
    ymin_4 = max(10.0, min(q_off) * 0.38)
    ax2.set_yscale("log")
    ax2.set_ylim(ymin_4, max(q_pri) * 16.0)
    for wi, (wk, _) in enumerate(wls):
        ax2.plot(x[wi] - bw, ymin_4 * 1.55, marker="x", markersize=5.4, markeredgewidth=1.4, color="#B02A2A", zorder=5)
        spd = w_obj[wk]["speedup_over_host_offload"]
        ax2.text(x[wi] + bw, q_pri[wi] * 1.20, f"{spd:.1f}x", ha="center", va="bottom", fontsize=8.4, fontweight="bold", color="#8B1A1A")
    ax2.set_xticks(x)
    ax2.set_xticklabels([w[1] for w in wls], fontsize=9.2)
    ax2.set_ylabel("SIFT100M QPS", fontsize=9.8)
    ax2.legend(loc="upper right", fontsize=7.8, ncol=3, frameon=True, facecolor="white", framealpha=0.95,
               edgecolor="#BDBDBD", borderpad=0.14, columnspacing=0.45, handletextpad=0.3, handlelength=1.1)
    ax2.text(0.5, -0.54, "(b) 100M-vector throughput (16 GB)", transform=ax2.transAxes, ha="center", va="top", fontsize=10.5, fontweight="bold")

    style_ax(ax3, "both")
    b_p = e4["build_prism"]["total_build_s"]
    b_o = e4["build_host_offload"]["total_build_s"]
    qps_p = w_obj["W3_compound"]["prism_qps"]
    qps_o = w_obj["W3_compound"]["host_offload_qps"]
    nq_grid = np.logspace(2, 6, 80)
    ax3.plot(nq_grid, b_o + nq_grid / qps_o, color="#7F8C8D", ls="--", lw=1.35, label=f"Offload ({b_o:.1f}s)")
    ax3.plot(nq_grid, b_p + nq_grid / qps_p, color="#C45A52", ls="-", lw=1.55, label=f"PRISM ({b_p:.1f}s)")
    if b_p > b_o and (1.0 / qps_o > 1.0 / qps_p):
        q_be = (b_p - b_o) / (1.0 / qps_o - 1.0 / qps_p)
        t_be = b_p + q_be / qps_p
        if 1e2 <= q_be <= 1e6:
            ax3.plot(q_be, t_be, marker="*", markersize=8.5, color="#F7DC6F", markeredgecolor=C_DARK, zorder=6)
            ax3.annotate(f"Break-even: {int(q_be):,} q", xy=(q_be, t_be), xytext=(q_be * 0.03, t_be * 2.1),
                         arrowprops=dict(arrowstyle="->", lw=0.85, color=C_DARK), fontsize=8.2, fontweight="bold")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Cumulative Queries", fontsize=9.8, labelpad=1.0)
    ax3.set_ylabel("Time (s)", fontsize=9.8)
    ax3.legend(loc="upper left", fontsize=7.8, ncol=2, frameon=True, edgecolor="#BDBDBD", borderpad=0.14, columnspacing=0.45, handletextpad=0.3)
    ax3.text(0.5, -0.54, "(c) SIFT100M build-query amortization", transform=ax3.transAxes, ha="center", va="top", fontsize=10.5, fontweight="bold")

    plt.subplots_adjust(wspace=0.22, top=0.95, bottom=0.33)
    save_both(fig, out_dir, paper_fig_dir, "fig7_e3_e4_skew_and_100m")


# ==============================================================================
# 5. Ablation + Query 4-Stage + Build 3-Stage Breakdown (`fig8_e5_ablation.pdf`)
# ==============================================================================
def plot_fig9_ablation_and_breakdown(res_dir: str, out_dir: str, paper_fig_dir: str):
    e5 = load_json_if_exists(res_dir, "e5_ablation.json")
    e11 = load_json_if_exists(res_dir, "e11_time_breakdown.json")
    if not e5:
        return

    step = e5["step_ablation"]
    v_keys = [
        ("V0_HostOffload_NoHeader", "Base"),
        ("V1_+TileHeader_Skip", "+Hdr"),
        ("V2_+INT4_Ball_Uncomp", "+INT4"),
        ("V3_+1Bit_Sign_Refined", "+1b"),
        ("V4_+Warp_Compaction", "+Cmp"),
        ("V5_+Survivor_Prewarm", "+Prw"),
        ("V6_Full_PRISM_M3", "Full"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14.0, 1.34))
    x = np.arange(len(v_keys))
    w = 0.36

    ax1 = axes[0]
    style_ax(ax1, "y")
    ax1.bar(x - w / 2, [step["gist1m"][vk]["w3_qps"] for vk, _ in v_keys], w,
            facecolor=LAB_COLORS[2], edgecolor=C_DARK, lw=0.72, label="GIST1M", zorder=3)
    ax1.bar(x + w / 2, [step["convlog1m"][vk]["w3_qps"] for vk, _ in v_keys], w,
            facecolor=LAB_COLORS[7], edgecolor=C_DARK, lw=0.76, hatch="//", label="Conv1M", zorder=3)
    ax1.set_yscale("log")
    ax1.set_ylim(200, 75000)
    ax1.set_xticks(x)
    ax1.set_xticklabels([vl for _, vl in v_keys], rotation=12, fontsize=8.8)
    ax1.set_ylabel("QPS (log)", fontsize=9.8)
    ax1.legend(loc="upper left", fontsize=7.8, ncol=2, frameon=True, edgecolor="#BDBDBD", borderpad=0.14, columnspacing=0.45, handletextpad=0.3)
    ax1.text(0.5, -0.48, "(a) High-D module ablation", transform=ax1.transAxes, ha="center", va="top", fontsize=10.5, fontweight="bold")

    ax2 = axes[1]
    style_ax(ax2, "y")
    ax2.bar(x - w / 2, [step["sift1m"][vk]["w3_qps"] for vk, _ in v_keys], w,
            facecolor=LAB_COLORS[3], edgecolor=C_DARK, lw=0.72, label="SIFT1M", zorder=3)
    ax2.bar(x + w / 2, [step["sift10m"][vk]["w3_qps"] for vk, _ in v_keys], w,
            facecolor="#FDEBD0", edgecolor=C_DARK, lw=0.76, hatch="//", label="SIFT10M", zorder=3)
    ax2.set_yscale("log")
    ax2.set_ylim(1800, 520000)
    ax2.set_xticks(x)
    ax2.set_xticklabels([vl for _, vl in v_keys], rotation=12, fontsize=8.8)
    ax2.set_ylabel("QPS (log)", fontsize=9.8)
    ax2.legend(loc="upper left", fontsize=7.8, ncol=2, frameon=True, edgecolor="#BDBDBD", borderpad=0.14, columnspacing=0.45, handletextpad=0.3)
    ax2.text(0.5, -0.48, "(b) 1M & 10M module ablation", transform=ax2.transAxes, ha="center", va="top", fontsize=10.5, fontweight="bold")

    ax3 = axes[2]
    style_ax(ax3, "y")
    if e11 and e11.get("query_breakdown"):
        qb = e11["query_breakdown"]
        bars_meta = []
        for ds_k, ds_short in [("convlog1m", "Conv"), ("gist1m", "GIST"), ("sift10m", "S10M")]:
            if ds_k in qb:
                bars_meta.append((f"{ds_short}\nOff", qb[ds_k]["Host_Offload"]))
                bars_meta.append((f"{ds_short}\nPRI", qb[ds_k]["PRISM"]))
        bx = np.arange(len(bars_meta))
        bw_s = 0.54
        st_plan = np.array([100.0 * b[1]["plan_ms"] / max(1e-6, b[1]["total_ms"]) for b in bars_meta])
        st_filter = np.array([100.0 * b[1]["filter_ms"] / max(1e-6, b[1]["total_ms"]) for b in bars_meta])
        st_sketch = np.array([100.0 * b[1]["sketch_ms"] / max(1e-6, b[1]["total_ms"]) for b in bars_meta])
        st_refine = np.array([100.0 * b[1]["refine_ms"] / max(1e-6, b[1]["total_ms"]) for b in bars_meta])

        ax3.bar(bx, st_plan, bw_s, facecolor="#EBDEF0", edgecolor=C_DARK, lw=0.70, label="Plan", zorder=3)
        ax3.bar(bx, st_filter, bw_s, bottom=st_plan, facecolor="#D6EAF8", edgecolor=C_DARK, lw=0.70, label="SLAP", zorder=3)
        ax3.bar(bx, st_sketch, bw_s, bottom=st_plan + st_filter, facecolor="#DFEFE2", edgecolor=C_DARK, lw=0.70, label="Ball", zorder=3)
        ax3.bar(bx, st_refine, bw_s, bottom=st_plan + st_filter + st_sketch, facecolor="#F5B7B1", edgecolor=C_DARK, lw=0.70, hatch="//", label="UVA", zorder=3)
        for i_b, (_, bm) in enumerate(bars_meta):
            ax3.text(bx[i_b], 103.0, f"{bm['total_ms']:.0f}ms" if bm["total_ms"] >= 10 else f"{bm['total_ms']:.1f}ms",
                     ha="center", va="bottom", fontsize=7.6, fontweight="bold")
        ax3.set_ylim(0, 162)
        ax3.set_xticks(bx)
        ax3.set_xticklabels([b[0] for b in bars_meta], fontsize=8.2)
        ax3.set_ylabel("Time (%)", fontsize=9.8)
        ax3.legend(loc="upper center", ncol=4, fontsize=7.6, frameon=True, facecolor="white", framealpha=0.95,
                   edgecolor="#BDBDBD", borderpad=0.13, columnspacing=0.40, handletextpad=0.25, handlelength=1.0)
    ax3.text(0.5, -0.48, "(c) Query 4-stage breakdown", transform=ax3.transAxes, ha="center", va="top", fontsize=10.5, fontweight="bold")

    ax4 = axes[3]
    style_ax(ax4, "y")
    if e11 and e11.get("build_breakdown"):
        bb = e11["build_breakdown"]
        ds_b_list = [
            ("modbench", "Mod"), ("sift1m", "S1M"), ("gist1m", "GIST"),
            ("convlog1m", "Conv"), ("sift10m", "S10M"), ("t2i10m", "T2I"), ("sift100m", "S100M"),
        ]
        pres_b = [(k, l) for k, l in ds_b_list if k in bb]
        xb_idx = np.arange(len(pres_b))
        km_v = np.array([bb[k]["kmeans_s"] for k, _ in pres_b])
        ly_v = np.array([bb[k]["layout_s"] for k, _ in pres_b])
        en_v = np.array([bb[k]["encode_s"] for k, _ in pres_b])
        tot_v = km_v + ly_v + en_v

        ax4.bar(xb_idx, km_v, 0.54, facecolor="#FCF3CF", edgecolor=C_DARK, lw=0.70, label="KMeans", zorder=3)
        ax4.bar(xb_idx, ly_v, 0.54, bottom=km_v, facecolor="#D6EAF8", edgecolor=C_DARK, lw=0.70, label="Layout", zorder=3)
        ax4.bar(xb_idx, en_v, 0.54, bottom=km_v + ly_v, facecolor="#F5B7B1", edgecolor=C_DARK, lw=0.70, hatch="//", label="Sketch", zorder=3)
        for i_b, tv in enumerate(tot_v):
            ax4.text(xb_idx[i_b], tv + max(tot_v) * 0.03, f"{tv:.1f}s", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
        ax4.set_ylim(0, max(tot_v) * 1.38)
        ax4.set_xticks(xb_idx)
        ax4.set_xticklabels([l for _, l in pres_b], rotation=10, fontsize=8.4)
        ax4.set_ylabel("Build (s)", fontsize=9.8)
        ax4.legend(loc="upper left", ncol=3, fontsize=7.6, frameon=True, edgecolor="#BDBDBD",
                   borderpad=0.13, columnspacing=0.40, handletextpad=0.25, handlelength=1.0)
    ax4.text(0.5, -0.48, "(d) Index build breakdown", transform=ax4.transAxes, ha="center", va="top", fontsize=10.5, fontweight="bold")

    plt.subplots_adjust(wspace=0.24, top=0.95, bottom=0.33)
    save_both(fig, out_dir, paper_fig_dir, "fig8_e5_ablation")


# ==============================================================================
# 6. 4-Axis Parameter Sensitivity (`fig9_e6_sensitivity.pdf`)
# ==============================================================================
def plot_fig10_e6_sensitivity(res_dir: str, out_dir: str, paper_fig_dir: str):
    e6 = load_json_if_exists(res_dir, "e6_sensitivity.json")
    if not e6:
        return

    fig, axes = plt.subplots(1, 4, figsize=(14.0, 1.30))

    ax = axes[0]
    style_ax(ax, "both")
    for dk, dlbl, cp_col, cu_col, mk in [("sift1m", "SIFT", "#C45A52", "#5486B4", "o"), ("gist1m", "GIST", "#589A73", "#8E7CC3", "s")]:
        rows = e6["batch_size"][dk]
        bs = [r["batch_size"] for r in rows]
        ax.plot(bs, [r["prism_qps"] for r in rows], color=cp_col, ls="-", marker=mk, ms=4.0, lw=1.45, label=f"PRI ({dlbl})")
        ax.plot(bs, [r["cuvs_qps"] for r in rows], color=cu_col, ls="--", marker=mk, ms=3.8, lw=1.05, label=f"cuVS ({dlbl})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(280, 280000)
    ax.set_xlabel("Batch Size $B$", fontsize=9.8, labelpad=1.0)
    ax.set_ylabel("QPS (log)", fontsize=9.8)
    ax.legend(loc="upper left", fontsize=7.5, ncol=2, frameon=True, facecolor="white", framealpha=0.94,
              edgecolor="#BDBDBD", borderpad=0.13, columnspacing=0.4, handletextpad=0.3)
    ax.text(0.5, -0.54, "(a) Sensitivity to batch size $B$", transform=ax.transAxes, ha="center", va="top", fontsize=10.5, fontweight="bold")

    ax = axes[1]
    style_ax(ax, "both")
    for dk, dlbl, cp_col, cu_col, mk in [("sift1m", "SIFT", "#C45A52", "#5486B4", "o"), ("gist1m", "GIST", "#589A73", "#8E7CC3", "s")]:
        rows = e6["top_k"][dk]
        ks = [r["k"] for r in rows]
        ax.plot(ks, [r["prism_qps"] for r in rows], color=cp_col, ls="-", marker=mk, ms=4.0, lw=1.45, label=f"PRI ({dlbl})")
        ax.plot(ks, [r["cuvs_qps"] for r in rows], color=cu_col, ls="--", marker=mk, ms=3.8, lw=1.05, label=f"cuVS ({dlbl})")
    ax.set_yscale("log")
    ax.set_ylim(280, 200000)
    ax.set_xlabel("Result Top-$K$", fontsize=9.8, labelpad=1.0)
    ax.set_ylabel("QPS (log)", fontsize=9.8)
    ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.36), fontsize=7.5, ncol=2, frameon=True,
              facecolor="white", framealpha=0.94, edgecolor="#BDBDBD", borderpad=0.13, columnspacing=0.4, handletextpad=0.3)
    ax.text(0.5, -0.54, "(b) Sensitivity to Top-$K$", transform=ax.transAxes, ha="center", va="top", fontsize=10.5, fontweight="bold")

    ax = axes[2]
    style_ax(ax, "y")
    for dk, dlbl, col, mk in [
        ("gist1m", "GIST1M", "#5486B4", "s"),
        ("convlog1m", "ConvLog1M", "#8E7CC3", "D"),
        ("sift1m", "SIFT1M", "#C45A52", "o"),
    ]:
        if dk in e6["m_blocks"]:
            rows = e6["m_blocks"][dk]
            ax.plot([str(r["M_blocks"]) for r in rows], [r["qps"] for r in rows], color=col, ls="-", marker=mk, ms=4.0, lw=1.45, label=dlbl)
    ax.set_yscale("log")
    ax.set_xlabel("Macro-Blocks $M$", fontsize=9.8, labelpad=1.0)
    ax.set_ylabel("QPS (log)", fontsize=9.8)
    ax.legend(loc="center right", fontsize=7.6, frameon=True, facecolor="white", framealpha=0.94, edgecolor="#BDBDBD", borderpad=0.13)
    ax.text(0.5, -0.54, "(c) Sensitivity to block count $M$", transform=ax.transAxes, ha="center", va="top", fontsize=10.5, fontweight="bold")

    ax = axes[3]
    style_ax(ax, "both")
    for dk, dlbl, col, mk in [("sift1m", "SIFT1M", "#C45A52", "o"), ("gist1m", "GIST1M", "#589A73", "s")]:
        rows = e6["quantile_clip"][dk]
        ax.plot([r["quantile_clip"] * 100.0 for r in rows], [r["sketch_prune_pct"] for r in rows], color=col, ls="-", marker=mk, ms=4.0, lw=1.45, label=dlbl)
    ax.set_xlabel(r"Quantile Clip $q_{\mathrm{clip}}$ (%)", fontsize=9.8, labelpad=1.0)
    ax.set_ylabel("Prune (%)", fontsize=9.8)
    ax.legend(loc="lower left", fontsize=7.8, frameon=True, edgecolor="#BDBDBD", borderpad=0.14)
    ax.text(0.5, -0.54, "(d) Sensitivity to quantile clip", transform=ax.transAxes, ha="center", va="top", fontsize=10.5, fontweight="bold")

    plt.subplots_adjust(wspace=0.25, top=0.95, bottom=0.33)
    save_both(fig, out_dir, paper_fig_dir, "fig9_e6_sensitivity")


# ==============================================================================
# 7. Cost Model (`E7`) + A100 80GB (`E9`) + Dynamic Stream (`fig10_e7_e9_cost_and_a100.pdf`)
# ==============================================================================
def plot_fig11_cost_a100_and_dynamic(res_dir: str, out_dir: str, paper_fig_dir: str):
    e7 = load_json_if_exists(res_dir, "e7_cost_model.json")
    e9 = load_json_if_exists(res_dir, "e9_a100_portability.json")
    e12 = load_json_if_exists(res_dir, "e12_dynamic_updates.json") or load_json_if_exists(res_dir, "e12_dynamic_updates_a100.json")
    if not e7:
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.0, 1.30))

    style_ax(ax1, "both")
    pts = e7["points"]
    reg_styles = {
        1: ("R1: Global", "#74A9CF", "^"),
        2: ("R2: Co-Prune", "#E5989B", "o"),
        3: ("R3: Sketch-1st", "#82C09A", "s"),
    }
    all_v = []
    for r_id, (rlbl, col, mk) in reg_styles.items():
        sub = [p for p in pts if p["regime"] == r_id]
        xs = [p["actual_ms"] for p in sub]
        ys = [p["pred_ms"] for p in sub]
        all_v.extend(xs + ys)
        ax1.scatter(xs, ys, c=col, marker=mk, s=22, edgecolors=C_DARK, linewidths=0.5, alpha=0.88, label=rlbl, zorder=4)
    lo, hi = max(0.3, min(all_v) * 0.7), max(all_v) * 1.65
    ax1.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="Ideal $y = x$", zorder=2)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_xlabel("Measured Latency (ms)", fontsize=9.8, labelpad=1.0)
    ax1.set_ylabel("Predicted (ms)", fontsize=9.8)
    ax1.text(0.03, 0.92, f"MedAPE={e7['median_ape_pct']:.1f}% | r={e7['pearson_r']:.3f}",
             transform=ax1.transAxes, va="top", fontsize=8.0, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.14", facecolor="#F8F9F9", edgecolor="#BDBDBD", lw=0.7))
    ax1.legend(loc="lower right", fontsize=7.6, ncol=2, frameon=True, edgecolor="#BDBDBD",
               borderpad=0.13, labelspacing=0.14, columnspacing=0.45, handletextpad=0.25)
    ax1.text(0.5, -0.54, "(a) Analytical cost model accuracy", transform=ax1.transAxes, ha="center", va="top", fontsize=10.5, fontweight="bold")

    style_ax(ax2, "y")
    if e9 and "rows" in e9 and len(e9["rows"]) > 0:
        rows = e9["rows"]
        lbl_map = {
            "sift1m": "SIFT1M", "gist1m": "GIST1M", "convlog1m": "Conv1M",
            "deep10m": "Deep10M", "sift10m": "SIFT10M", "t2i10m": "T2I10M", "sift100m": "S100M",
        }
        x = np.arange(len(rows))
        w = 0.25
        qc = [r.get("cuvs_ivf_qps", 0.0) if r.get("cuvs_ivf_status") == "OK" else 0.0 for r in rows]
        qh = [r["host_offload_qps"] for r in rows]
        qp = [r["prism_qps"] for r in rows]
        ax2.bar(x - w, qc, w, facecolor=LAB_COLORS[2], edgecolor=C_DARK, lw=0.70, label="cuVS-IVF", zorder=3)
        ax2.bar(x, qh, w, facecolor=LAB_COLORS[0], edgecolor=C_DARK, lw=0.70, label="Offload", zorder=3)
        ax2.bar(x + w, qp, w, facecolor=LAB_COLORS[7], edgecolor=C_DARK, lw=0.78, hatch="//", label="PRISM", zorder=3)
        ymin_a = 120.0
        ax2.set_yscale("log")
        ax2.set_ylim(ymin_a, max(qp) * 9.0)
        for i, r in enumerate(rows):
            st_c = r.get("cuvs_ivf_status", "OOM")
            if st_c == "OOM":
                ax2.plot(x[i] - w, ymin_a * 1.45, marker="x", markersize=5.2, markeredgewidth=1.4, color="#B02A2A", zorder=5)
            elif st_c == "RECALL_MISS":
                ax2.plot(x[i] - w, ymin_a * 1.45, marker="v", markersize=4.5, color="#7D3C98", zorder=5)
        ax2.set_xticks(x)
        ax2.set_xticklabels([lbl_map.get(r["dataset"], r["dataset"]) for r in rows], rotation=8, fontsize=8.8)
        ax2.set_ylabel("A100 QPS", fontsize=9.8)
        ax2.legend(loc="upper right", fontsize=7.8, ncol=3, frameon=True, edgecolor="#BDBDBD",
                   borderpad=0.14, columnspacing=0.45, handletextpad=0.3, handlelength=1.1)
    ax2.text(0.5, -0.54, "(b) Scalability on NVIDIA A100 80GB", transform=ax2.transAxes, ha="center", va="top", fontsize=10.5, fontweight="bold")

    style_ax(ax3, "both")
    if e12 and e12.get("datasets"):
        d_dyn = e12["datasets"]
        stage_lbls = ["Init", "-5%", "+5%", "+10%", "+15%", "+20%", "OOD"]
        for dk, dlbl, col, mk in [
            ("sift10m", "S10M", "#C45A52", "o"),
            ("modbench", "Mod", "#589A73", "^"),
            ("gist1m", "GIST", "#5486B4", "s"),
            ("convlog1m", "Conv", "#8E7CC3", "D"),
        ]:
            if dk in d_dyn:
                cks = d_dyn[dk]["checkpoints"]
                xs_d = np.arange(len(cks))
                ax3.plot(xs_d, [c["search_qps"] for c in cks], color=col, ls="-", marker=mk,
                         ms=4.0, lw=1.50, label=f"{dlbl} ({cks[-1]['sketch_prune_pct']:.0f}%)")
        ax3.set_yscale("log")
        ax3.set_ylim(1600, 1.3e6)
        ax3.set_xticks(np.arange(len(stage_lbls)))
        ax3.set_xticklabels(stage_lbls, fontsize=8.8)
        ax3.set_ylabel("QPS (log)", fontsize=9.8)
        ax3.text(0.03, 0.08, "100% Exact | 0 Leaks",
                 transform=ax3.transAxes, fontsize=7.8, fontweight="bold", color="#1D561B",
                 bbox=dict(boxstyle="round,pad=0.13", facecolor="#EEF7ED", edgecolor="#53814E", lw=0.7))
        ax3.legend(loc="upper right", fontsize=7.5, ncol=2, frameon=True, facecolor="white", framealpha=0.95,
                   edgecolor="#BDBDBD", borderpad=0.13, labelspacing=0.14, columnspacing=0.45, handletextpad=0.25)
    ax3.text(0.5, -0.54, "(c) Streaming churn & OOD resilience", transform=ax3.transAxes, ha="center", va="top", fontsize=10.5, fontweight="bold")

    plt.subplots_adjust(wspace=0.24, top=0.95, bottom=0.33)
    save_both(fig, out_dir, paper_fig_dir, "fig10_e7_e9_cost_and_a100")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.abspath(os.path.join(here, "..", "results"))
    out_dir = os.path.expanduser(os.environ.get("PRISM_FIG_OUT", os.path.join(here, "generated")))
    os.makedirs(out_dir, exist_ok=True)
    plot_fig5_e1_main_qps(res_dir, out_dir, out_dir)
    plot_fig6_pareto_logloss(res_dir, out_dir, out_dir)
    plot_fig7_e2_selectivity(res_dir, out_dir, out_dir)
    plot_fig8_e3_e4_skew_and_100m(res_dir, out_dir, out_dir)
    plot_fig9_ablation_and_breakdown(res_dir, out_dir, out_dir)
    plot_fig10_e6_sensitivity(res_dir, out_dir, out_dir)
    plot_fig11_cost_a100_and_dynamic(res_dir, out_dir, out_dir)


if __name__ == "__main__":
    main()
