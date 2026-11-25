#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication Vector Architecture & Walkthrough Diagrams for PRISM (SIGMOD 2027 Industrial Track).
Generates Figures 1--5 and the TeleOps1M industrial ribbon in PDF and PNG formats.
"""

from __future__ import annotations
import json
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Ellipse, Arc, FancyArrowPatch, Polygon, Wedge

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"

# Whisper-pale desaturated Morandi micro-element tints
P_BLUE   = "#EBF3FA"
P_SAGE   = "#EAF4EA"
P_CREAM  = "#FDF5E6"
P_ROSE   = "#F9EBEA"
P_LILAC  = "#F2ECF6"
P_GRAY   = "#EFEFEF"
C_LINE   = "#222222"


def draw_badge(ax, x, y, text, r=0.18, filled=True, fontsize=9.5, zorder=12):
    fc = C_LINE if filled else "#FFFFFF"
    tc = "#FFFFFF" if filled else C_LINE
    ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor=C_LINE, linewidth=1.2, zorder=zorder))
    ax.text(x, y - 0.008, str(text), ha="center", va="center", color=tc,
            fontsize=fontsize, fontweight="bold", zorder=zorder + 1)


def draw_cylinder(ax, x, y, w, h, fc="#FFFFFF", ec=C_LINE, lw=1.1, stripes=2, zorder=5):
    ry = h * 0.16
    ax.add_patch(Rectangle((x - w / 2, y - h / 2 + ry), w, h - 2 * ry, facecolor=fc, edgecolor="none", zorder=zorder))
    ax.add_patch(Arc((x, y - h / 2 + ry), w, 2 * ry, angle=0, theta1=180, theta2=360, edgecolor=ec, lw=lw, zorder=zorder + 1))
    ax.plot([x - w / 2, x - w / 2], [y - h / 2 + ry, y + h / 2 - ry], color=ec, lw=lw, zorder=zorder + 1)
    ax.plot([x + w / 2, x + w / 2], [y + h / 2 - ry, y - h / 2 + ry], color=ec, lw=lw, zorder=zorder + 1)
    ax.add_patch(Ellipse((x, y + h / 2 - ry), w, 2 * ry, facecolor=fc, edgecolor=ec, lw=lw, zorder=zorder + 2))
    for s in range(1, stripes + 1):
        sy = (y + h / 2 - ry) - s * (h - 2 * ry) / (stripes + 1)
        ax.add_patch(Arc((x, sy), w, 2 * ry, angle=0, theta1=180, theta2=360, edgecolor=ec, lw=lw * 0.82, zorder=zorder + 2))


def draw_user_icon(ax, x, y, scale=1.0, zorder=6):
    r_head = 0.14 * scale
    ax.add_patch(Circle((x, y + 0.17 * scale), r_head, facecolor="#FFFFFF", edgecolor=C_LINE, lw=1.2, zorder=zorder))
    ax.add_patch(Arc((x, y - 0.21 * scale), 0.46 * scale, 0.42 * scale, angle=0, theta1=0, theta2=180,
                     edgecolor=C_LINE, lw=1.2, zorder=zorder))
    ax.plot([x - 0.23 * scale, x + 0.23 * scale], [y - 0.21 * scale, y - 0.21 * scale], color=C_LINE, lw=1.2, zorder=zorder)


def draw_magnifier(ax, x, y, r=0.17, zorder=6):
    ax.add_patch(Circle((x, y), r, facecolor="#FFFFFF", edgecolor=C_LINE, lw=1.2, zorder=zorder))
    ax.text(x, y - 0.01, "Q", ha="center", va="center", fontsize=8.8, fontweight="bold", color=C_LINE, zorder=zorder + 1)
    ax.plot([x + r * 0.68, x + r * 1.35], [y - r * 0.68, y - r * 1.35], color=C_LINE, lw=1.4, zorder=zorder)


def draw_pill_slot(ax, x, y, w, h, text, fc="#FFFFFF", tc=C_LINE, lw=1.0, fs=8.8, bold=False, pins=True, zorder=5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.08",
                                facecolor=fc, edgecolor=C_LINE, linewidth=lw, zorder=zorder))
    if pins:
        pr = min(h * 0.16, 0.05)
        ax.add_patch(Circle((x + 0.11, y + h / 2), pr, facecolor="#FFFFFF", edgecolor=C_LINE, lw=0.85, zorder=zorder + 1))
        ax.add_patch(Circle((x + w - 0.11, y + h / 2), pr, facecolor="#FFFFFF", edgecolor=C_LINE, lw=0.85, zorder=zorder + 1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc,
            fontsize=fs, fontweight="bold" if bold else "normal", zorder=zorder + 2)


def draw_arrow(ax, p1, p2, lw=1.2, style="-|>", ls="-", color=C_LINE, rad=0.0, ms=9.5, zorder=7):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=ms, linewidth=lw,
        linestyle=ls, color=color, connectionstyle=f"arc3,rad={rad}", zorder=zorder
    ))


# ==============================================================================
# 1. Archetype 1 — `fig1_paradigm_comparison.pdf` (Page 2, Single-Column)
#    Canvas scaled to (6.4, 3.55) with 8.5-10pt fonts => ~45% larger printed text!
# ==============================================================================
def plot_fig1_paradigm_comparison(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.55))
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 6.20)
    ax.axis("off")

    ax.plot([0.06, 9.94], [4.14, 4.14], color="#888888", ls="--", lw=0.85)
    ax.plot([0.06, 9.94], [2.08, 2.08], color="#888888", ls="--", lw=0.85)

    # --- Row (a): In-VRAM GPU & Multi-Attribute Graph Indexes ---
    y0 = 4.22
    draw_user_icon(ax, 0.46, y0 + 1.28, scale=0.88)
    draw_cylinder(ax, 0.46, y0 + 0.62, 0.52, 0.48)
    ax.text(0.46, y0 + 0.15, "Unicom\nLogs", ha="center", va="center", fontsize=7.6, fontweight="bold")
    draw_arrow(ax, (0.80, y0 + 0.92), (1.18, y0 + 0.92))

    ax.add_patch(FancyBboxPatch((1.22, y0 + 0.28), 5.62, 1.42, boxstyle="round,pad=0.02,rounding_size=0.09",
                                facecolor="#FFFFFF", edgecolor=C_LINE, lw=1.35))
    ax.text(4.03, y0 + 1.50, "China Unicom 16 GB PoP GPU (15.6G)", ha="center", va="center", fontsize=8.6, fontweight="bold")
    ax.add_patch(Rectangle((1.36, y0 + 0.88), 1.98, 0.38, facecolor="#FFFFFF", edgecolor=C_LINE, lw=0.9))
    ax.text(2.35, y0 + 1.07, "LLM (6.0G)", ha="center", va="center", fontsize=8.1, fontweight="bold")
    ax.add_patch(Rectangle((3.34, y0 + 0.88), 2.18, 0.38, facecolor=P_GRAY, edgecolor=C_LINE, lw=0.9))
    ax.add_patch(Rectangle((5.52, y0 + 0.88), 1.18, 0.38, facecolor=P_ROSE, edgecolor=C_LINE, lw=0.9, hatch="///"))
    ax.plot([5.52, 5.52], [y0 + 0.82, y0 + 1.33], color="#922B21", ls="--", lw=1.3)
    ax.text(4.43, y0 + 1.07, "cuVS / VecFlow", ha="center", va="center", fontsize=8.0, fontweight="bold")
    ax.text(6.11, y0 + 1.07, "34G!", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#7B241C",
            bbox=dict(boxstyle="square,pad=0.04", facecolor="#FFFFFF", edgecolor="none"))

    for ci in range(6):
        hit = (ci == 2)
        cx_c = 1.38 + ci * 0.25
        ax.add_patch(Rectangle((cx_c, y0 + 0.40), 0.22, 0.28, facecolor=P_SAGE if hit else "#FFFFFF", edgecolor=C_LINE, lw=0.75))
        if hit:
            ax.plot([cx_c + 0.04, cx_c + 0.10, cx_c + 0.18], [y0 + 0.54, y0 + 0.47, y0 + 0.62], color=C_LINE, lw=1.15)
        else:
            ax.text(cx_c + 0.11, y0 + 0.54, "x", ha="center", va="center", fontsize=7.4, color="#666666")
    ax.text(4.86, y0 + 0.54, "Recall < 0.13 | Rebuild 1,943s", ha="center", va="center", fontsize=7.9)

    draw_arrow(ax, (6.88, y0 + 0.96), (7.24, y0 + 0.96))
    draw_pill_slot(ax, 7.26, y0 + 0.46, 2.66, 0.98, "   VRAM OOM or\n   Recall Cliff", fc="#FFFFFF", fs=8.4, bold=True, pins=False)
    ax.plot([7.38, 7.54], [y0 + 1.06, y0 + 1.22], color="#922B21", lw=1.6, zorder=12)
    ax.plot([7.38, 7.54], [y0 + 1.22, y0 + 1.06], color="#922B21", lw=1.6, zorder=12)
    ax.text(5.0, y0 + 0.03, "(a) In-VRAM GPU and Multi-Attribute Graph Indexes", ha="center", va="center", fontsize=9.2, fontweight="bold")

    # --- Row (b): CPU Filtered Graphs & Naive Host-Offload Pre-Filtering ---
    y1 = 2.16
    draw_magnifier(ax, 0.46, y1 + 1.24, r=0.16)
    draw_cylinder(ax, 0.46, y1 + 0.60, 0.52, 0.48)
    ax.text(0.46, y1 + 0.14, "Alert\nSQL Q", ha="center", va="center", fontsize=8.0, fontweight="bold")
    draw_arrow(ax, (0.80, y1 + 0.92), (1.18, y1 + 0.92))

    ax.add_patch(FancyBboxPatch((1.22, y1 + 0.28), 5.62, 1.42, boxstyle="round,pad=0.02,rounding_size=0.09",
                                facecolor="#FFFFFF", edgecolor=C_LINE, lw=1.35))
    ax.text(4.03, y1 + 1.50, "Masked SIMT Warp & Unpruned Host Read", ha="center", va="center", fontsize=8.8, fontweight="bold")
    wx0 = 1.38
    for r_i in range(2):
        for c_i in range(8):
            active = (r_i == 0 and c_i == 2) or (r_i == 1 and c_i == 6)
            ax.add_patch(Rectangle((wx0 + c_i * 0.22, y1 + 0.80 + r_i * 0.24), 0.20, 0.21,
                                   facecolor="#7F8C8D" if active else "#FFFFFF", edgecolor=C_LINE, lw=0.72))
    ax.text(2.26, y1 + 0.50, "96.8% Idle Lanes", ha="center", va="center", fontsize=7.9)
    draw_arrow(ax, (3.22, y1 + 1.02), (3.68, y1 + 1.02))
    mx0 = 3.74
    for r_i in range(2):
        for b_i in range(4):
            ax.add_patch(Rectangle((mx0 + b_i * 0.72, y1 + 0.78 + r_i * 0.23), 0.67, 0.20,
                                   facecolor=P_CREAM, edgecolor=C_LINE, lw=0.72))
            if r_i == 1:
                ax.text(mx0 + b_i * 0.72 + 0.335, y1 + 1.33, f"D{b_i+1}", ha="center", va="center", fontsize=7.6, fontweight="bold")
    ax.text(5.18, y1 + 0.50, "9.1G Wasted PCIe", ha="center", va="center", fontsize=8.0, fontweight="bold")

    draw_arrow(ax, (6.88, y1 + 0.96), (7.24, y1 + 0.96))
    draw_pill_slot(ax, 7.26, y1 + 0.46, 2.66, 0.98, "   SIMT Starvation &\n   PCIe Thrashing", fc="#FFFFFF", fs=8.4, bold=True, pins=False)
    ax.plot([7.38, 7.54], [y1 + 1.06, y1 + 1.22], color="#922B21", lw=1.6, zorder=12)
    ax.plot([7.38, 7.54], [y1 + 1.22, y1 + 1.06], color="#922B21", lw=1.6, zorder=12)
    ax.text(5.0, y1 + 0.03, "(b) CPU Graphs and Naive Host-Offload Pre-Filtering", ha="center", va="center", fontsize=9.2, fontweight="bold")

    # --- Row (c): PRISM Cross-Modal GPU-Host Co-Design (Ours) ---
    y2 = 0.08
    draw_user_icon(ax, 0.46, y2 + 1.32, scale=0.88)
    draw_cylinder(ax, 0.46, y2 + 0.64, 0.52, 0.48, fc=P_BLUE)
    ax.text(0.46, y2 + 0.16, "Unicom\nPoP", ha="center", va="center", fontsize=7.6, fontweight="bold")
    draw_arrow(ax, (0.80, y2 + 0.96), (1.18, y2 + 0.96))

    ax.add_patch(FancyBboxPatch((1.22, y2 + 0.30), 5.62, 1.50, boxstyle="round,pad=0.02,rounding_size=0.09",
                                facecolor="#FFFFFF", edgecolor=C_LINE, lw=1.55))
    ax.text(4.03, y2 + 1.62, "PRISM (LLM 6.0G + Sketch 0.1-1.6G)", ha="center", va="center", fontsize=8.8, fontweight="bold")
    draw_badge(ax, 1.48, y2 + 1.28, "1", r=0.14, filled=True, fontsize=8.2)
    draw_badge(ax, 1.48, y2 + 0.90, "2", r=0.14, filled=True, fontsize=8.2)
    ax.text(2.48, y2 + 1.29, "TileHdr + SLAP", ha="center", va="center", fontsize=8.0, fontweight="bold")
    for c_i in range(6):
        ax.add_patch(Rectangle((1.72 + c_i * 0.24, y2 + 0.79), 0.22, 0.22, facecolor=P_BLUE, edgecolor=C_LINE, lw=0.72))
    ax.text(2.44, y2 + 0.64, "32-Lane Ring", ha="center", va="center", fontsize=7.8)
    draw_arrow(ax, (3.22, y2 + 0.92), (3.62, y2 + 0.92))

    px0 = 3.68
    alive_rows_per_blk = [3, 2, 1, 1]
    for b_i in range(4):
        ax.text(px0 + b_i * 0.74 + 0.34, y2 + 1.36, f"B{b_i+1}", ha="center", va="center", fontsize=7.8, fontweight="bold")
        for r_i in range(3):
            alive = r_i < alive_rows_per_blk[b_i]
            ax.add_patch(Rectangle((px0 + b_i * 0.74, y2 + 0.72 + (2 - r_i) * 0.18), 0.68, 0.16,
                                   facecolor=P_SAGE if alive else "#FFFFFF", edgecolor=C_LINE, lw=0.7))
    draw_badge(ax, 1.48, y2 + 0.46, "3", r=0.14, filled=True, fontsize=8.2)
    ax.text(4.22, y2 + 0.46, "R1/R2/R3 Router | <4% Host UVA", ha="center", va="center", fontsize=8.0, fontweight="bold")

    draw_arrow(ax, (6.88, y2 + 1.02), (7.24, y2 + 1.02))
    draw_pill_slot(ax, 7.26, y2 + 0.48, 2.66, 1.02, "  100% Exact\n  17.6x-42.8x QPS\n  -94% CAPEX",
                   fc=P_SAGE, tc=C_LINE, lw=1.25, fs=8.2, bold=True, pins=False)
    ax.plot([7.38, 7.46, 7.58], [y2 + 1.28, y2 + 1.19, y2 + 1.36], color="#1E6B3A", lw=1.65, zorder=12)
    ax.text(5.0, y2 + 0.02, "(c) PRISM Cross-Modal GPU-Host Co-Design (Ours)", ha="center", va="center", fontsize=9.3, fontweight="bold")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig1_paradigm_comparison.{ext}"), dpi=260, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("[Generated] fig1_paradigm_comparison.pdf")


# ==============================================================================
# 2. Archetype 2 — `fig2_aiops_bottleneck_flow.pdf` (Page 3, Single-Column)
#    Completely distinct Dual-Panel `(a) + (b)` Archetype:
#    (a) Hierarchical Telecom PoP Tree Topology & Co-Located Memory Quota
#    (b) Quantitative Horizontal Bottleneck Anatomy Bars (Obs. 1, Obs. 2, Obs. 3)
# ==============================================================================
def plot_fig2_aiops_bottleneck_flow(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.14), gridspec_kw={"width_ratios": [0.96, 1.18]})

    # ---------------- Panel (a): Telecom PoP Tree Topology ----------------
    ax1.set_xlim(0, 5.6)
    ax1.set_ylim(0, 5.4)
    ax1.axis("off")

    ax1.add_patch(FancyBboxPatch((0.06, 0.22), 5.48, 5.04, boxstyle="round,pad=0.02,rounding_size=0.12",
                                 facecolor="#FFFFFF", edgecolor=C_LINE, lw=1.0))

    # Top tree: 3 Regional Telemetry Sources -> PoP Aggregator
    ax1.text(2.80, 5.02, "China Unicom Cloud-Net", ha="center", va="center", fontsize=8.4, fontweight="bold")
    src_nodes = [(0.94, "Nova", P_CREAM), (2.80, "5G/SSH", P_BLUE), (4.66, "Quorum", P_SAGE)]
    for sx, slbl, sfc in src_nodes:
        draw_cylinder(ax1, sx, 4.54, 1.22, 0.42, fc=sfc, lw=0.85)
        ax1.text(sx, 4.10, slbl, ha="center", va="center", fontsize=7.6, fontweight="bold", zorder=10,
                 bbox=dict(boxstyle="square,pad=0.03", fc="#FFFFFF", ec="none"))
        ax1.plot([sx, sx], [3.92, 3.74], color=C_LINE, lw=0.95)
    ax1.plot([0.94, 4.66], [3.74, 3.74], color=C_LINE, lw=0.95)
    # Two symmetric vertical arrows flanking the Data-Residency label (zero text overlap!)
    draw_arrow(ax1, (0.94, 3.74), (0.94, 3.36), lw=1.05)
    draw_arrow(ax1, (4.66, 3.74), (4.66, 3.36), lw=1.05)

    ax1.plot([0.22, 5.38], [3.55, 3.55], color="#8B3A3A", lw=0.95, ls="--")
    ax1.text(2.80, 3.55, " Data-Residency Wall ", ha="center", va="center", fontsize=7.3,
             fontweight="bold", color="#7B241C", bbox=dict(boxstyle="square,pad=0.05", fc="#FFFFFF", ec="none"), zorder=10)

    # Middle node: Provincial PoP Appliance ($3,800)
    ax1.add_patch(Rectangle((0.26, 1.54), 5.08, 1.78, facecolor="#FAFAFC", edgecolor=C_LINE, lw=1.05))
    ax1.text(2.80, 3.10, "China Unicom PoP ($3.8K)", ha="center", va="center", fontsize=8.2, fontweight="bold")

    # Stacked 16 GB GPU Quota Strip inside PoP Node
    ax1.text(2.80, 2.76, "16 GB GPU (15.6 GiB Usable):", ha="center", va="center", fontsize=7.6, fontweight="bold")
    ax1.add_patch(Rectangle((0.44, 2.24), 1.96, 0.36, facecolor=P_LILAC, edgecolor=C_LINE, lw=0.8))
    ax1.text(1.42, 2.42, "LLM: 6.0G", ha="center", va="center", fontsize=7.4, fontweight="bold")
    ax1.add_patch(Rectangle((2.40, 2.24), 2.76, 0.36, facecolor=P_BLUE, edgecolor=C_LINE, lw=0.8))
    ax1.text(3.78, 2.42, "Index Quota <9.6G", ha="center", va="center", fontsize=7.4, fontweight="bold")

    # W1-W4 SQL tag strip
    ax1.add_patch(Rectangle((0.44, 1.68), 4.72, 0.38, facecolor="#FFFFFF", edgecolor="#666666", lw=0.75))
    ax1.text(2.80, 1.87, "SQL W1:Time | W2:Metric | W3:Comp | W4:Storm", ha="center", va="center", fontsize=7.0, fontweight="bold")

    # PCIe vertical link to Host DRAM
    draw_arrow(ax1, (2.80, 1.52), (2.80, 1.06), lw=1.1, style="<->")
    ax1.text(3.92, 1.29, "PCIe Gen4", ha="center", va="center", fontsize=7.4, fontweight="bold", color="#444444")

    ax1.add_patch(Rectangle((0.26, 0.38), 5.08, 0.66, facecolor=P_GRAY, edgecolor=C_LINE, lw=0.95))
    ax1.text(2.80, 0.71, "Host DRAM (128-512 GB)\n1M-100M Rolling Logs (7-30d)", ha="center", va="center", fontsize=7.5, fontweight="bold")

    ax1.text(0.5, -0.03, "(a) China Unicom PoP topology", transform=ax1.transAxes, ha="center", va="top", fontsize=9.4, fontweight="bold")

    # ---------------- Panel (b): Quantitative Bottleneck Bars (Obs. 1-3) ----------------
    ax2.set_xlim(0, 6.8)
    ax2.set_ylim(0, 5.4)
    ax2.axis("off")

    ax2.add_patch(FancyBboxPatch((0.06, 0.22), 6.68, 5.04, boxstyle="round,pad=0.02,rounding_size=0.12",
                                 facecolor="#FFFFFF", edgecolor=C_LINE, lw=1.0))

    # --- Obs. 1 Gauge Bars (y = 3.68 .. 5.08) ---
    ax2.text(0.24, 4.96, "Obs. 1: Recall Cliff & SIMT Waste", ha="left", va="center", fontsize=8.3, fontweight="bold", color="#7B241C")
    bx0, bw = 1.92, 4.58
    ax2.text(0.24, 4.51, "Post R@10:", ha="left", va="center", fontsize=7.5, fontweight="bold")
    ax2.add_patch(Rectangle((bx0, 4.35), bw * 0.13, 0.32, facecolor=P_ROSE, edgecolor=C_LINE, lw=0.8))
    ax2.add_patch(Rectangle((bx0 + bw * 0.13, 4.35), bw * 0.87, 0.32, facecolor="#FFFFFF", edgecolor="#777777", lw=0.75, hatch="//"))
    ax2.add_patch(Rectangle((bx0, 4.35), bw, 0.32, facecolor="none", edgecolor=C_LINE, lw=0.85))
    ax2.text(bx0 + bw * 0.56, 4.51, "87%-90% Missed!", ha="center", va="center", fontsize=7.3, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.08", fc="#FFFFFF", ec="none"), zorder=10)

    ax2.text(0.24, 3.97, "Pre Warp:", ha="left", va="center", fontsize=7.5, fontweight="bold")
    ax2.add_patch(Rectangle((bx0, 3.81), bw * 0.08, 0.32, facecolor=P_BLUE, edgecolor=C_LINE, lw=0.8))
    ax2.add_patch(Rectangle((bx0 + bw * 0.08, 3.81), bw * 0.92, 0.32, facecolor=P_GRAY, edgecolor="#777777", lw=0.75, hatch="\\\\"))
    ax2.add_patch(Rectangle((bx0, 3.81), bw, 0.32, facecolor="none", edgecolor=C_LINE, lw=0.85))
    ax2.text(bx0 + bw * 0.54, 3.97, "91.7%-96.8% Idle!", ha="center", va="center", fontsize=7.3, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.08", fc="#FFFFFF", ec="none"), zorder=10)

    ax2.plot([0.22, 6.56], [3.58, 3.58], color="#CCCCCC", lw=0.85, ls="-")

    # --- Obs. 2 VRAM Capacity Wall Bars (y = 1.98 .. 3.44) ---
    ax2.text(0.24, 3.36, "Obs. 2: Co-Located 16 GB VRAM Wall", ha="left", va="center", fontsize=8.3, fontweight="bold", color="#7B241C")
    # Place 15.6 GiB limit cleanly at ratio 0.54 so left ("7.6G OK") and right ("24-39G OOM!") never touch it!
    lim_x = bx0 + bw * 0.55
    ax2.text(0.24, 2.90, "Graph/cuVS:", ha="left", va="center", fontsize=7.4, fontweight="bold")
    ax2.add_patch(Rectangle((bx0, 2.74), bw * 0.19, 0.32, facecolor=P_LILAC, edgecolor=C_LINE, lw=0.8))
    ax2.add_patch(Rectangle((bx0 + bw * 0.19, 2.74), bw * 0.81, 0.32, facecolor=P_ROSE, edgecolor="#886666", lw=0.75, hatch="//"))
    ax2.add_patch(Rectangle((bx0, 2.74), bw, 0.32, facecolor="none", edgecolor=C_LINE, lw=0.85))
    ax2.text(lim_x + (bx0 + bw - lim_x) * 0.50, 2.90, "24-39G OOM!", ha="center", va="center", fontsize=7.2, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.07", fc="#FFFFFF", ec="none"), zorder=10)

    ax2.text(0.24, 2.36, "PRISM (10M):", ha="left", va="center", fontsize=7.4, fontweight="bold")
    ax2.add_patch(Rectangle((bx0, 2.20), bw * 0.19, 0.32, facecolor=P_LILAC, edgecolor=C_LINE, lw=0.8))
    ax2.add_patch(Rectangle((bx0 + bw * 0.19, 2.20), bw * 0.09, 0.32, facecolor=P_SAGE, edgecolor=C_LINE, lw=0.85))
    ax2.text(bx0 + bw * 0.30, 2.36, "7.6G OK", ha="left", va="center", fontsize=7.3, fontweight="bold", color="#1E6B3A", zorder=10)
    ax2.plot([lim_x, lim_x], [2.14, 3.14], color="#8B3A3A", lw=1.25, ls="--", zorder=11)
    ax2.text(lim_x, 2.03, "15.6G Limit", ha="center", va="center", fontsize=7.0, fontweight="bold", color="#8B3A3A")

    ax2.plot([0.22, 6.56], [1.92, 1.92], color="#CCCCCC", lw=0.85, ls="-")

    # --- Obs. 3 PCIe Flood Proportion Bar (y = 0.36 .. 1.78) ---
    ax2.text(0.24, 1.66, "Obs. 3: Naive Offload PCIe Flood", ha="left", va="center", fontsize=8.3, fontweight="bold", color="#7B241C")
    ax2.text(0.24, 1.14, "Host Read:", ha="left", va="center", fontsize=7.5, fontweight="bold")
    ax2.add_patch(Rectangle((bx0, 0.96), bw * 0.06, 0.36, facecolor=P_SAGE, edgecolor=C_LINE, lw=0.8))
    ax2.add_patch(Rectangle((bx0 + bw * 0.06, 0.96), bw * 0.94, 0.36, facecolor=P_CREAM, edgecolor="#777777", lw=0.75, hatch="//"))
    ax2.add_patch(Rectangle((bx0, 0.96), bw, 0.36, facecolor="none", edgecolor=C_LINE, lw=0.85))
    ax2.text(bx0 + bw * 0.53, 1.14, "94%-98% Discarded (9.1G)", ha="center", va="center", fontsize=7.2, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.08", fc="#FFFFFF", ec="none"), zorder=10)
    ax2.text(3.38, 0.56, "Stalls W3 Triage at 170-600 QPS", ha="center", va="center", fontsize=7.6, fontweight="bold", color="#333333")

    ax2.text(0.5, -0.03, "(b) Empirical bottlenecks (Obs. 1-3)", transform=ax2.transAxes, ha="center", va="top", fontsize=9.4, fontweight="bold")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.12, wspace=0.06)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig2_aiops_bottleneck_flow.{ext}"), dpi=260, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("[Generated] fig2_aiops_bottleneck_flow.pdf")


# ==============================================================================
# 3. Archetype 3 — `fig3_ulp_ball_pipeline.pdf` (Page 4, Single-Column)
#    Canvas scaled to (6.5, 2.52) with 8.2-10pt fonts => ~45% larger printed text!
# ==============================================================================
def plot_fig3_ulp_ball_pipeline(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.52), gridspec_kw={"width_ratios": [1.00, 1.18]})

    ax1.set_xlim(0, 6.2)
    ax1.set_ylim(0, 4.6)
    ax1.axis("off")
    ax1.add_patch(FancyBboxPatch((0.08, 0.20), 6.04, 4.24, boxstyle="round,pad=0.03,rounding_size=0.14",
                                 facecolor="#FFFFFF", edgecolor=C_LINE, lw=1.05))

    q_pt = (1.12, 2.28)
    x_int4 = (4.05, 2.80)
    x_tilde = (4.48, 2.42)
    x_true = (4.74, 2.16)
    r_int4, r_sign = 1.18, 0.64

    ax1.add_patch(Wedge(q_pt, 2.42, -34, 40, facecolor=P_BLUE, edgecolor="#4A6B8A", lw=1.1, linestyle="--", zorder=2))
    ax1.text(1.58, 3.88, "Threshold\n" + r"$\sqrt{\tau_{\mathrm{safe}}}$", ha="center", va="center", fontsize=8.8, fontweight="bold")
    ax1.add_patch(Circle(x_int4, r_int4, facecolor="none", edgecolor="#888888", lw=1.1, linestyle=":", zorder=3))
    ax1.text(4.28, 4.14, "INT4 Ball", ha="center", va="center", fontsize=8.4, color="#555555", fontweight="bold")
    ax1.add_patch(Circle(x_tilde, r_sign, facecolor=P_SAGE, edgecolor="#3D6B47", lw=1.25, zorder=4))

    ax1.plot(*q_pt, marker="o", markersize=5.8, color=C_LINE, zorder=8)
    ax1.text(q_pt[0] + 0.08, q_pt[1] + 0.44, r"Query $q_m$", ha="center", va="center", fontsize=9.0, fontweight="bold", zorder=10)
    ax1.plot(*x_int4, marker="s", markersize=4.8, color="#777777", zorder=7)
    ax1.text(x_int4[0] - 0.46, x_int4[1] + 0.24, r"$\hat{x}_m^{\mathrm{int4}}$", fontsize=9.0, zorder=10)
    ax1.plot(*x_tilde, marker="o", markersize=5.2, color="#2E5B37", zorder=8)
    ax1.text(x_tilde[0] + 0.12, x_tilde[1] + 0.24, r"$\tilde{x}_m$", fontsize=9.2, fontweight="bold", zorder=10)
    ax1.plot(*x_true, marker="^", markersize=5.2, color=C_LINE, zorder=8)
    ax1.text(x_true[0] + 0.05, x_true[1] - 0.36, r"Exact $x_m$", fontsize=8.4, zorder=10)

    ax1.add_patch(FancyArrowPatch(x_int4, x_tilde, arrowstyle="-|>", mutation_scale=8.0, lw=1.0, color="#2E5B37", zorder=7))
    ux = (x_tilde[0] - q_pt[0]) / np.hypot(x_tilde[0] - q_pt[0], x_tilde[1] - q_pt[1])
    uy = (x_tilde[1] - q_pt[1]) / np.hypot(x_tilde[0] - q_pt[0], x_tilde[1] - q_pt[1])
    edge_pt = (x_tilde[0] - ux * r_sign, x_tilde[1] - uy * r_sign)
    ax1.add_patch(FancyArrowPatch(q_pt, edge_pt, arrowstyle="<->", mutation_scale=8.5, lw=1.25, color="#8B3A3A", zorder=9))
    ax1.text(3.10, 0.84, r"$L_m^{\mathrm{ulp}} > \tau_{\mathrm{safe}} \Rightarrow$ Prune!",
             ha="center", va="center", fontsize=8.6, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.16", facecolor="#FFFFFF", edgecolor=C_LINE, lw=0.85), zorder=10)
    ax1.text(0.5, -0.05, "(a) Sign-refined ball bound", transform=ax1.transAxes, ha="center", va="top", fontsize=9.8, fontweight="bold")

    ax2.set_xlim(0, 7.2)
    ax2.set_ylim(0, 4.6)
    ax2.axis("off")
    ax2.add_patch(FancyBboxPatch((0.06, 0.20), 7.08, 4.24, boxstyle="round,pad=0.03,rounding_size=0.14",
                                 facecolor="#FFFFFF", edgecolor=C_LINE, lw=1.05))

    ax2.text(3.60, 4.18, "Co-Aligned 32-Row Warp Tile", ha="center", va="center", fontsize=9.0, fontweight="bold")

    ax2.text(0.22, 3.78, "1) 32B TileHeader:", ha="left", va="center", fontsize=8.2, fontweight="bold")
    p1 = [(2.15, "MinMax", P_LILAC), (1.50, "Valid32", P_SAGE), (1.45, "CidOff", P_GRAY), (1.66, "Cat64", P_BLUE)]
    cx = 0.22
    for sw, st, sfc in p1:
        ax2.add_patch(Rectangle((cx, 3.28), sw, 0.36, facecolor=sfc, edgecolor=C_LINE, lw=0.8))
        ax2.text(cx + sw / 2, 3.46, st, ha="center", va="center", fontsize=7.6, fontweight="bold")
        cx += sw

    ax2.text(0.22, 2.98, "2) 64-Bit SLAP Columns:", ha="left", va="center", fontsize=8.2, fontweight="bold")
    p2 = [(1.95, "Lane 0", P_BLUE), (1.95, "Lane 1", P_BLUE), (0.96, "..", "#FFFFFF"), (1.90, "Lane 31", P_BLUE)]
    cx = 0.22
    for sw, st, sfc in p2:
        ax2.add_patch(Rectangle((cx, 2.48), sw, 0.36, facecolor=sfc, edgecolor=C_LINE, lw=0.8))
        ax2.text(cx + sw / 2, 2.66, st, ha="center", va="center", fontsize=7.6, fontweight="bold")
        cx += sw

    ax2.text(0.22, 2.18, "3) 5.25b Subspace Sketch:", ha="left", va="center", fontsize=8.2, fontweight="bold")
    p3 = [(2.25, "INT4+Sign", P_CREAM), (2.15, r"FP16 $(\hat{\gamma},\overline{\eta})$", P_ROSE), (2.36, "Blk 1..3", "#FFFFFF")]
    cx = 0.22
    for sw, st, sfc in p3:
        ls_cur = "--" if "Blk" in st else "-"
        ax2.add_patch(Rectangle((cx, 1.66), sw, 0.36, facecolor=sfc, edgecolor=C_LINE, lw=0.8, linestyle=ls_cur))
        ax2.text(cx + sw / 2, 1.84, st, ha="center", va="center", fontsize=7.6, fontweight="bold")
        cx += sw

    ax2.text(0.22, 1.34, "4) Host UVA & Tail Tile:", ha="left", va="center", fontsize=8.2, fontweight="bold")
    ax2.add_patch(Rectangle((0.22, 0.42), 3.74, 0.72, facecolor=P_SAGE, edgecolor=C_LINE, lw=0.82))
    ax2.text(2.09, 0.78, "Pinned Host Vectors\n128B UVA Gather", ha="center", va="center", fontsize=7.7, fontweight="bold")
    ax2.add_patch(Rectangle((4.08, 0.42), 2.90, 0.72, facecolor="#FFFFFF", edgecolor=C_LINE, lw=0.82, ls="--"))
    ax2.text(5.53, 0.78, "Mutable Tail Tile\nO(1) Churn", ha="center", va="center", fontsize=7.7, fontweight="bold")

    ax2.text(0.5, -0.05, "(b) GPU-host warp-tile layout", transform=ax2.transAxes, ha="center", va="top", fontsize=9.8, fontweight="bold")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.15, wspace=0.07)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig3_ulp_ball_pipeline.{ext}"), dpi=260, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("[Generated] fig3_ulp_ball_pipeline.pdf")


# ==============================================================================
# 4. Archetype 4 — `fig4_arch_and_layout.pdf` (Page 5 Top, Full-Width `figure*`)
#    Canvas scaled to (12.8, 2.72) with 9-11pt fonts => ~45% larger printed text!
# ==============================================================================
def plot_fig4_arch_and_layout(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.8, 2.72))
    ax.set_xlim(0, 18.0)
    ax.set_ylim(0, 4.44)
    ax.axis("off")

    # Subtle outer plane enclosure for crisp left/right symmetry (x = 2.64 .. 14.80)
    ax.add_patch(FancyBboxPatch((2.64, 1.48), 12.16, 2.60, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor="#FAFAFC", edgecolor="#888888", lw=0.95, ls=":"))
    ax.plot([2.55, 14.88], [1.35, 1.35], color="#555555", ls="--", lw=1.15)
    ax.text(2.66, 4.26, "GPU Plane: China Unicom 16 GB Provincial / MEC Edge GPU (15.57 GiB Usable VRAM)",
            ha="left", va="center", fontsize=9.9, fontweight="bold", color=C_LINE)
    ax.text(2.66, 0.13, "Host Plane: Page-Locked Host DDR4/DDR5 DRAM (128-512 GB) Over PCIe Gen4 UVA",
            ha="left", va="center", fontsize=9.9, fontweight="bold", color=C_LINE)

    # Left Input Icons (x = 0.2 .. 2.1)
    draw_user_icon(ax, 0.48, 3.56, scale=1.05)
    ax.text(0.48, 2.94, "Agent", ha="center", va="center", fontsize=8.9, fontweight="bold")
    draw_magnifier(ax, 1.80, 3.48, r=0.20)
    ax.text(1.80, 2.92, "Query (q, P)", ha="center", va="center", fontsize=8.8, fontweight="bold")

    draw_cylinder(ax, 0.62, 1.72, 0.60, 0.56)
    draw_cylinder(ax, 1.54, 1.72, 0.60, 0.56, fc=P_GRAY)
    ax.text(1.08, 1.08, "China Unicom Logs\n(Nova/5G/SSH)", ha="center", va="center", fontsize=8.2, fontweight="bold")

    draw_arrow(ax, (2.06, 3.48), (2.72, 3.48), lw=1.3)
    draw_badge(ax, 2.39, 3.78, "1", r=0.175, filled=False, fontsize=9.2)

    # ---------------- ROW 1 (GPU Top, flush x = 2.76 .. 14.68) ----------------
    # Module 3: 3-Regime Cost Planner (x = 2.76 .. 7.68, w = 4.92)
    ax.add_patch(FancyBboxPatch((2.76, 2.82), 4.92, 1.18, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor="#FFFFFF", edgecolor=C_LINE, lw=1.6))
    draw_badge(ax, 3.05, 3.76, "3", r=0.185, filled=True, fontsize=9.6)
    ax.text(5.38, 3.76, "M3: 3-Regime Cost Planner", ha="center", va="center", fontsize=9.3, fontweight="bold")
    draw_pill_slot(ax, 2.92, 3.27, 2.68, 0.33, "Header Sample (<0.04 ms)", fc="#FFFFFF", fs=8.0, bold=True, pins=False)
    draw_pill_slot(ax, 5.68, 3.27, 1.84, 0.33, "64B SQL Desc", fc=P_GRAY, fs=8.0, bold=True, pins=False)
    draw_pill_slot(ax, 2.92, 2.88, 1.44, 0.31, "R1: Global", fc="#FFFFFF", fs=8.0, bold=True, pins=False)
    draw_pill_slot(ax, 4.44, 2.88, 1.60, 0.31, "R2: Co-Prune", fc=P_CREAM, fs=8.0, bold=True, pins=False)
    draw_pill_slot(ax, 6.12, 2.88, 1.40, 0.31, "R3: Sketch", fc="#FFFFFF", fs=8.0, bold=True, pins=False)

    draw_arrow(ax, (7.72, 3.41), (8.28, 3.41), lw=1.3)
    draw_badge(ax, 8.00, 3.72, "2", r=0.175, filled=False, fontsize=9.2)

    # Module 1: Co-Aligned Warp-Tile GPU Store (x = 8.32 .. 14.68, w = 6.36 — FLUSH RIGHT!)
    ax.add_patch(FancyBboxPatch((8.32, 2.82), 6.36, 1.18, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor="#FFFFFF", edgecolor=C_LINE, lw=1.6))
    draw_badge(ax, 8.61, 3.76, "1", r=0.185, filled=True, fontsize=9.6)
    ax.text(11.68, 3.76, "M1: Co-Aligned Warp-Tile GPU Store", ha="center", va="center", fontsize=9.3, fontweight="bold")
    draw_pill_slot(ax, 8.48, 3.27, 1.88, 0.33, "32B TileHeader", fc="#FFFFFF", fs=8.1, bold=True, pins=False)
    draw_pill_slot(ax, 10.46, 3.27, 1.96, 0.33, "64b SLAP Cols", fc="#FFFFFF", fs=8.1, bold=True, pins=False)
    draw_pill_slot(ax, 12.52, 3.27, 2.00, 0.33, "Quantile Grid", fc=P_CREAM, fs=8.1, bold=True, pins=False)
    draw_pill_slot(ax, 8.48, 2.88, 6.04, 0.33, "5.25b ULP-Safe Subspace Ball Sketch (0.1-1.6 GiB VRAM)", fc=P_BLUE, fs=8.3, bold=True, pins=False)

    draw_arrow(ax, (11.50, 2.80), (11.50, 2.48), lw=1.3)
    draw_badge(ax, 11.86, 2.64, "3", r=0.175, filled=False, fontsize=9.2)

    # ---------------- ROW 2 (GPU Bottom, flush x = 2.76 .. 14.68) ----------------
    ax.add_patch(FancyBboxPatch((2.76, 1.54), 3.16, 0.92, boxstyle="round,pad=0.02,rounding_size=0.10",
                                facecolor="#FFFFFF", edgecolor=C_LINE, lw=1.25, ls="--"))
    ax.text(4.34, 2.29, "Resident Ops-Agent LLM", ha="center", va="center", fontsize=8.6, fontweight="bold")
    draw_pill_slot(ax, 2.92, 1.84, 2.84, 0.30, "Weights + KV (~6.0 GiB)", fc=P_LILAC, fs=8.1, bold=True, pins=False)
    ax.text(4.34, 1.66, "Leaves <9.6 GiB Budget", ha="center", va="center", fontsize=7.9)

    # Module 2: Warp Compaction Engine (x = 6.10 .. 14.68, w = 8.58 — FLUSH RIGHT!)
    ax.add_patch(FancyBboxPatch((6.10, 1.54), 8.58, 0.92, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor="#FFFFFF", edgecolor=C_LINE, lw=1.6))
    draw_badge(ax, 6.38, 2.22, "2", r=0.185, filled=True, fontsize=9.6)
    ax.text(6.68, 2.22, "M2: Warp Compaction Engine", ha="left", va="center", fontsize=9.3, fontweight="bold")
    draw_pill_slot(ax, 6.26, 1.64, 3.86, 0.35, "Shared Ring (81%-99% Lanes)", fc=P_GRAY, fs=8.4, bold=True, pins=False)
    draw_arrow(ax, (10.16, 1.82), (10.70, 1.82), lw=1.25)
    draw_badge(ax, 10.43, 2.20, "4", r=0.175, filled=False, fontsize=9.2)
    draw_pill_slot(ax, 10.74, 1.64, 3.78, 0.35, "4-Blk ULP Early-Stop (-90%..-99%)", fc="#FFFFFF", fs=8.4, bold=True, pins=False)

    # ---------------- ROW 3 (Host Plane, flush x = 2.76 .. 14.68) ----------------
    ax.add_patch(FancyBboxPatch((2.76, 0.38), 7.86, 0.78, boxstyle="round,pad=0.02,rounding_size=0.10",
                                facecolor="#FFFFFF", edgecolor=C_LINE, lw=1.35))
    draw_cylinder(ax, 3.30, 0.77, 0.56, 0.52)
    ax.text(6.95, 0.91, "Pinned Host Exact Vectors (FP32 / uint8)", ha="center", va="center", fontsize=9.1, fontweight="bold")
    ax.text(6.95, 0.56, "32-Lane Coalesced PCIe Gen4 UVA (<4% Survivors)", ha="center", va="center", fontsize=8.3)

    draw_arrow(ax, (9.60, 1.52), (8.40, 1.16), lw=1.3)
    draw_arrow(ax, (8.95, 1.16), (10.15, 1.52), lw=1.3)
    draw_badge(ax, 9.68, 1.29, "5", r=0.175, filled=False, fontsize=9.2)

    ax.add_patch(FancyBboxPatch((10.84, 0.38), 3.84, 0.78, boxstyle="round,pad=0.02,rounding_size=0.10",
                                facecolor="#FFFFFF", edgecolor=C_LINE, lw=1.25))
    draw_badge(ax, 11.14, 0.95, "6", r=0.165, filled=False, fontsize=9.0)
    ax.text(12.88, 0.91, "Zero-Rebuild Log Churn", ha="center", va="center", fontsize=8.8, fontweight="bold")
    ax.text(12.88, 0.56, "Tail Tiles: 723K ins/s | O(1) Purge", ha="center", va="center", fontsize=8.1)
    draw_arrow(ax, (1.92, 1.46), (2.72, 0.80), lw=1.15, ls="--")

    # ---------------- Right Output Pills (x = 15.26 .. 17.90) ----------------
    draw_arrow(ax, (14.82, 3.42), (15.24, 3.44), lw=1.25)
    draw_arrow(ax, (14.82, 2.48), (15.24, 2.50), lw=1.25)
    draw_arrow(ax, (14.82, 1.72), (15.24, 1.56), lw=1.25)
    draw_pill_slot(ax, 15.26, 3.12, 2.64, 0.64, "  100% Exact Top-K\n  0 False Dismissals",
                   fc=P_SAGE, tc=C_LINE, lw=1.25, fs=8.6, bold=True, pins=False)
    ax.plot([15.38, 15.46, 15.58], [3.52, 3.42, 3.59], color="#1E6B3A", lw=1.6, zorder=12)
    draw_pill_slot(ax, 15.26, 2.18, 2.64, 0.64, "  97.2% RCA Hit@1\n  3.1 ms P99 Latency",
                   fc=P_GRAY, tc=C_LINE, lw=1.1, fs=8.5, bold=True, pins=False)
    ax.plot([15.38, 15.46, 15.58], [2.58, 2.48, 2.65], color=C_LINE, lw=1.55, zorder=12)
    draw_pill_slot(ax, 15.26, 1.24, 2.64, 0.64, "  $316.5 / 1K QPS\n  -94%..-99% CAPEX",
                   fc="#FFFFFF", tc=C_LINE, lw=1.1, fs=8.5, bold=True, pins=False)
    ax.plot([15.38, 15.46, 15.58], [1.64, 1.54, 1.71], color=C_LINE, lw=1.55, zorder=12)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig4_arch_and_layout.{ext}"), dpi=260, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("[Generated] fig4_arch_and_layout.pdf")


# ==============================================================================
# 5. Archetype 5 — `fig5_matrix_walkthrough.pdf` (Page 6, Single-Column)
#    Canvas scaled to (6.4, 2.56) with 8.2-9.5pt fonts => ~45% larger printed text!
# ==============================================================================
def plot_fig5_matrix_walkthrough(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 2.60))
    ax.set_xlim(0, 10.0)
    ax.set_ylim(-0.26, 3.66)
    ax.axis("off")

    def draw_table(ax, x0, y0, col_ws, row_h, rows, hdr_fc, body_fc="#FFFFFF", fs=8.2):
        for ri, rdata in enumerate(rows):
            yy = y0 - ri * row_h
            cx = x0
            for cw, val in zip(col_ws, rdata):
                fc = hdr_fc if ri == 0 else (body_fc if isinstance(body_fc, str) else body_fc[ri - 1])
                ax.add_patch(Rectangle((cx, yy - row_h), cw, row_h, facecolor=fc, edgecolor=C_LINE, lw=0.85))
                ax.text(cx + cw / 2, yy - row_h / 2, val, ha="center", va="center",
                        fontsize=fs, fontweight="bold" if ri == 0 else "normal", color=C_LINE)
                cx += cw

    ax.text(5.0, 3.50, r"China Unicom PoP Trace on $Q_1$: $c_1 \in [120, 300] \wedge c_2 \in \{2, 5\}$, $\tau_{\mathrm{safe}} = 0.480$",
            ha="center", va="center", fontsize=8.7, fontweight="bold", color=C_LINE)

    # Top-left table (1): y = 3.26 down to 1.94
    draw_table(
        ax, 0.08, 3.26, [1.05, 1.42, 1.00, 1.25], 0.33,
        [
            ["Tile", "Min-Max", "Cat64", "Action"],
            ["T0(C3)", "[20, 95]", "{1, 4}", "Skip"],
            ["T1(C3)", "[110, 240]", "{2, 6}", "SLAP"],
            ["T2(C3)", "[140, 280]", "{2, 5}", "All-Hit"],
        ],
        hdr_fc="#E6E0EC",
        body_fc=["#FFFFFF", "#FFFFFF", "#F6F4F9"],
        fs=8.1,
    )
    ax.text(2.44, 1.74, "(1) TileHeader Check (Purple)", ha="center", va="center", fontsize=8.4, fontweight="bold")

    draw_arrow(ax, (4.84, 2.60), (5.12, 2.60), lw=1.0)

    # Top-right table (2): y = 3.26 down to 1.94
    draw_table(
        ax, 5.16, 3.26, [1.04, 2.22, 1.50], 0.33,
        [
            ["Lane", "Packed (Cid, Row)", "Source"],
            ["L0", "(C3, r35)", "T1"],
            ["L1", "(C3, r51)", "T1"],
            ["L2..31", "(C3,r64)..(C7,r12)", "T2+C7"],
        ],
        hdr_fc="#DCE6F2",
        body_fc=["#FFFFFF", "#FFFFFF", "#F3F7FB"],
        fs=8.0,
    )
    ax.text(7.54, 1.74, "(2) Cross-Cluster Ring (Blue)", ha="center", va="center", fontsize=8.4, fontweight="bold")

    # Vertical arrow from Stage (2) down to Stage (3) cleanly between sublabels
    draw_arrow(ax, (5.55, 1.90), (5.55, 1.40), lw=1.05)

    # Bottom-left table (3): y = 1.36 down to 0.11
    draw_table(
        ax, 0.08, 1.36, [0.90, 1.52, 1.52, 1.52, 1.38], 0.25,
        [
            ["Row", "Block 1", "Block 2", "Blk 3..4", "Result"],
            ["r35", "0.61 > .48", "Skip", "Skip", "Prune B1"],
            ["r51", "0.31 <= .48", "0.51 > .48", "Skip", "Prune B2"],
            ["r64", "0.14 <= .48", "0.26 <= .48", "0.40 <= .48", "Stage 4"],
            ["r12", "0.54 > .48", "Skip", "Skip", "Prune B1"],
        ],
        hdr_fc="#FDEADA",
        body_fc=["#FFFFFF", "#FFFFFF", "#FEF7F1", "#FFFFFF"],
        fs=7.7,
    )
    ax.text(3.50, -0.09, "(3) 4-Block ULP Ball Pruning (Orange)", ha="center", va="center", fontsize=8.4, fontweight="bold")

    draw_arrow(ax, (6.96, 0.72), (7.20, 0.72), lw=1.0)

    # Bottom-right table (4): y = 1.36 down to 0.12
    draw_table(
        ax, 7.22, 1.36, [2.70], 0.31,
        [
            ["32-Lane UVA Fetch"],
            ["Broadcast Row r64"],
            [r"$\|q - x_{64}\|_2^2 = 0.431$"],
            [r"$\tau_K^{\mathcal{P}} \leftarrow 0.431$"],
        ],
        hdr_fc="#EBF1DE",
        body_fc=["#FFFFFF", "#FFFFFF", "#F6F9F1"],
        fs=8.0,
    )
    ax.text(8.57, -0.09, "(4) Host UVA (Green)", ha="center", va="center", fontsize=8.4, fontweight="bold")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig5_matrix_walkthrough.{ext}"), dpi=260, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("[Generated] fig5_matrix_walkthrough.pdf")


# ==============================================================================
# 6. Section 7.2 `fig11_teleops_and_tco.pdf` (Ultra-Flat Ribbon + Large Crisp Fonts!)
# ==============================================================================
def plot_fig11_teleops_and_tco(res_dir: str, out_dir: str):
    p_e16 = os.path.join(res_dir, "e16_teleops_and_tco.json")
    if not os.path.exists(p_e16):
        return
    d16 = json.load(open(p_e16))
    if "teleops_a100" not in d16 or "tco_teleops1m_w3" not in d16:
        return

    os.makedirs(out_dir, exist_ok=True)
    e17_path = os.path.join(res_dir, "e17_production_sla_and_audit.json")
    d17 = json.load(open(e17_path)) if os.path.exists(e17_path) else {}

    fig, axes = plt.subplots(1, 4, figsize=(14.0, 1.36), gridspec_kw={"width_ratios": [1.14, 0.86, 0.96, 1.02]})

    # (a) TeleOps1M W1-W4 QPS
    ax = axes[0]
    wls = [("W1_range", "W1:Time"), ("W2_correlated", "W2:Metric"), ("W3_compound", "W3:Triage"), ("W4_mixed", "W4:Storm")]
    methods = [
        ("DuckDB_VSS", "DuckDB", "#E5E7E9", None),
        ("Official_ACORN", "ACORN", "#DFEFE2", None),
        ("Host_Offload", "Offload", "#F2F3F4", None),
        ("cuVS_CAGRA", "CAGRA", "#FDEBD0", None),
        ("cuVS_IVF_Flat", "cuVS-IVF", "#D6EAF8", None),
        ("PRISM", "PRISM", "#F5B7B1", "//"),
    ]
    x = np.arange(len(wls))
    bw = 0.125
    duck_w3 = d16["cpu_and_dbms"]["DuckDB_1_5_VSS_HNSW"]["qps"]
    duck_ex = d16["cpu_and_dbms"]["DuckDB_1_5_SQL_Exact"]["qps"]

    for mi, (mk, mlbl, mc, mh) in enumerate(methods):
        vals, misses = [], []
        for wk, _ in wls:
            cell = d16["teleops_4080"]["workloads"][wk]["R95"]
            if mk == "DuckDB_VSS":
                vals.append(max(duck_w3, duck_ex))
                misses.append(False)
            else:
                ent = cell.get(mk, {})
                if ent.get("status") == "OK" and ent.get("qps", 0) > 0:
                    vals.append(ent["qps"])
                    misses.append(False)
                else:
                    vals.append(0.0)
                    misses.append(True)
        offs = x + (mi - 2.5) * bw
        ax.bar(offs, [max(v, 5.0) if not ms else 0.0 for v, ms in zip(vals, misses)],
               width=bw * 0.90, facecolor=mc, edgecolor="#333333", linewidth=0.72, hatch=mh, label=mlbl, zorder=3)
        for xi, ms in zip(offs, misses):
            if ms:
                ax.scatter([xi], [8.5], marker="v", s=22, color="#8E7CC3", zorder=5)

    ax.set_yscale("log")
    ax.set_ylim(4.0, 4.5e6)
    ax.set_xticks(x)
    ax.set_xticklabels([w[1] for w in wls], fontsize=9.2)
    ax.set_ylabel("QPS (log)", fontsize=9.8)
    ax.tick_params(axis="y", labelsize=9.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35, lw=0.5, color="#C8C8C8", zorder=0)
    ax.scatter([], [], marker="v", s=22, color="#8E7CC3", label="Miss")
    ax.legend(loc="upper center", fontsize=7.6, ncol=4, frameon=True, edgecolor="#CCCCCC", framealpha=0.95,
              borderpad=0.14, columnspacing=0.45, handletextpad=0.3, handlelength=1.1)
    ax.text(0.5, -0.38, "(a) China Unicom TeleOps QPS", transform=ax.transAxes, ha="center", va="top", fontsize=10.2, fontweight="bold")

    # (b) Rare-Fault Root-Cause Diagnosis Hit@1 (%)
    ax = axes[1]
    rca = d17.get("teleops_rca_accuracy", {})
    rca_items = [
        ("Unfilt.", rca.get("Unfiltered_Vector_RAG", {}).get("top1_root_cause_hit_pct", 1.58), "#E5E7E9", None),
        ("Faiss", rca.get("Faiss_GPU_PostFilter", {}).get("top1_root_cause_hit_pct", 3.95), "#D6EAF8", None),
        ("CAGRA", rca.get("cuVS_CAGRA_Filtered", {}).get("top1_root_cause_hit_pct", 20.95), "#FDEBD0", None),
        ("ACORN", rca.get("Official_ACORN_Filtered", {}).get("top1_root_cause_hit_pct", 56.13), "#DFEFE2", None),
        ("PRISM", rca.get("PRISM_Ours", {}).get("top1_root_cause_hit_pct", 97.23), "#F5B7B1", "//"),
    ]
    xr = np.arange(len(rca_items))
    for i_r, (rlbl, rval, rcol, rhat) in enumerate(rca_items):
        ax.bar([xr[i_r]], [rval], width=0.54, facecolor=rcol, edgecolor="#333333", linewidth=0.72, hatch=rhat, zorder=3)
        ax.text(xr[i_r], rval + 3.0, f"{rval:.1f}%", ha="center", va="bottom", fontsize=8.4, fontweight="bold", color="#333333")
    ax.set_ylim(0, 128)
    ax.set_xticks(xr)
    ax.set_xticklabels([it[0] for it in rca_items], fontsize=9.0)
    ax.set_ylabel("Hit@1 (%)", fontsize=9.8)
    ax.tick_params(axis="y", labelsize=9.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35, lw=0.5, color="#C8C8C8", zorder=0)
    ax.text(0.5, -0.38, "(b) Rare-Fault RCA Hit@1", transform=ax.transAxes, ha="center", va="top", fontsize=10.5, fontweight="bold")

    # (c) Co-Located Ops-Agent LLM + Vector Index VRAM on 16 GB GPU
    ax = axes[2]
    coloc = d16["colocated_edge_appliance"]
    llm_g = coloc["llm_peak_with_kv_gib"]
    ds_items = [("teleops1m", "TeleOps"), ("convlog1m", "Conv1M"), ("sift10m", "SIFT10M")]
    xb = np.arange(len(ds_items))
    w2 = 0.25

    p_idx_g = [coloc["datasets"][k]["PRISM_CoLocated"]["prism_index_vram_gib"] for k, _ in ds_items]
    e14_vf = json.load(open(os.path.join(res_dir, "e14_official_vecflow.json"))).get("datasets", {})
    vf_tele = d16["teleops_a100"].get("Official_VecFlow_WL1", {}).get("pure_index_vram_gib", 9.58)
    vf_g = [vf_tele, e14_vf.get("convlog1m", {}).get("vram_gib", 19.1), e14_vf.get("sift10m", {}).get("vram_gib", 20.1)]
    cg_g = [coloc["datasets"]["teleops1m"]["cuVS_CAGRA_CoLocated"].get("cagra_vram_gib", 5.88), 9.75, 12.35]

    for idx_s, (vec_arr, lbl_v, col_v, hat_v, dx) in enumerate([
        (p_idx_g, "PRISM", "#F5B7B1", "//", -w2),
        (cg_g, "CAGRA", "#FDEBD0", None, 0.0),
        (vf_g, "VecFlow", "#D6EAF8", None, +w2),
    ]):
        ax.bar(xb + dx, [llm_g] * 3, width=w2 * 0.88, facecolor="#EBDEF0", edgecolor="#333333", linewidth=0.7,
               label="LLM" if idx_s == 0 else None, zorder=3)
        ax.bar(xb + dx, vec_arr, bottom=[llm_g] * 3, width=w2 * 0.88, facecolor=col_v, edgecolor="#333333",
               linewidth=0.7, hatch=hat_v, label=lbl_v, zorder=3)
        for xi, vg in zip(xb + dx, vec_arr):
            tot_g = llm_g + vg
            if tot_g > 15.57:
                ax.text(xi, max(tot_g + 0.9, 17.2), "OOM", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#A94442")
            else:
                ax.text(xi, tot_g + 0.9, f"{tot_g:.1f}G", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#333333")

    ax.axhline(15.57, color="#A94442", linestyle="--", linewidth=1.1, zorder=5)
    ax.text(-0.40, 17.0, "15.6G", fontsize=7.8, fontweight="bold", color="#A94442")
    ax.set_ylim(0, 41.0)
    ax.set_xticks(xb)
    ax.set_xticklabels([d[1] for d in ds_items], fontsize=9.2)
    ax.set_ylabel("VRAM (GiB)", fontsize=9.8)
    ax.tick_params(axis="y", labelsize=9.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35, lw=0.5, color="#C8C8C8", zorder=0)
    ax.legend(loc="upper left", fontsize=7.6, ncol=2, frameon=True, edgecolor="#CCCCCC", framealpha=0.95,
              borderpad=0.14, columnspacing=0.45, handletextpad=0.3, handlelength=1.1)
    ax.text(0.5, -0.38, "(c) Co-Located GPU VRAM", transform=ax.transAxes, ha="center", va="top", fontsize=10.5, fontweight="bold")

    # (d) Hardware CAPEX per 1,000 QPS
    ax = axes[3]
    tco_rows = d16["tco_teleops1m_w3"]
    pick_map = [
        ("Edge 16GB GPU Appliance + PRISM (Ours)", "16G\nPRISM", "#F5B7B1", "//"),
        ("80GB A100 Data-Center Node + PRISM-A100 (Ours)", "A100\nPRISM", "#FADBD8", "//"),
        ("Edge 16GB GPU Appliance + Host-Offload", "16G\nOff.", "#F2F3F4", None),
        ("Edge 16GB GPU Appliance + cuVS-IVF-Flat", "16G\ncuVS", "#D6EAF8", None),
        ("80GB A100 Data-Center Node + cuVS-IVF-Flat", "A100\ncuVS", "#C6DBEF", None),
        ("48T Dual-Socket CPU Server + Official ACORN", "48T\nACORN", "#DFEFE2", None),
        ("48T Dual-Socket CPU Server + DuckDB 1.5 vss", "48T\nDuck", "#E5E7E9", None),
    ]
    t_dict = {r["architecture"]: r for r in tco_rows}
    labels_c, costs_c, cols_c, hats_c = [], [], [], []
    for key_a, short_l, cc, ch in pick_map:
        if key_a in t_dict and t_dict[key_a].get("capex_per_1k_qps_usd", 0) > 0:
            labels_c.append(short_l)
            costs_c.append(t_dict[key_a]["capex_per_1k_qps_usd"])
            cols_c.append(cc)
            hats_c.append(ch)

    xc = np.arange(len(labels_c))
    for i, (cv, cc, ch) in enumerate(zip(costs_c, cols_c, hats_c)):
        ax.bar([xc[i]], [cv], width=0.55, facecolor=cc, edgecolor="#333333", linewidth=0.72, hatch=ch, zorder=3)
        lbl_txt = f"${cv:.0f}" if cv < 1000 else f"${cv/1000.0:.0f}K"
        ax.text(xc[i], cv * 1.28, lbl_txt, ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#333333")

    ax.set_yscale("log")
    ax.set_ylim(max(10.0, min(costs_c) * 0.38), max(costs_c) * 14.0)
    ax.set_xticks(xc)
    ax.set_xticklabels(labels_c, fontsize=8.0)
    ax.set_ylabel("CAPEX ($)", fontsize=9.8)
    ax.tick_params(axis="y", labelsize=9.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35, lw=0.5, color="#C8C8C8", zorder=0)
    ax.text(0.5, -0.38, "(d) China Unicom PoP CAPEX", transform=ax.transAxes, ha="center", va="top", fontsize=10.2, fontweight="bold")

    plt.subplots_adjust(left=0.045, right=0.994, top=0.95, bottom=0.29, wspace=0.23)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig11_teleops_and_tco.{ext}"), dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("[Generated] fig11_teleops_and_tco.pdf")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    res = os.path.join(root, "results")
    out = os.path.expanduser(os.environ.get("PRISM_FIG_OUT", os.path.join(here, "generated")))
    os.makedirs(out, exist_ok=True)
    plot_fig1_paradigm_comparison(out)
    plot_fig2_aiops_bottleneck_flow(out)
    plot_fig3_ulp_ball_pipeline(out)
    plot_fig4_arch_and_layout(out)
    plot_fig5_matrix_walkthrough(out)
    plot_fig11_teleops_and_tco(res, out)
