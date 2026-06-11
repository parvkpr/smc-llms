#!/usr/bin/env python3
"""Publication figure: TAP regular vs BCA deploy-set overlap (pooled seeds 42--45)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results/judge_swap/multiseed/tap_complementarity_data.json"
OUT_DIR = ROOT / "gpu_llmchecker/figures"
OUT_PDF = OUT_DIR / "tap_complementarity_overlap.pdf"
OUT_PNG = OUT_DIR / "tap_complementarity_overlap.png"

COLORS = {
    "both": "#15803d",
    "regular_only": "#2563eb",
    "bca_only": "#9333ea",
    "neither": "#94a3b8",
    "regular_edge": "#1d4ed8",
    "bca_edge": "#7e22ce",
}


def load_summary() -> dict:
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    for ds in payload["datasets"]:
        if ds["id"] == "pooled_42_45":
            return ds["summary"]
    raise SystemExit("pooled_42_45 dataset not found")


def draw_figure(summary: dict) -> None:
    both = summary["both"]
    reg_only = summary["regular_only"]
    bca_only = summary["bca_only"]
    neither = summary["neither"]
    union = summary["union"]
    n = both + reg_only + bca_only + neither

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 10,
        }
    )

    fig = plt.figure(figsize=(7.2, 3.4), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.08)
    ax_venn = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    # ── Left: overlapping deploy sets ──
    ax_venn.set_aspect("equal")
    ax_venn.set_xlim(-2.35, 2.35)
    ax_venn.set_ylim(-2.55, 2.05)
    ax_venn.axis("off")

    r = 1.15
    cx_reg, cx_bca = -0.42, 0.42
    cy = 0.15

    reg_circle = Circle(
        (cx_reg, cy),
        r,
        facecolor=COLORS["regular_only"],
        edgecolor=COLORS["regular_edge"],
        alpha=0.28,
        linewidth=2.0,
        zorder=1,
    )
    bca_circle = Circle(
        (cx_bca, cy),
        r,
        facecolor=COLORS["bca_only"],
        edgecolor=COLORS["bca_edge"],
        alpha=0.28,
        linewidth=2.0,
        zorder=1,
    )
    ax_venn.add_patch(reg_circle)
    ax_venn.add_patch(bca_circle)

    overlap = Circle(
        (0.0, cy),
        0.72,
        facecolor=COLORS["both"],
        edgecolor=COLORS["both"],
        alpha=0.72,
        linewidth=1.8,
        zorder=2,
    )
    ax_venn.add_patch(overlap)

    ax_venn.text(cx_reg - 0.55, cy, f"{reg_only}\nregular\nonly", ha="center", va="center", fontsize=11, fontweight="bold")
    ax_venn.text(0.0, cy, f"{both}\nboth", ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    ax_venn.text(cx_bca + 0.55, cy, f"{bca_only}\nBCA\nonly", ha="center", va="center", fontsize=11, fontweight="bold")

    ax_venn.text(cx_reg, cy + r + 0.28, "Regular TAP deploy", ha="center", fontsize=10.5, fontweight="bold")
    ax_venn.text(cx_bca, cy + r + 0.28, "BCA TAP deploy\n(judge pick)", ha="center", fontsize=10.5, fontweight="bold")

    neither_box = FancyBboxPatch(
        (-1.05, -2.05),
        2.1,
        0.62,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        facecolor="#f8fafc",
        edgecolor=COLORS["neither"],
        linewidth=1.5,
        zorder=3,
    )
    ax_venn.add_patch(neither_box)
    ax_venn.text(
        0.0,
        -1.74,
        f"{neither} behaviors: neither method deploys a jailbreak",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#334155",
    )

    ax_venn.text(
        0.0,
        -2.45,
        f"Union = {union}/{n} ({100 * union / n:.1f}%)   ·   best single method = {max(both + reg_only, both + bca_only)}/{n}",
        ha="center",
        fontsize=9,
        color="#475569",
    )

    # ── Right: outcome mosaic + coverage bars ──
    ax_bar.set_xlim(0, 10)
    ax_bar.set_ylim(0, 10)
    ax_bar.axis("off")
    ax_bar.text(0, 9.55, "Deploy outcomes (102 paired runs)", fontsize=10.5, fontweight="bold")

    segments = [
        ("both", both, "Both succeed"),
        ("regular_only", reg_only, "Regular only"),
        ("bca_only", bca_only, "BCA only"),
        ("neither", neither, "Neither"),
    ]
    x0, y0, bar_h = 0.2, 7.35, 1.05
    total_w = 9.6
    x = x0
    for key, count, label in segments:
        w = total_w * count / n
        ax_bar.add_patch(
            Rectangle(
                (x, y0),
                w,
                bar_h,
                facecolor=COLORS[key],
                edgecolor="white",
                linewidth=1.2,
            )
        )
        if w > 0.9:
            ax_bar.text(
                x + w / 2,
                y0 + bar_h / 2,
                f"{count}\n({100 * count / n:.1f}%)",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white" if key != "neither" else "#1e293b",
            )
        x += w

    legend_y = 6.55
    for i, (key, count, label) in enumerate(segments):
        lx = 0.2 + (i % 2) * 4.8
        ly = legend_y - (i // 2) * 0.42
        ax_bar.add_patch(Rectangle((lx, ly - 0.12), 0.28, 0.28, facecolor=COLORS[key], clip_on=False))
        ax_bar.text(lx + 0.42, ly, f"{label} ({count})", va="center", fontsize=8.5, color="#334155")

    # Coverage comparison bars
    reg_cov = both + reg_only
    bca_cov = both + bca_only
    bar_labels = [
        ("Regular alone", reg_cov, COLORS["regular_only"]),
        ("BCA alone", bca_cov, COLORS["bca_only"]),
        ("Union (either)", union, COLORS["both"]),
    ]
    for i, (lbl, val, color) in enumerate(bar_labels):
        y = 3.55 - i * 1.05
        ax_bar.text(0, y + 0.35, lbl, fontsize=9, color="#334155")
        ax_bar.add_patch(Rectangle((0, y - 0.08), 9.6 * val / n, 0.42, facecolor=color, alpha=0.85))
        ax_bar.text(9.65, y + 0.08, f"{val}/{n} ({100 * val / n:.1f}%)", ha="left", va="center", fontsize=8.5)

    ax_bar.text(
        0,
        0.35,
        "Seeds 42–45 · L=8 · Llama-3.1-8B target · BCA trees use judge-pick deploy",
        fontsize=8,
        color="#64748b",
    )

    fig.suptitle(
        "TAP deploy-set complementarity: judge-guided vs BCA-guided search",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.08, dpi=200)
    plt.close(fig)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    draw_figure(load_summary())
