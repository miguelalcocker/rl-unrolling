"""
Visualize K_2 Sweep Results — w Filter Order in Architecture 2
==============================================================

Genera una gráfica por configuración mostrando el Policy Optimality Gap (joint)
en función del orden del filtro w (K_2 + 1).

DATA:
  k2_sweep_results/all_experiments_results.csv   (generado por k2_sweep_experiments.py)

OUTPUTS (en k2_sweep_results/visualizations/):
  k2_sweep_unrolls{U}_K{K}_pog_joint.png
  k2_sweep_unrolls{U}_K{K}_pog_both_norms.png

Usage:
    python experiments/visualize_k2_sweep.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.plots import ARCH_COLORS

RESULTS_DIR = Path("k2_sweep_results")
VIZ_DIR = RESULTS_DIR / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

C_ARCH1 = ARCH_COLORS[1]
C_ARCH2 = ARCH_COLORS[2]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 200,
})

CONFIGS = [
    {"num_unrolls": 6, "K": 5},
    {"num_unrolls": 5, "K": 5},
]


def load_results():
    csv_path = RESULTS_DIR / "all_experiments_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run k2_sweep_experiments.py first."
        )
    df = pd.read_csv(csv_path)
    return df[df["success"] == True].copy()


def _log_yaxis(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())


def _plot_arch_band(ax, df_arch, x_col, y_col, color, label, marker="o", ls="-"):
    """Median + 25th–75th percentile band for one architecture."""
    grouped = df_arch.groupby(x_col)[y_col]
    med = grouped.median()
    p25 = grouped.quantile(0.25)
    p75 = grouped.quantile(0.75)
    x = med.index.values

    ax.fill_between(x, p25.values, p75.values, alpha=0.15, color=color)
    ax.plot(x, med.values, color=color, lw=1.8, ls=ls,
            marker=marker, markersize=5, label=label, zorder=3)


def _add_arch1_hline(ax, pog_values, color, label, split=False):
    """Arch 1: horizontal line at its median POG (single w_order value)."""
    med = float(np.median(pog_values))
    if split:
        ax.axhline(med, color=color, lw=1.8, ls="--", alpha=0.85,
                   label=f"{label} (med={med:.4f})", zorder=2)
    else:
        ax.axhline(med, color=color, lw=1.8, ls="--", alpha=0.85, label=label, zorder=2)
    return med


def plot_k2_sweep(df, config):
    """Single-panel: joint POG train+test vs. w_order for one config."""
    K = config["K"]
    U = config["num_unrolls"]

    df_cfg = df[(df["K"] == K) & (df["num_unrolls"] == U)]
    df_a1 = df_cfg[df_cfg["architecture_type"] == 1]
    df_a2 = df_cfg[df_cfg["architecture_type"] == 2]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Arch 1 — dashed horizontal line
    if len(df_a1) > 0:
        _add_arch1_hline(ax, df_a1["policy_optimality_gap_joint_train"], C_ARCH1, "Arch 1 (train)")
        _add_arch1_hline(ax, df_a1["policy_optimality_gap_joint_test"],  C_ARCH1, "Arch 1 (test)")

    # Arch 2 — median + band vs w_order
    if len(df_a2) > 0:
        _plot_arch_band(ax, df_a2, "w_order", "policy_optimality_gap_joint_train",
                        C_ARCH2, "Arch 2 (train)", ls="-")
        _plot_arch_band(ax, df_a2, "w_order", "policy_optimality_gap_joint_test",
                        C_ARCH2, "Arch 2 (test)", ls="--")

    _log_yaxis(ax)
    ax.set_xlabel(r"Orden filtro $w$ ($K_2 + 1$)", fontsize=12)
    ax.set_ylabel("Policy Optimality Gap (joint, log)", fontsize=12)
    ax.set_title(f"K_2 Sweep — $K={K}$, $U={U}$ — POG joint", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.92)
    ax.grid(True, which="both", alpha=0.18)
    fig.tight_layout()

    fname = VIZ_DIR / f"k2_sweep_unrolls{U}_K{K}_pog_joint.png"
    fig.savefig(fname, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  ✓ {fname}")


def plot_k2_sweep_both_norms(df, config):
    """Two-panel: joint POG (left) and separate POG (right) vs. w_order."""
    K = config["K"]
    U = config["num_unrolls"]

    df_cfg = df[(df["K"] == K) & (df["num_unrolls"] == U)]
    df_a1 = df_cfg[df_cfg["architecture_type"] == 1]
    df_a2 = df_cfg[df_cfg["architecture_type"] == 2]

    metric_pairs = [
        ("policy_optimality_gap_joint_train",    "policy_optimality_gap_joint_test",
         "POG joint (log)"),
        ("policy_optimality_gap_separate_train", "policy_optimality_gap_separate_test",
         "POG separate (log)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, (train_col, test_col, ylabel) in zip(axes, metric_pairs):
        if len(df_a1) > 0:
            ax.axhline(float(df_a1[train_col].median()), color=C_ARCH1, lw=1.8,
                       ls="--", alpha=0.85, label="Arch 1 (train)", zorder=2)
            ax.axhline(float(df_a1[test_col].median()), color=C_ARCH1, lw=1.4,
                       ls=":", alpha=0.85, label="Arch 1 (test)", zorder=2)

        if len(df_a2) > 0:
            _plot_arch_band(ax, df_a2, "w_order", train_col,
                            C_ARCH2, "Arch 2 (train)", ls="-")
            _plot_arch_band(ax, df_a2, "w_order", test_col,
                            C_ARCH2, "Arch 2 (test)", ls="--")

        _log_yaxis(ax)
        ax.set_xlabel(r"Orden filtro $w$ ($K_2 + 1$)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=9, framealpha=0.92)
        ax.grid(True, which="both", alpha=0.18)

    axes[0].set_title(f"POG joint — $K={K}$, $U={U}$", fontsize=12)
    axes[1].set_title(f"POG separate — $K={K}$, $U={U}$", fontsize=12)
    fig.suptitle(
        f"K_2 Sweep: Impacto del orden del filtro $w$ — $K={K}$, $U={U}$",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    fname = VIZ_DIR / f"k2_sweep_unrolls{U}_K{K}_pog_both_norms.png"
    fig.savefig(fname, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  ✓ {fname}")


def main():
    print("Loading k2_sweep results...")
    df = load_results()
    print(f"  {len(df)} successful runs loaded.")

    for config in CONFIGS:
        K = config["K"]
        U = config["num_unrolls"]
        print(f"\nConfig: unrolls={U}, K={K}")
        df_cfg = df[(df["K"] == K) & (df["num_unrolls"] == U)]
        if len(df_cfg) == 0:
            print(f"  ⚠ No data for this config — skipping.")
            continue
        plot_k2_sweep(df, config)
        plot_k2_sweep_both_norms(df, config)

    print(f"\nAll figures saved to {VIZ_DIR}/")


if __name__ == "__main__":
    main()
