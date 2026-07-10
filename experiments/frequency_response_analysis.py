"""
Frequency Response Analysis — Transfer Operator Comparison
===========================================================

Generates spectral analysis figures for TFG Chapter 4.

Uses pre-saved coefficients from freq_analysis_results/ (no retraining needed).
The main TFG figure is T_comparison.png.

CONFIGS:
  {"K": 5,  "num_unrolls": 4}
  {"K": 10, "num_unrolls": 10}

DATA:
  freq_analysis_results/policy_arch{1,2}_K{K}_unrolls{U}_run*.npz

OUTPUTS:
  freq_analysis_results/T_comparison.png   ← TFG figure
  freq_analysis_results/T_composition.png

Usage:
    python experiments/frequency_response_analysis.py
"""

import sys
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environments import CliffWalkingEnv
from src.plots import ARCH_COLORS


RESULTS_DIR = Path("freq_analysis_results")
GAMMA = 0.99

C_IDEAL = "#555555"
C_PRACT = "#009960"
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
    {"K": 5, "num_unrolls": 4},
    {"K": 10, "num_unrolls": 10},
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def get_eigenvalues():
    """Absolute eigenvalues of P_π under uniform policy, sorted descending."""
    env = CliffWalkingEnv()
    nS, nA = env.nS, env.nA
    P = env.P.clone()
    Pi = torch.ones(nS, nA) / nA
    Pi_ext = torch.zeros(nS, nS * nA)
    for s in range(nS):
        for a in range(nA):
            Pi_ext[s, s * nA + a] = Pi[s, a]
    eigs = np.linalg.eigvals((P @ Pi_ext).numpy())
    return np.sort(np.abs(eigs))[::-1]


def load_coeffs(K, U, arch):
    """Load h_coeffs and K_2 from representative saved npz file."""
    candidates = [
        p for p in RESULTS_DIR.glob(f"policy_arch{arch}_K{K}_unrolls{U}_run*.npz")
        if "_temp" not in p.name and "_train" not in p.name
    ]
    if not candidates:
        return None, None
    data = np.load(candidates[0])
    K_2 = int(data["K_2"])
    return data["h_coeffs"], (None if K_2 < 0 else K_2)


def build_T_aprendida_coeffs(h, K, K_2_int):
    """
    Return polynomial coefficients [c_0, ..., c_{K+1}] of
    T_aprendida(λ) = H_r(λ) + G_q(λ).

    K_2_int < 0 → Arch 1 (h already holds all K+2 coefficients).
    K_2_int ≥ 0 → Arch 2 (separate H_r and G_q parts).
    """
    if K_2_int < 0:
        return h.copy()
    H_r = h[:K + 1]
    k_start = K - K_2_int + 1
    G_q = np.zeros(K + 2)
    for i, wi in enumerate(h[K + 1:]):
        G_q[k_start + i] += wi
    coeffs = np.zeros(K + 2)
    coeffs[:K + 1] += H_r
    coeffs[:len(G_q)] += G_q
    return coeffs


def get_Hr_Gq(h, K, K_2_int):
    """Return (H_r_coeffs, G_q_coeffs) as complex arrays."""
    H_r = h[:K + 1].astype(complex)
    if K_2_int < 0:
        G_q = np.zeros(K + 2, dtype=complex)
        G_q[K + 1] = h[K + 1]
    else:
        k_start = K - K_2_int + 1
        G_q = np.zeros(k_start + K_2_int + 1, dtype=complex)
        for i, wi in enumerate(h[K + 1:]):
            G_q[k_start + i] += wi
    return H_r, G_q


def eval_poly(coeffs, lam):
    return sum(c * lam**t for t, c in enumerate(coeffs))


def T_ideal(lam, gamma=GAMMA):
    denom = 1.0 - gamma * lam
    return np.where(np.abs(denom) > 1e-9, 1.0 / denom, np.nan)


def T_practica(lam, K, gamma=GAMMA):
    """Truncated Neumann series: Σ_{t=0}^{K+1} γ^t λ^t."""
    coeffs = np.array([gamma**t for t in range(K + 2)])
    return sum(c * lam**t for t, c in enumerate(coeffs))


# ── Plots ─────────────────────────────────────────────────────────────────────


def _draw_eigenvalue_rug(ax, abs_eigenvalues):
    abs_ev_rug = abs_eigenvalues[(abs_eigenvalues > 0.05) & (abs_eigenvalues < 0.98)]
    for lev in abs_ev_rug:
        ax.axvline(lev, color="#bbbbbb", lw=0.6, alpha=0.25, zorder=1)
    ax.axvline(0.9357, color="#CC0000", lw=1.1, ls=":", alpha=0.7, zorder=2)
    ax.text(0.9357, 1.1, r"$\lambda_2{\approx}0.94$", color="#CC0000",
            fontsize=7.5, ha="right", va="bottom", rotation=90)


def plot_T_comparison(ax, K, U, abs_eigenvalues, gamma=GAMMA):
    """Three T curves: ideal, practical (truncated Neumann), learned (both archs)."""
    lam = np.linspace(0.001, 0.999, 2000)

    ax.semilogy(lam, T_ideal(lam, gamma), color=C_IDEAL, lw=2.2, ls="--",
                label=r"$T_{\rm ideal}(\lambda)=\frac{1}{1-\gamma\lambda}$", zorder=5)
    ax.semilogy(lam, np.abs(T_practica(lam, K, gamma)), color=C_PRACT, lw=2.0, ls="-",
                label=r"$T_{\rm práctica}$ ($\gamma^t$, trunc. Neumann)", zorder=4)

    for arch, col, ls, mrk in [(1, C_ARCH1, "-", "o"), (2, C_ARCH2, "--", "D")]:
        h, K_2 = load_coeffs(K, U, arch)
        if h is None:
            continue
        K_2_int = K_2 if K_2 is not None else -1
        coeffs = build_T_aprendida_coeffs(h, K, K_2_int)
        Ta = np.array([eval_poly(coeffs, l) for l in lam])
        ax.semilogy(lam, np.abs(Ta), color=col, lw=1.8, ls=ls,
                    label=f"$T_{{\\rm aprend.}}$ Arch {arch} ($h_t$ aprendidos)", zorder=3)

        abs_ev = abs_eigenvalues[abs_eigenvalues > 0.05]
        Ta_ev = np.array([abs(eval_poly(coeffs, l)) for l in abs_ev])
        ax.scatter(abs_ev, Ta_ev, s=20, color=col, marker=mrk,
                   edgecolors="white", linewidths=0.4, zorder=6, alpha=0.85)

    _draw_eigenvalue_rug(ax, abs_eigenvalues)
    ax.set_xlabel(r"$|\lambda|$", fontsize=11)
    ax.set_ylabel(r"$|T(\lambda)|$ (log)", fontsize=11)
    ax.set_title(f"$K={K}$, $U={U}$", fontsize=11, pad=4)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.8, 200)
    ax.grid(True, which="both", alpha=0.15)
    ax.legend(fontsize=8.5, framealpha=0.92, loc="upper left")
    ax.axvline(1.0, color="#aaaaaa", lw=0.8, ls=":", alpha=0.6)


def plot_T_composition(ax, K, U, abs_eigenvalues, arch, gamma=GAMMA):
    """T^u(λ) for u=1,...,U with color gradient (frozen-policy approximation)."""
    lam = np.linspace(0.001, 0.999, 2000)

    ax.semilogy(lam, T_ideal(lam, gamma), color=C_IDEAL, lw=2.0, ls="--",
                label=r"$T_{\rm ideal}$", zorder=6)
    ax.semilogy(lam, np.abs(T_practica(lam, K, gamma)), color=C_PRACT, lw=1.8, ls="-.",
                label=r"$T_{\rm práctica}$ ($\gamma^t$)", zorder=5)

    h, K_2 = load_coeffs(K, U, arch)
    if h is None:
        ax.text(0.5, 0.5, f"Sin datos Arch {arch}", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return

    K_2_int = K_2 if K_2 is not None else -1
    H_r_c, G_q_c = get_Hr_Gq(h, K, K_2_int)
    base_col = np.array(plt.cm.Blues(0.9)[:3]) if arch == 1 else np.array(plt.cm.Oranges(0.9)[:3])
    col_final = C_ARCH1 if arch == 1 else C_ARCH2

    for u in range(1, U + 1):
        alpha = 0.25 + 0.75 * (u / U)
        lw = 0.9 + 1.3 * (u / U)
        col = tuple(base_col * alpha + np.ones(3) * (1 - alpha))

        def T_u(l, _u=u):
            Hr = eval_poly(H_r_c, l)
            Gq = eval_poly(G_q_c, l)
            denom = 1.0 - Gq
            return Hr * (1.0 - Gq**_u) / denom if abs(denom) > 1e-9 else Hr * _u

        Ta_u = np.array([abs(T_u(l)) for l in lam])
        label = f"$T^{{u={u}}}$ aprendida" if u in (1, U) else None
        ax.semilogy(lam, Ta_u, color=col_final if u == U else col,
                    lw=lw, ls="-" if u == U else ":", alpha=0.85, label=label, zorder=3)

    _draw_eigenvalue_rug(ax, abs_eigenvalues)
    ax.set_xlabel(r"$|\lambda|$", fontsize=10)
    ax.set_ylabel(r"$|T^u(\lambda)|$ (log)", fontsize=10)
    ax.set_title(f"Arch {arch} — $K={K}$, $U={U}$", fontsize=10, pad=3)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.5, 500)
    ax.grid(True, which="both", alpha=0.13)
    ax.legend(fontsize=7.5, framealpha=0.9, loc="upper left")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    print("Loading eigenvalues...")
    abs_eigs = get_eigenvalues()
    print(f"  |λ_0|={abs_eigs[0]:.6f}, |λ_1|={abs_eigs[1]:.6f}")

    # Figure 1: T_comparison (TFG figure) ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, cfg in zip(axes, CONFIGS):
        plot_T_comparison(ax, cfg["K"], cfg["num_unrolls"], abs_eigs)
    axes[1].set_ylabel("")
    fig.suptitle(
        "Comparación de operadores de transferencia $T(\\lambda)$\n"
        r"Ideal ($\infty$ términos) vs Práctica ($\gamma^t$, K+2 términos)"
        r" vs Aprendida ($h_t$, K+2 términos)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = RESULTS_DIR / "T_comparison.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"✓ Saved: {out}")

    # Figure 2: layer composition ──────────────────────────────────────────────
    print("\nGenerating layer composition figure...")
    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for row, cfg in enumerate(CONFIGS):
        for col, arch in enumerate([1, 2]):
            plot_T_composition(axes2[row, col], cfg["K"], cfg["num_unrolls"], abs_eigs, arch)
            if col > 0:
                axes2[row, col].set_ylabel("")
    fig2.suptitle(
        "Impacto de la composición de capas sobre $T^u(\\lambda)$\n"
        r"(Aprox. política congelada: $T^u = H_r \cdot (1-G_q^u)/(1-G_q)$, $q_0=0$)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig2.tight_layout()
    out2 = RESULTS_DIR / "T_composition.png"
    fig2.savefig(out2, bbox_inches="tight", dpi=200)
    plt.close(fig2)
    print(f"✓ Saved: {out2}")

    # Summary table ────────────────────────────────────────────────────────────
    print("\n=== T at λ=1 (T_ideal max = 100) ===")
    print(f"{'Config':<22} {'T_ideal':>10} {'T_pract':>10} {'T_aprend':>12}")
    print("-" * 58)
    for cfg in CONFIGS:
        K, U = cfg["K"], cfg["num_unrolls"]
        Ti1 = 1.0 / (1 - GAMMA)
        Tp1 = float(T_practica(np.array([1.0]), K)[0])
        for arch in [1, 2]:
            h, K_2 = load_coeffs(K, U, arch)
            if h is None:
                continue
            K_2_int = K_2 if K_2 is not None else -1
            coeffs = build_T_aprendida_coeffs(h, K, K_2_int)
            Ta1 = eval_poly(coeffs, 1.0)
            print(f"  Arch{arch} K={K} U={U}: {Ti1:10.2f} {Tp1:10.4f} {Ta1:12.4f}")


if __name__ == "__main__":
    main()
