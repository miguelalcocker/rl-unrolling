"""
Empirical Amplification Analysis
==================================

Para cada modelo entrenado (K, arch, U_max), mide empíricamente cuánto amplifica
la red cada autovector de P_π^{uniforme}:

    amp_U_i = ||q_U(v_i)||_2 / ||v_i||_2

donde v_i es el i-ésimo autovector de P_π (política uniforme), ordenados por |λ_i| desc.

Referencia teórica: resolvente T_res(λ_i) = |H_r(λ_i) / (1 - G_q(λ_i))|,
que corresponde al límite U→∞ con política fija y q_0=0.

Autovectores complejos → se usa la parte real (normalizada).

NOTA: Este script depende de `freq_response_per_unroll.py`, que define CONFIGS, CFG,
build_Hr_Gq_coeffs, eval_poly y las constantes de color C_ARCH1/C_ARCH2/C_IDEAL/C_PRACT.
Ese fichero se perdió del repositorio (nunca fue commiteado) y debe ser recreado
antes de poder ejecutar este script. Los resultados precomputados en
freq_per_unroll_results/ y las figuras finales en TFG/memoria/images/cap4/ están
disponibles.

Uso:
    python experiments/empirical_amplification.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Re-usar configuración del experimento ────────────────────────────────────
from freq_response_per_unroll import (
    CONFIGS, CFG,
    build_Hr_Gq_coeffs, eval_poly,
    C_ARCH1, C_ARCH2, C_IDEAL, C_PRACT,
)

RESULTS_DIR = Path(CFG["results_dir"])
GAMMA       = CFG["gamma"]

# ─────────────────────────────────────────────────────────────────────────────
# Reconstrucción de entorno y autovectores
# ─────────────────────────────────────────────────────────────────────────────

def get_env_tensors():
    """Devuelve (P, r, nS, nA) para CliffWalkingEnv."""
    from src.environments import CliffWalkingEnv
    env = CliffWalkingEnv()
    return env.P, env.r, env.nS, env.nA


def uniform_policy(nS, nA):
    """Política uniforme como tensor (nS, nA)."""
    return torch.ones(nS, nA) / nA


def build_Ppi(P, Pi, nS, nA):
    """P_π = P @ Π_ext  →  shape (nS*nA, nS*nA)."""
    Pi_ext = torch.zeros(nS, nS * nA)
    rows = torch.arange(nS).repeat_interleave(nA)
    cols = torch.arange(nS * nA)
    Pi_ext[rows, cols] = Pi.flatten()
    return P @ Pi_ext   # (nS*nA, nS*nA)


def get_eigenvectors(Ppi_np):
    """
    Autovalores y autovectores (reales) de P_π.
    Para autovectores complejos se usa la parte real normalizada.
    Devuelve (abs_eigs, real_eigvecs, sort_idx).
      abs_eigs:   array (n,) con |λ_i|, ordenado descendente
      real_eigvecs: array (n, n) cada columna es el autovector de entrada
      sort_idx:   índices de ordenación
    """
    lam, V = np.linalg.eig(Ppi_np)
    order  = np.argsort(np.abs(lam))[::-1]  # desc |λ|
    lam    = lam[order]
    V      = V[:, order]                     # (n, n), columnas = autovectores

    # Tomar parte real de cada autovector y normalizar
    V_real = np.real(V).astype(np.float32)
    norms  = np.linalg.norm(V_real, axis=0, keepdims=True)
    norms  = np.where(norms < 1e-9, 1.0, norms)
    V_real /= norms

    return np.abs(lam).real.astype(np.float32), V_real, order


# ─────────────────────────────────────────────────────────────────────────────
# Reconstrucción del PE layer a partir de h_coeffs
# ─────────────────────────────────────────────────────────────────────────────

def pe_forward(h_np, K, K_2, arch, q, Pi, P, r, nS, nA, beta=1.0):
    """
    Aplica una capa PolicyEvaluationLayer reconstruida con h_np.
    Devuelve q' (numpy array).
    """
    from src.models import PolicyEvaluationLayer
    h_t = torch.from_numpy(h_np).float()
    layer = PolicyEvaluationLayer(
        P=P, r=r, nS=nS, nA=nA,
        K=K, beta=beta,
        shared_h=None,
        architecture_type=arch,
        K_2=K_2 if K_2 >= 0 else None,
    )
    layer.h.data = h_t
    layer.eval()
    with torch.no_grad():
        out = layer(q, Pi)
    return out.numpy()


def full_net_forward(h_np, K, K_2, arch, U, q_init, Pi_init, P, r, nS, nA, beta=1.0, tau=5.0):
    """
    Aplica la red completa de U capas (PE + PI alternadas) con h_np compartido.
    Devuelve q_final (numpy array).
    """
    from src.models import PolicyEvaluationLayer, PolicyImprovementLayer
    h_t = torch.from_numpy(h_np).float()
    pe  = PolicyEvaluationLayer(
        P=P, r=r, nS=nS, nA=nA,
        K=K, beta=beta,
        shared_h=None,
        architecture_type=arch,
        K_2=K_2 if K_2 >= 0 else None,
    )
    pe.h.data = h_t
    pe.eval()
    pi_layer = PolicyImprovementLayer(nS=nS, nA=nA, tau=tau)

    q  = q_init.clone()
    Pi = Pi_init.clone()
    with torch.no_grad():
        for _ in range(U):
            q  = pe(q, Pi)
            Pi = pi_layer(q)
    return q.numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Cálculo de amplificaciones empíricas
# ─────────────────────────────────────────────────────────────────────────────

def compute_amplification(h_np, K, K_2, arch, U, V_real, P, r, nS, nA,
                          Pi_unif, tau=5.0):
    """
    Calcula λ_emp_i = ||y_i||_2 / ||v_i||_2  para cada autovector v_i.

    Dos variantes:
      amp_1 : 1 capa PE  → y_i = PE(v_i, Π)            (lineal + r)
      amp_U : U capas    → y_i = red_completa(v_i, Π)   (con softmax)

    Nota: sin resta de baseline. El sesgo de r está presente y es el
    comportamiento real de la red ante la excitación v_i.
    """
    n = V_real.shape[1]
    amp_1 = np.empty(n, dtype=np.float32)
    amp_U = np.empty(n, dtype=np.float32)

    for i in range(n):
        v = torch.from_numpy(V_real[:, i]).float()
        norm_v = float(torch.norm(v))
        if norm_v < 1e-9:
            amp_1[i] = np.nan
            amp_U[i] = np.nan
            continue

        # 1 capa PE: ||PE(v_i)||/||v_i||
        yi_1   = pe_forward(h_np, K, K_2, arch, v, Pi_unif, P, r, nS, nA)
        amp_1[i] = float(np.linalg.norm(yi_1)) / norm_v

        # U capas (red completa con softmax): ||q_U(v_i)||/||v_i||
        yi_U   = full_net_forward(h_np, K, K_2, arch, U, v, Pi_unif,
                                  P, r, nS, nA, tau=tau)
        amp_U[i] = float(np.linalg.norm(yi_U)) / norm_v

    return amp_1, amp_U


# ─────────────────────────────────────────────────────────────────────────────
# Figura
# ─────────────────────────────────────────────────────────────────────────────

def compute_amp_incremental(h_np, K, K_2, arch, U, V_real, P, r, nS, nA,
                            Pi_unif, tau=5.0):
    """
    amp_incr_i = ||q_U(v_i) - q_U(0)|| / ||v_i||

    Resta el baseline q_U(0) (respuesta solo al reward, sin eigenmode),
    aislando la contribución de v_i a través de los U pasos no lineales.

    Para U=1 (lineal): coincide exactamente con |G_q(λ_i)| (verificación).
    Para U>1 con softmax: captura la amplificación no lineal real.
    """
    n = V_real.shape[1]
    amp_incr = np.empty(n, dtype=np.float32)

    zeros = torch.zeros(V_real.shape[0])
    q_U_0 = full_net_forward(h_np, K, K_2, arch, U, zeros, Pi_unif,
                              P, r, nS, nA, tau=tau)

    for i in range(n):
        v = torch.from_numpy(V_real[:, i]).float()
        norm_v = float(torch.norm(v))
        if norm_v < 1e-9:
            amp_incr[i] = np.nan
            continue
        q_U_vi = full_net_forward(h_np, K, K_2, arch, U, v, Pi_unif,
                                  P, r, nS, nA, tau=tau)
        amp_incr[i] = float(np.linalg.norm(q_U_vi - q_U_0)) / norm_v

    return amp_incr


def plot_amplification_incremental(results_dir, df, P, r, nS, nA,
                                    abs_eigs, V_real, configs_list, fname_suffix=""):
    """
    Amplificación incremental ||q_U(v_i) - q_U(0)|| / ||v_i||:
    estimación empírica de autovalores del operador no lineal (con softmax).

    Referencia teórica: |G_q(λ_i)|^U  (predicción lineal, política fija U pasos).

    Si empírico < |G_q|^U → el softmax amortigua el modo (estabilización).
    Si empírico > |G_q|^U → el softmax amplifica el modo más de lo lineal.
    """
    Pi_unif = uniform_policy(nS, nA)
    eig_idx = np.arange(1, len(abs_eigs) + 1)

    n_rows = len(configs_list)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 5.5 * n_rows), sharey=False)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, cfg_row in enumerate(configs_list):
        K, U_max = cfg_row["K"], cfg_row["U_max"]

        for col, arch in enumerate([1, 2]):
            ax = axes[row, col]
            col_main = C_ARCH1 if arch == 1 else C_ARCH2

            # ── Cargar modelo representativo ─────────────────────────────────
            sub = df[
                (df["K"] == K) & (df["num_unrolls"] == U_max) &
                (df["arch"] == arch) & (df["policy_file"].notna())
            ]
            if len(sub) == 0:
                ax.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                        transform=ax.transAxes, color="gray"); continue

            npz_path = results_dir / sub.iloc[0]["policy_file"]
            if not npz_path.exists():
                ax.text(0.5, 0.5, "Archivo no encontrado", ha="center",
                        va="center", transform=ax.transAxes, color="gray"); continue

            data = np.load(npz_path)
            h_np = data["h_coeffs"]
            K_2  = int(data["K_2"])

            # POG test median
            pog_sub = df[(df["K"] == K) & (df["num_unrolls"] == U_max) &
                         (df["arch"] == arch) & (df["success"] == True)]
            pog_med = (pog_sub["policy_optimality_gap_joint_test"].median()
                       if len(pog_sub) else float("nan"))

            # ── Amplificación incremental empírica ───────────────────────────
            amp_incr = compute_amp_incremental(
                h_np, K, K_2, arch, U_max, V_real,
                P, r, nS, nA, Pi_unif, tau=CFG["tau"]
            )

            # ── Referencia teórica: |G_q(λ_i)|^U ────────────────────────────
            _, Gq = build_Hr_Gq_coeffs(h_np, K, K_2)
            gq_vals = np.array([abs(eval_poly(Gq, float(l))) for l in abs_eigs],
                               dtype=np.float32)
            gq_U = gq_vals ** U_max

            # ── Límites dinámicos del eje Y ──────────────────────────────────
            emp_pos = amp_incr[np.isfinite(amp_incr) & (amp_incr > 0)]
            gq_pos = gq_U[np.isfinite(gq_U) & (gq_U > 0)] # Tomar en cuenta la curva teórica
            
            if len(emp_pos) == 0:
                ax.text(0.5, 0.5, "Sin datos finitos", ha="center",
                        va="center", transform=ax.transAxes); continue
            
            # Encontrar el mínimo absoluto real combinando ambas curvas
            min_val = float(np.min(emp_pos))
            if len(gq_pos) > 0:
                min_val = min(min_val, float(np.min(gq_pos)))
            
            # Bajar el tope a 1e-35 para que quepan las caídas exponenciales de Arch 1
            ymin = max(min_val * 0.1, 1e-35)
            ymax = min(float(np.percentile(emp_pos, 99)) * 3, 2000)

            # ── Trazar ───────────────────────────────────────────────────────
            kw_line = dict(lw=0.6, alpha=0.35, zorder=2)
            mrk = "o" if arch == 1 else "D"

            # 1. |G_q(λ_i)|^U — teórico lineal
            gq_U_clip = np.where(gq_U > ymax * 10, np.nan, gq_U)
            ax.semilogy(eig_idx, gq_U_clip, color=C_IDEAL, ls="--", **kw_line)
            ax.scatter(eig_idx, gq_U_clip, color=C_IDEAL, s=16, marker="s", zorder=4,
                       label=rf"$|G_q(\lambda_i)|^{{U={U_max}}}$ (lineal, pol. fija)")

            # 2. Empírico incremental (softmax, no lineal)
            amp_clip = np.where(amp_incr > ymax * 10, np.nan, amp_incr)
            ax.semilogy(eig_idx, amp_clip, color=col_main, ls="-", **kw_line)
            ax.scatter(eig_idx, amp_clip, color=col_main, s=22,
                       marker=mrk, zorder=6,
                       label=rf"Empírico $U={U_max}$: $\|q_U(v_i)-q_U(0)\|/\|v_i\|$")

            # ── Ticks eje X ──────────────────────────────────────────────────
            n_eigs    = len(eig_idx)
            tick_step = max(1, n_eigs // 10)
            tick_pos  = eig_idx[::tick_step]
            tick_lbl  = [rf"$i={i}$" + "\n" + rf"$|\lambda|={abs_eigs[i-1]:.3f}$"
                         for i in tick_pos]
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_lbl, fontsize=6.5)

            ax.set_xlabel(r"Índice $i$ ($|\lambda_1|\geq|\lambda_2|\geq\cdots$)",
                          fontsize=9)
            ax.set_ylabel(
                r"$\|q_U(v_i)-q_U(0)\|_2\,/\,\|v_i\|_2$  (log)", fontsize=8)
            ax.set_title(
                f"Arch {arch}  —  $K={K}$,  $U_{{\\rm max}}={U_max}$"
                f"  |  POG$_{{\\rm test}}={pog_med:.4f}$",
                fontsize=9.5, pad=4
            )
            ax.set_xlim(0.5, n_eigs + 0.5)
            ax.set_ylim(ymin, ymax)
            ax.grid(True, which="both", alpha=0.13)
            ax.legend(fontsize=7.5, framealpha=0.92, loc="upper right")

    fig.suptitle(
        r"Amplificación incremental $\|q_U(v_i)-q_U(0)\|/\|v_i\|$"
        " — estimación empírica de autovalores del operador no lineal"
        "\n"
        r"Gris ─ ─: $|G_q(\lambda_i)|^U$ (teórico lineal)  ·  "
        r"Color: empírico $U$ capas (softmax)",
        fontsize=10, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = results_dir / f"empirical_amplification_incremental{fname_suffix}.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"✓ Figura guardada: {out}")


def plot_amplification(results_dir, df, P, r, nS, nA,
                       abs_eigs, V_real, configs_list, fname_suffix=""):
    """
    Para cada panel (K, arch):
      · T_res(λ_i) = |H_r(λ_i) / (1 - G_q(λ_i))|  ← resolvente teórica (referencia)
      · Empírico U capas: ||q_U(v_i)||/||v_i||       ← red completa con softmax
    """
    Pi_unif = uniform_policy(nS, nA)
    eig_idx = np.arange(1, len(abs_eigs) + 1)

    n_rows = len(configs_list)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 5.5 * n_rows), sharey=False)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, cfg_row in enumerate(configs_list):
        K, U_max = cfg_row["K"], cfg_row["U_max"]

        for col, arch in enumerate([1, 2]):
            ax = axes[row, col]
            col_main = C_ARCH1 if arch == 1 else C_ARCH2

            # ── Cargar modelo representativo ─────────────────────────────────
            sub = df[
                (df["K"] == K) & (df["num_unrolls"] == U_max) &
                (df["arch"] == arch) & (df["policy_file"].notna())
            ]
            if len(sub) == 0:
                ax.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                        transform=ax.transAxes, color="gray"); continue

            npz_path = results_dir / sub.iloc[0]["policy_file"]
            if not npz_path.exists():
                ax.text(0.5, 0.5, "Archivo no encontrado", ha="center",
                        va="center", transform=ax.transAxes, color="gray"); continue

            data = np.load(npz_path)
            h_np = data["h_coeffs"]
            K_2  = int(data["K_2"])

            # POG test median (sobre todas las runs exitosas)
            pog_sub = df[(df["K"] == K) & (df["num_unrolls"] == U_max) &
                         (df["arch"] == arch) & (df["success"] == True)]
            pog_med = (pog_sub["policy_optimality_gap_joint_test"].median()
                       if len(pog_sub) else float("nan"))

            # ── Calcular amplificación empírica U capas ───────────────────────
            _, amp_U = compute_amplification(
                h_np, K, K_2, arch, U_max, V_real,
                P, r, nS, nA, Pi_unif, tau=CFG["tau"]
            )

            # ── Calcular resolvente teórica T_res(λ_i) ───────────────────────
            Hr, Gq = build_Hr_Gq_coeffs(h_np, K, K_2)
            Hr_l = np.array([eval_poly(Hr, float(l)) for l in abs_eigs], dtype=complex)
            Gq_l = np.array([eval_poly(Gq, float(l)) for l in abs_eigs], dtype=complex)
            denom = 1.0 - Gq_l
            t_res = np.abs(
                np.where(np.abs(denom) > 1e-9, Hr_l / denom, np.nan + 0j)
            ).astype(np.float32)
            rho = float(np.max(np.abs(Gq_l[np.isfinite(Gq_l)])))

            # ── Límites dinámicos del eje Y ──────────────────────────────────
            # amp_U ≈ constante (bias de r domina) — usarlo para escalar el eje.
            # t_res puede divergir (ρ>1): no lo usamos para ymin.
            amp_pos = amp_U[np.isfinite(amp_U) & (amp_U > 0)]
            if len(amp_pos) == 0:
                ax.text(0.5, 0.5, "Sin datos finitos", ha="center",
                        va="center", transform=ax.transAxes); continue
            # Usamos np.min para no perder ningún valor y bajamos el tope a 1e-20
            ymin = max(float(np.min(amp_pos)) * 0.5, 1e-20)
            ymax = min(float(np.percentile(amp_pos, 99)) * 3, 2000)

            # ── Trazar ───────────────────────────────────────────────────────
            kw_line = dict(lw=0.6, alpha=0.35, zorder=2)
            mrk = "o" if arch == 1 else "D"

            # 1. T_res(λ_i) teórica (resolvente)
            t_res_clipped = np.where(t_res > ymax * 10, np.nan, t_res)
            ax.semilogy(eig_idx, t_res_clipped, color=C_IDEAL, ls="--", **kw_line)
            ax.scatter(eig_idx, t_res_clipped, color=C_IDEAL, s=16, marker="s", zorder=4,
                       label=rf"$T_{{\rm res}}(\lambda_i) = |H_r/(1-G_q)|$  ($\varrho={rho:.2f}$)")

            # 2. Empírico U capas (red completa con softmax)
            amp_U_clipped = np.where(amp_U > ymax * 10, np.nan, amp_U)
            ax.semilogy(eig_idx, amp_U_clipped, color=col_main, ls="-", **kw_line)
            ax.scatter(eig_idx, amp_U_clipped, color=col_main, s=22,
                       marker=mrk, zorder=6,
                       label=rf"Empírico $U={U_max}$ capas: $\|q_U(v_i)\|/\|v_i\|$")

            # ── Anotación ρ > 1 ──────────────────────────────────────────────
            if rho > 1:
                ax.text(0.97, 0.97,
                        r"$\varrho > 1$: $T_{\rm res}\to\infty$",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=8, color="red",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec="red", alpha=0.8))

            # ── Ticks eje X ──────────────────────────────────────────────────
            n_eigs    = len(eig_idx)
            tick_step = max(1, n_eigs // 10)
            tick_pos  = eig_idx[::tick_step]
            tick_lbl  = [rf"$i={i}$" + "\n" + rf"$|\lambda|={abs_eigs[i-1]:.3f}$"
                         for i in tick_pos]
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_lbl, fontsize=6.5)

            ax.set_xlabel(r"Índice $i$ ($|\lambda_1|\geq|\lambda_2|\geq\cdots$)",
                          fontsize=9)
            ax.set_ylabel(r"$\|y_i\|_2\,/\,\|v_i\|_2$  (log)", fontsize=8)
            ax.set_title(
                f"Arch {arch}  —  $K={K}$,  $U_{{\\rm max}}={U_max}$"
                f"  |  POG$_{{\\rm test}}={pog_med:.4f}$",
                fontsize=9.5, pad=4
            )
            ax.set_xlim(0.5, n_eigs + 0.5)
            ax.set_ylim(ymin, ymax)
            ax.grid(True, which="both", alpha=0.13)
            ax.legend(fontsize=7.5, framealpha=0.92, loc="upper right")

    fig.suptitle(
        r"Amplificación empírica $\|q_U(v_i)\|/\|v_i\|$ por autovector $v_i$ de $P_\pi$"
        "\n"
        r"Gris ─ ─: resolvente $T_{\rm res}(\lambda_i) = |H_r/(1-G_q)|$ (teórico)  ·  "
        r"Color: empírico $U$ capas (softmax)",
        fontsize=10, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = results_dir / f"empirical_amplification{fname_suffix}.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"✓ Figura guardada: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    csv_path = RESULTS_DIR / "all_results.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} no encontrado. Ejecuta primero freq_response_per_unroll.py")
        return

    df = pd.read_csv(csv_path)
    # Mantener solo éxitos con policy_file asignado
    df = df[df["success"] == True].copy()

    print("Construyendo entorno y autovectores...")
    P, r, nS, nA = get_env_tensors()
    Pi_unif      = uniform_policy(nS, nA)
    Ppi          = build_Ppi(P, Pi_unif, nS, nA)
    Ppi_np       = Ppi.numpy()

    abs_eigs, V_real, _ = get_eigenvectors(Ppi_np)
    print(f"  nS={nS}, nA={nA}  →  {len(abs_eigs)} autovalores")
    print(f"  |λ_1|={abs_eigs[0]:.5f},  |λ_2|={abs_eigs[1]:.5f},  ...")

    u_groups = sorted({c["U_max"] for c in CONFIGS})
    for U_group in u_groups:
        cfg_sub = [c for c in CONFIGS if c["U_max"] == U_group]
        suf     = f"_U{U_group}"
        print(f"\nGenerando figura U_max={U_group}  ({len(cfg_sub)} configs)...")
        plot_amplification(
            RESULTS_DIR, df, P, r, nS, nA,
            abs_eigs, V_real, cfg_sub, fname_suffix=suf,
        )
        plot_amplification_incremental(
            RESULTS_DIR, df, P, r, nS, nA,
            abs_eigs, V_real, cfg_sub, fname_suffix=suf,
        )

    print("\nListo.")


if __name__ == "__main__":
    main()
