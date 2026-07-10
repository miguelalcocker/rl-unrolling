"""Visualize Cliff Variations Experimental Results.

Generates three types of figures per variant, mirroring the style of
visualize_unrolls_results_tfg.py but adapted for cliff transfer experiments:

  1. {variant}_agreement.png  — 3×2: POG (non-sq), PVG, Agreement vs K
                                   IQR bands for all metrics (from CSV)
                                   Fixed num_unrolls = NUM_UNROLLS_SHOW
  2. {variant}_{U}_policy_maps_train.png  — policy maps TRAIN (universal scale)
     {variant}_{U}_policy_maps_test.png   — policy maps TEST  (universal scale)
  3. {variant}_{U}_comprehensive.png      — 4×2 all 8 metrics vs K

POG and PVG use non-squared (relative L2 error) form for comparability.
POG is stored squared in CSV → we apply sqrt before plotting.
PVG is stored non-squared (already in correct form).

Usage:
    python visualize_cliff_variations.py                          # all variants
    python visualize_cliff_variations.py --variants std_mirrored  # single variant
    python visualize_cliff_variations.py --unrolls 10             # fixed unrolls for maps/comprehensive
"""

import argparse
import glob as glob_module
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
import pandas as pd
import torch
from pathlib import Path

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.environments import GeneralizedCliffWalkingEnv
from src.algorithms.generalized_policy_iteration import PolicyIterationTrain
from src.plots import plot_policy_and_value, ARCH_COLORS
from pytorch_lightning import Trainer


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_BASE    = Path("cliff_variations_results")
NUM_UNROLLS_SHOW = 10          # default fixed unrolls for agreement/maps figures
K_SHOW           = 10          # K used for policy map representative selection

COLORS_ARCH = ARCH_COLORS

ALL_VARIANTS = [
    "std_mirrored", "mirrored_std", "std_vertical",
    "std_narrow",   "std_tall",     "large_mirrored",
]

# Environment factory for each variant (train / test)
def _build_env(env_type):
    if env_type == "standard":
        return GeneralizedCliffWalkingEnv.standard()
    elif env_type == "mirrored":
        return GeneralizedCliffWalkingEnv.mirrored()
    elif env_type == "vertical_cliff":
        return GeneralizedCliffWalkingEnv.vertical_cliff()
    elif env_type == "narrow":
        return GeneralizedCliffWalkingEnv(
            nrows=4, ncols=12,
            cliff_cells=[(0, c) for c in range(3, 10)],
            start=(0, 0), goal=(0, 11),
        )
    elif env_type == "tall":
        return GeneralizedCliffWalkingEnv(
            nrows=4, ncols=12,
            cliff_cells=[(0, c) for c in range(1, 11)] + [(1, c) for c in range(1, 11)],
            start=(2, 0), goal=(2, 11),
        )
    elif env_type == "scaled_mirrored":
        return GeneralizedCliffWalkingEnv.scaled(scale=2, mirrored=True)
    else:
        raise ValueError(f"Unknown env_type: {env_type!r}")


VARIANT_ENVS = {
    "std_mirrored": ("standard",      "mirrored",       3, 0),
    "mirrored_std": ("mirrored",      "standard",       0, 3),
    "std_vertical": ("standard",      "vertical_cliff",  3, 0),
    "std_narrow":   ("standard",      "narrow",          3, 0),
    "std_tall":     ("standard",      "tall",            3, 2),
    "large_mirrored": ("standard",    "scaled_mirrored", 3, 0),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_opt_q(env, goal_row):
    m = PolicyIterationTrain(env, gamma=0.99, goal_row=goal_row, max_eval_iters=50)
    t = Trainer(max_epochs=50, enable_progress_bar=False, enable_model_summary=False,
                logger=False, enable_checkpointing=False)
    t.fit(m)
    return m.q.detach()


def _get_repr_policy_path(results_dir, K, arch, num_unrolls):
    """Test-representative policy file (no _train suffix), fallback to _train."""
    pattern = str(results_dir / f"policy_arch{arch}_K{K}_unrolls{num_unrolls}_run*.npz")
    candidates = glob_module.glob(pattern)
    test_files  = [f for f in candidates if not f.endswith('_train.npz')]
    train_files = [f for f in candidates if f.endswith('_train.npz')]
    if test_files:
        return Path(test_files[0])
    if train_files:
        return Path(train_files[0])
    return None


def _get_train_policy_path(results_dir, K, arch, num_unrolls):
    """Train-representative policy file (_train suffix), fallback to test repr."""
    pattern_tr = str(results_dir / f"policy_arch{arch}_K{K}_unrolls{num_unrolls}_run*_train.npz")
    train_files = glob_module.glob(pattern_tr)
    if train_files:
        return Path(train_files[0])
    return _get_repr_policy_path(results_dir, K, arch, num_unrolls)


def _plot_metric_on_axis(ax, df_u, col_tr, col_te, title, ylabel,
                         use_log=False, use_sci=False, transform_sqrt=False):
    """Plot metric with IQR bands for both architectures, train+test."""
    K_values = sorted(df_u['K'].unique())
    clean_fmt = ticker.FormatStrFormatter('%g')

    for arch in [1, 2]:
        color = COLORS_ARCH[arch]
        for mode, ls, mk, col in [('train', '-', 'o', col_tr), ('test', '--', 's', col_te)]:
            if col not in df_u.columns:
                continue
            med_vals, p25_vals, p75_vals = [], [], []
            for K in K_values:
                v = df_u[(df_u['architecture_type'] == arch) & (df_u['K'] == K)][col].dropna().values
                if transform_sqrt:
                    v = np.sqrt(np.maximum(v, 0.0))
                med_vals.append(np.median(v)        if len(v) else np.nan)
                p25_vals.append(np.percentile(v, 25) if len(v) else np.nan)
                p75_vals.append(np.percentile(v, 75) if len(v) else np.nan)
            med_vals = np.array(med_vals)
            ax.plot(K_values, med_vals, ls, marker=mk, linewidth=2.5,
                    markersize=8, color=color,
                    label=f'Arch {arch} — {mode.capitalize()}')
            ax.fill_between(K_values,
                            np.array(p25_vals), np.array(p75_vals),
                            alpha=0.15, color=color)

    if use_log:
        ax.set_yscale('log')
    ax.set_xlabel('Filter Order K', fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(K_values)
    ax.grid(True, alpha=0.2, linestyle=':')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.yaxis.set_major_formatter(clean_fmt)
    if not use_log:
        ax.set_ylim(bottom=0)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Agreement figure (3×2)
# ─────────────────────────────────────────────────────────────────────────────

def plot_agreement_figure(variant_name, results_dir, num_unrolls=NUM_UNROLLS_SHOW):
    """3×2 figure: Row 0 POG (non-sq), Row 1 PVG, Row 2 Agreement, all vs K.

    All metrics use non-squared relative L2 form so POG and PVG are on the
    same scale.  POG stored squared → sqrt applied here.
    """
    csv_path = results_dir / "all_experiments_results.csv"
    if not csv_path.exists():
        print(f"  [skip {variant_name}] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df = df[df['success'] == True]
    df_u = df[df['num_unrolls'] == num_unrolls].copy()
    if len(df_u) == 0:
        print(f"  [skip {variant_name}] no data for num_unrolls={num_unrolls}")
        return

    K_values = sorted(df_u['K'].unique())
    n_runs   = df_u['run_idx'].nunique()

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    clean_fmt = ticker.FormatStrFormatter('%g')

    # ── Row 0: POG (non-squared = sqrt of stored squared) ─────────────
    pog_specs = [
        ('policy_optimality_gap_joint_train',    'policy_optimality_gap_joint_test',
         r'POG Joint  $\frac{\|q_\pi - q^*\|}{\|q^*\|}$',
         r'rel. L2 error (non-sq)', True),
        ('policy_optimality_gap_separate_train', 'policy_optimality_gap_separate_test',
         r'POG Separate  $\left\|\frac{q_\pi}{\|q_\pi\|} - \frac{q^*}{\|q^*\|}\right\|$',
         r'rel. L2 error (non-sq)', False),
    ]
    for col, (col_tr, col_te, title, ylabel, use_log) in enumerate(pog_specs):
        _plot_metric_on_axis(axes[0, col], df_u, col_tr, col_te,
                             title, ylabel, use_log=use_log, transform_sqrt=True)

    # ── Row 1: PVG (already non-squared) ──────────────────────────────
    pvg_specs = [
        ('policy_value_gap_joint_train',    'policy_value_gap_joint_test',
         r'PVG Joint  $\frac{\|V_\pi - V^*\|}{\|V^*\|}$',
         r'rel. L2 error'),
        ('policy_value_gap_separate_train', 'policy_value_gap_separate_test',
         r'PVG Separate  $\left\|\frac{V_\pi}{\|V_\pi\|} - \frac{V^*}{\|V^*\|}\right\|$',
         r'rel. L2 error'),
    ]
    for col, (col_tr, col_te, title, ylabel) in enumerate(pvg_specs):
        _plot_metric_on_axis(axes[1, col], df_u, col_tr, col_te, title, ylabel)

    # ── Row 2: Agreement (%) ───────────────────────────────────────────
    agree_specs = [
        ('agreement_hard_train', 'agreement_hard_test',
         'Hard Agreement (Greedy in A*)', 'Agreement (%)'),
        ('agreement_soft_train', 'agreement_soft_test',
         'Soft Agreement (prob mass on A*)', 'Agreement (%)'),
    ]
    for col, (col_tr, col_te, title, ylabel) in enumerate(agree_specs):
        ax = axes[2, col]
        _plot_metric_on_axis(ax, df_u, col_tr, col_te, title, ylabel)
        ax.set_ylim(-2, 105)
        ax.axhline(100, color='green', linestyle=':', linewidth=1.2, alpha=0.6)

    fig.suptitle(
        f'Cliff Variant: {variant_name}  |  num_unrolls={num_unrolls},  n={n_runs} runs\n'
        r'POG & PVG: non-squared relative L$_2$ error (POG stored squared → $\sqrt{\cdot}$ applied)'
        '  |  median ± IQR (25th–75th)',
        fontsize=11, fontweight='bold',
    )
    plt.tight_layout()

    out_dir = results_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{variant_name}_U{num_unrolls}_agreement.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Policy Maps with universal scale
# ─────────────────────────────────────────────────────────────────────────────

def _render_policy_map(ax, q_flat, Pi_np, q_opt_np, env_obj, goal_row, vmin, vmax, title):
    """Render a single policy map into ax using a precomputed V scale."""
    nS, nA = env_obj.nS, env_obj.nA
    nrows, ncols = env_obj.nrows, env_obj.ncols

    q_t  = torch.from_numpy(q_flat).float()
    Pi_t = torch.from_numpy(Pi_np).float()

    try:
        temp_fig = plot_policy_and_value(
            q_t.view(nS, nA), Pi_t,
            highlight_cliffs=True, goal_row=goal_row,
            shape=(nrows, ncols), min_prob=0.02, plot_all_trans=False,
            vmin=vmin, vmax=vmax,
        )
        temp_fig.canvas.draw()
        img = np.frombuffer(temp_fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(temp_fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        ax.imshow(img)
        plt.close(temp_fig)
    except Exception:
        ax.text(0.5, 0.5, 'render error', ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='red')
    ax.axis('off')
    ax.set_title(title, fontsize=9, fontweight='bold')


def plot_policy_maps(variant_name, results_dir, q_opt, q_opt_test, env_tr, env_te,
                     goal_row_tr, goal_row_te, num_unrolls=NUM_UNROLLS_SHOW):
    """Policy maps TRAIN and TEST with universal scale per mode.

    Rows = K values, Cols = [Arch1, Optimal, Arch2].
    Universal scale = [min V over all (K, arch)] to [max V].
    """
    csv_path = results_dir / "all_experiments_results.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    df = df[df['success'] == True]
    K_values = sorted(df['K'].unique())

    q_opt_np      = q_opt.cpu().numpy()
    q_opt_test_np = q_opt_test.cpu().numpy()

    for mode, env_obj, goal_row, q_opt_np_m, suffix in [
        ('train', env_tr, goal_row_tr, q_opt_np,      'train'),
        ('test',  env_te, goal_row_te, q_opt_test_np,  'test'),
    ]:
        nS, nA = env_obj.nS, env_obj.nA
        V_opt = torch.from_numpy(q_opt_np_m).float().view(nS, nA).max(dim=1).values.numpy()

        # Compute universal V scale
        all_V = [V_opt]
        for K in K_values:
            for arch in [1, 2]:
                path = (_get_train_policy_path(results_dir, K, arch, num_unrolls)
                        if mode == 'train' else
                        _get_repr_policy_path(results_dir, K, arch, num_unrolls))
                if path is None or not path.exists():
                    continue
                npz = np.load(path)
                q_key = f'q_{mode}'
                if q_key not in npz:
                    continue
                q_np = npz[q_key]
                if q_np.shape[0] != nS * nA:
                    continue
                V = torch.from_numpy(q_np).float().view(nS, nA).max(dim=1).values.numpy()
                all_V.append(V)
        if not all_V:
            continue
        vmin = float(min(v.min() for v in all_V))
        vmax = float(max(v.max() for v in all_V))

        n_K   = len(K_values)
        n_cols = 3  # Arch1 | Optimal | Arch2
        fig, axes = plt.subplots(n_K, n_cols,
                                 figsize=(4.5 * n_cols, 3.5 * n_K))
        if n_K == 1:
            axes = axes[np.newaxis, :]

        # Optimal column (same for all K rows)
        for row, K in enumerate(K_values):
            ax = axes[row, 1]
            _render_policy_map(ax, q_opt_np_m,
                               np.eye(nA)[np.zeros(nS, dtype=int)],   # dummy Pi
                               q_opt_np_m, env_obj, goal_row,
                               vmin, vmax, f'Optimal  (K={K})')

        # Arch columns
        for col_idx, arch in enumerate([1, 2]):
            col = 0 if arch == 1 else 2
            for row, K in enumerate(K_values):
                ax = axes[row, col]
                path = (_get_train_policy_path(results_dir, K, arch, num_unrolls)
                        if mode == 'train' else
                        _get_repr_policy_path(results_dir, K, arch, num_unrolls))
                if path is None or not path.exists():
                    ax.axis('off')
                    ax.set_title(f'Arch {arch}, K={K}: no file', fontsize=8)
                    continue
                npz  = np.load(path)
                q_key, pi_key = f'q_{mode}', f'Pi_{mode}'
                if q_key not in npz or pi_key not in npz:
                    ax.axis('off')
                    continue
                q_np, Pi_np = npz[q_key], npz[pi_key]
                if q_np.shape[0] != nS * nA or Pi_np.shape[0] != nS:
                    ax.axis('off')
                    ax.set_title(f'Arch {arch}, K={K}: shape mismatch', fontsize=8)
                    continue
                _render_policy_map(ax, q_np, Pi_np, q_opt_np_m, env_obj, goal_row,
                                   vmin, vmax, f'Arch {arch}, K={K}')

        fig.suptitle(
            f'{variant_name} — Policy Maps {mode.upper()}  '
            f'(U={num_unrolls}, universal scale [{vmin:.1f}, {vmax:.1f}])',
            fontsize=12, fontweight='bold',
        )
        plt.tight_layout()
        out_dir = results_dir / "visualizations"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{variant_name}_U{num_unrolls}_policy_maps_{suffix}_universal_scale.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Comprehensive 8-metric figure (4×2)
# ─────────────────────────────────────────────────────────────────────────────

def plot_comprehensive(variant_name, results_dir, num_unrolls=NUM_UNROLLS_SHOW):
    """4×2 comprehensive figure — all 8 metrics vs K, median±IQR.

    POG shown as non-squared (sqrt applied) so all policy metrics are on the
    same relative-L2-error scale.  Other metrics (Bellman, OG-q, OG-V) are
    already non-squared.
    """
    csv_path = results_dir / "all_experiments_results.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    df = df[df['success'] == True]
    df_u = df[df['num_unrolls'] == num_unrolls].copy()
    if len(df_u) == 0:
        return

    n_runs = df_u['run_idx'].nunique()

    specs = [
        # (col_train, col_test, title, ylabel, use_log, use_sci, sqrt)
        ('bellman_optimality_error_train', 'bellman_optimality_error_test',
         'Bellman OE (Normalized)', r'$\|\delta q\| / \|r+\gamma P_\pi q\|$', True, True, False),
        ('bellman_optimality_error_unnormalized_train', 'bellman_optimality_error_unnormalized_test',
         'Bellman OE (Unnormalized)', r'$\|\delta q\|$', True, False, False),
        ('policy_optimality_gap_joint_train', 'policy_optimality_gap_joint_test',
         r'POG Joint  $\frac{\|q_\pi - q^*\|}{\|q^*\|}$  (non-sq)',
         r'rel. L2', True, True, True),
        ('policy_optimality_gap_separate_train', 'policy_optimality_gap_separate_test',
         r'POG Separate  $\|\hat{q}_\pi - \hat{q}^*\|$  (non-sq)',
         r'rel. L2', False, False, True),
        ('optimality_gap_joint_train', 'optimality_gap_joint_test',
         r'OG-q Joint  $\frac{\|q - q^*\|}{\|q^*\|}$',
         r'rel. L2', True, False, False),
        ('optimality_gap_separate_train', 'optimality_gap_separate_test',
         r'OG-q Separate', r'rel. L2', False, False, False),
        ('optimality_gap_V_joint_train', 'optimality_gap_V_joint_test',
         r'OG-V Joint  $\frac{\|V - V^*\|}{\|V^*\|}$',
         r'rel. L2', False, False, False),
        ('optimality_gap_V_separate_train', 'optimality_gap_V_separate_test',
         r'OG-V Separate', r'rel. L2', False, False, False),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    for (row, col), (col_tr, col_te, title, ylabel, use_log, use_sci, do_sqrt) in zip(
        [(r, c) for r in range(4) for c in range(2)], specs
    ):
        _plot_metric_on_axis(axes[row, col], df_u, col_tr, col_te,
                             title, ylabel, use_log=use_log,
                             transform_sqrt=do_sqrt)

    fig.suptitle(
        f'Cliff Variant: {variant_name}  |  num_unrolls={num_unrolls},  n={n_runs} runs\n'
        'All policy metrics (POG) shown non-squared for comparability  |  median ± IQR',
        fontsize=11, fontweight='bold',
    )
    plt.tight_layout()

    out_dir = results_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{variant_name}_U{num_unrolls}_comprehensive_8metrics.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='+', default=None,
                        help='Subset of variants to process (default: all)')
    parser.add_argument('--unrolls', type=int, default=NUM_UNROLLS_SHOW,
                        help='Fixed num_unrolls for agreement/maps/comprehensive figures')
    args = parser.parse_args()

    variants = args.variants or ALL_VARIANTS
    num_unrolls = args.unrolls

    print("=" * 70)
    print("VISUALIZE CLIFF VARIATIONS")
    print("=" * 70)
    print(f"Variants:   {variants}")
    print(f"Num unrolls (fixed): {num_unrolls}")
    print()

    for var_name in variants:
        if var_name not in VARIANT_ENVS:
            print(f"[WARN] Unknown variant: {var_name}")
            continue

        results_dir = RESULTS_BASE / var_name
        if not results_dir.exists():
            print(f"[skip] {var_name}: results dir not found")
            continue

        train_type, test_type, goal_row_tr, goal_row_te = VARIANT_ENVS[var_name]

        print(f"\n{'─'*60}")
        print(f"Variant: {var_name}")

        print("  Computing optimal Q-values...")
        env_tr = _build_env(train_type)
        q_opt  = _get_opt_q(env_tr, goal_row_tr)

        env_te     = _build_env(test_type)
        q_opt_test = _get_opt_q(env_te, goal_row_te)

        print("  1. Agreement figure...")
        plot_agreement_figure(var_name, results_dir, num_unrolls=num_unrolls)

        print("  2. Policy maps (universal scale)...")
        plot_policy_maps(var_name, results_dir, q_opt, q_opt_test,
                         env_tr, env_te, goal_row_tr, goal_row_te,
                         num_unrolls=num_unrolls)

        print("  3. Comprehensive 8-metric figure...")
        plot_comprehensive(var_name, results_dir, num_unrolls=num_unrolls)

    print("\nDone.")


if __name__ == "__main__":
    main()
