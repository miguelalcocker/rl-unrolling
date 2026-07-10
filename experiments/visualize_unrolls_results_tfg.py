"""
Visualize Unrolls Experimental Results - Version 2
====================================================

Genera visualizaciones para cada num_unrolls basándose en las NUEVAS métricas del CSV.

MÉTRICAS ESPERADAS EN EL CSV (del script unrolls_experiments_analysis.py actualizado):
- bellman_optimality_error_train/test (normalizado)
- bellman_optimality_error_unnormalized_train/test
- optimality_gap_joint_train/test (usa q del modelo)
- optimality_gap_separate_train/test (usa q del modelo)
- policy_optimality_gap_joint_train/test (usa q_π, squared)
- policy_optimality_gap_separate_train/test (usa q_π, squared)

FIGURAS GENERADAS:
- unrolls{N}_v2_comprehensive_6metrics.png: 3x2 con todas las métricas
- unrolls{N}_v2_policy_maps_train_universal_scale.png: Policy maps TRAIN con escala universal
- unrolls{N}_v2_policy_maps_test_universal_scale.png: Policy maps TEST con escala universal

Author: Miguel Alcocer
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import matplotlib.ticker as ticker

from src.plots import plot_policy_and_value, ARCH_COLORS
from src.utils import get_optimal_q

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = Path("unrolls_results")
OUTPUT_DIR = RESULTS_DIR / "visualizations_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS_ARCH = ARCH_COLORS

# ============================================================================
# DATA LOADING
# ============================================================================

def load_results():
    """Load experimental results."""
    df = pd.read_csv(RESULTS_DIR / "all_experiments_results.csv")
    df = df[df['success'] == True]

    print(f"Loaded {len(df)} successful experiments")
    print(f"K values: {sorted(df['K'].unique())}")
    print(f"All num_unrolls values: {sorted(df['num_unrolls'].unique())}")
    print(f"Architectures: {sorted(df['architecture_type'].unique())}")
    print(f"Runs: {len(df['run_idx'].unique())}")

    # Filter only num_unrolls that have data in the NEW metrics
    # (the old data doesn't have these columns populated)
    required_col = 'bellman_optimality_error_train'
    if required_col in df.columns:
        valid_unrolls = []
        for u in df['num_unrolls'].unique():
            subset = df[df['num_unrolls'] == u]
            if subset[required_col].notna().sum() > 0:
                valid_unrolls.append(u)

        print(f"num_unrolls with NEW metrics: {sorted(valid_unrolls)}")

        # Filter DataFrame to only include valid unrolls
        df = df[df['num_unrolls'].isin(valid_unrolls)]
        print(f"Filtered to {len(df)} experiments with new metrics")

    return df


# ============================================================================
# HELPER FUNCTION: Generic metric plotting
# ============================================================================

def plot_metric_on_axis(ax, df_unrolls, metric_train, metric_test, title, ylabel,
                        use_log=True, use_scientific=False):
    """Plot a single metric on the given axis."""

    K_values = sorted(df_unrolls['K'].unique())
    architectures = sorted(df_unrolls['architecture_type'].unique())

    clean_formatter = ticker.FormatStrFormatter('%g')

    for arch in architectures:
        df_arch = df_unrolls[df_unrolls['architecture_type'] == arch]

        stats = {'train': {'m': [], 'p25': [], 'p75': []},
                 'test':  {'m': [], 'p25': [], 'p75': []}}

        for K in K_values:
            for mode, col in [('train', metric_train), ('test', metric_test)]:
                if col in df_arch.columns:
                    vals = df_arch[df_arch['K'] == K][col].values
                    if len(vals) > 0:
                        vals = vals[~np.isnan(vals)]
                        if len(vals) > 0:
                            if use_log:
                                vals = np.clip(vals, 1e-10, None)
                            stats[mode]['m'].append(np.median(vals))
                            stats[mode]['p25'].append(np.percentile(vals, 25))
                            stats[mode]['p75'].append(np.percentile(vals, 75))
                        else:
                            for k in stats[mode]: stats[mode][k].append(np.nan)
                    else:
                        for k in stats[mode]: stats[mode][k].append(np.nan)
                else:
                    for k in stats[mode]: stats[mode][k].append(np.nan)

        color = COLORS_ARCH[arch]
        ax.plot(K_values, stats['train']['m'], linestyle='-', marker='o',
               linewidth=2.5, markersize=8, color=color, label=f'Arch {arch} - Train')
        ax.fill_between(K_values, stats['train']['p25'], stats['train']['p75'],
                       alpha=0.15, color=color)

        ax.plot(K_values, stats['test']['m'], linestyle='--', marker='s',
               linewidth=2.5, markersize=8, color=color, label=f'Arch {arch} - Test')
        ax.fill_between(K_values, stats['test']['p25'], stats['test']['p75'],
                       alpha=0.15, color=color)

    ax.set_xlabel('Filter Order (K)', fontweight='bold', fontsize=11)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xticks(K_values)
    ax.grid(True, alpha=0.2, linestyle=':', which='major')

    if use_log:
        ax.set_yscale('log')
        if use_scientific:
            ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        else:
            ax.yaxis.set_major_formatter(clean_formatter)
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.5, 1.0)))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
    else:
        ax.set_ylim(bottom=-0.01)
        ax.yaxis.set_major_formatter(clean_formatter)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))

    ax.legend(fontsize=8, loc='best', framealpha=0.9)


# ============================================================================
# FIGURE: COMPREHENSIVE 6-METRIC PLOT (3x2)
# ============================================================================

def plot_comprehensive_6metrics(df, num_unrolls):
    """
    Plot all 6 metrics in a 3x2 comprehensive figure:
    - Row 1: Bellman Optimality Error (Normalized) | Bellman Optimality Error (Unnormalized)
    - Row 2: Policy Optimality Gap (Joint Norm) | Policy Optimality Gap (Separate Norm)
    - Row 3: Optimality Gap (Joint Norm) | Optimality Gap (Separate Norm)
    """

    df_unrolls = df[df['num_unrolls'] == num_unrolls].copy()

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    # Define metrics for each subplot
    metrics = [
        # Row 1: Bellman Optimality Error
        ('bellman_optimality_error_train', 'bellman_optimality_error_test',
         'Bellman Optimality Error (Normalized)',
         r'$\frac{\|q - (r + \gamma P_\pi q)\|}{\|r + \gamma P_\pi q\|}$',
         True, False),
        ('bellman_optimality_error_unnormalized_train', 'bellman_optimality_error_unnormalized_test',
         'Bellman Optimality Error (Unnormalized)',
         r'$\|q - (r + \gamma P_\pi q)\|$',
         True, False),

        # Row 2: Policy Optimality Gap (uses q_π via policy evaluation)
        ('policy_optimality_gap_joint_train', 'policy_optimality_gap_joint_test',
         'Policy Optimality Gap (Joint Norm)',
         r'$\left(\frac{\|q_\pi - q^*\|}{\|q^*\|}\right)^2$',
         True, True),
        ('policy_optimality_gap_separate_train', 'policy_optimality_gap_separate_test',
         'Policy Optimality Gap (Separate Norm)',
         r'$\left\|\frac{q_\pi}{\|q_\pi\|} - \frac{q^*}{\|q^*\|}\right\|^2$',
         False, False),

        # Row 3: Optimality Gap (uses q from model directly)
        ('optimality_gap_joint_train', 'optimality_gap_joint_test',
         'Optimality Gap (Joint Norm)',
         r'$\frac{\|q - q^*\|}{\|q^*\|}$',
         True, False),
        ('optimality_gap_separate_train', 'optimality_gap_separate_test',
         'Optimality Gap (Separate Norm)',
         r'$\left\|\frac{q}{\|q\|} - \frac{q^*}{\|q^*\|}\right\|$',
         False, False),
    ]

    for idx, (metric_train, metric_test, title, ylabel, use_log, use_scientific) in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        plot_metric_on_axis(ax, df_unrolls, metric_train, metric_test,
                           title, ylabel, use_log, use_scientific)

    n_runs = len(df['run_idx'].unique())
    fig.suptitle(f'Comprehensive Metrics Analysis (num_unrolls={num_unrolls})\n'
                f'(Median with 25th-75th percentile bands, n={n_runs} runs)',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'unrolls{num_unrolls}_v2_comprehensive_6metrics.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: unrolls{num_unrolls}_v2_comprehensive_6metrics.png")


# ============================================================================
# FIGURE 7: POLICY MAPS TRAIN WITH UNIVERSAL SCALE
# ============================================================================

def plot_policy_maps_train_universal_scale(df, num_unrolls, q_opt):
    """Plot policy maps for TRAIN with UNIVERSAL SCALE across all maps (same vmin/vmax)."""

    df_unrolls = df[df['num_unrolls'] == num_unrolls].copy()

    K_values = sorted(df_unrolls['K'].unique())

    # Create figure: rows = K values, columns = Arch 1, Optimal, Arch 2
    n_rows = len(K_values)
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # ========================================================================
    # COMPUTE GLOBAL MIN/MAX across ALL policies for this num_unrolls (TRAIN)
    # ========================================================================

    global_min = float('inf')
    global_max = float('-inf')

    # Include optimal Q-values
    q_opt_reshaped = q_opt.view(48, 4)
    V_opt = q_opt_reshaped.max(dim=1).values
    global_min = min(global_min, float(V_opt.min()))
    global_max = max(global_max, float(V_opt.max()))

    # Include all trained policies (Arch 1 and Arch 2)
    for K in K_values:
        for arch in [1, 2]:
            df_subset = df_unrolls[
                (df_unrolls['K'] == K) &
                (df_unrolls['architecture_type'] == arch) &
                (df_unrolls['policy_file'].notna())
            ]

            if len(df_subset) > 0:
                policy_file = df_subset.iloc[0]['policy_file']
                policy_path = RESULTS_DIR / policy_file

                if policy_path.exists():
                    data = np.load(policy_path)
                    q_train = torch.from_numpy(data['q_train']).float()
                    q_train_reshaped = q_train.view(48, 4)
                    V_train = q_train_reshaped.max(dim=1).values
                    global_min = min(global_min, float(V_train.min()))
                    global_max = max(global_max, float(V_train.max()))

    print(f"    Universal scale for TRAIN num_unrolls={num_unrolls}: vmin={global_min:.2f}, vmax={global_max:.2f}")

    # ========================================================================
    # PLOT ALL MAPS WITH UNIVERSAL SCALE
    # ========================================================================

    for i, K in enumerate(K_values):
        # Column 0: Architecture 1
        ax = axes[i, 0]
        df_subset = df_unrolls[
            (df_unrolls['K'] == K) &
            (df_unrolls['architecture_type'] == 1) &
            (df_unrolls['policy_file'].notna())
        ]

        if len(df_subset) > 0:
            policy_file = df_subset.iloc[0]['policy_file']
            policy_path = RESULTS_DIR / policy_file

            if policy_path.exists():
                data = np.load(policy_path)
                q_train = torch.from_numpy(data['q_train']).float()
                Pi_train = torch.from_numpy(data['Pi_train']).float()

                temp_fig = plot_policy_and_value(
                    q_train.view(48, 4),
                    Pi_train,
                    highlight_cliffs=True,
                    goal_row=3,
                    shape=(4, 12),
                    min_prob=0.02,
                    plot_all_trans=False,
                    vmin=global_min,
                    vmax=global_max
                )

                temp_fig.canvas.draw()
                img = np.frombuffer(temp_fig.canvas.buffer_rgba(), dtype=np.uint8)
                img = img.reshape(temp_fig.canvas.get_width_height()[::-1] + (4,))
                img = img[:, :, :3]

                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f'K={K}, Arch 1', fontweight='bold', fontsize=10)
                plt.close(temp_fig)
            else:
                ax.text(0.5, 0.5, 'Policy file\nnot found',
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.axis('off')

        # Column 1: Optimal Policy
        ax = axes[i, 1]

        q_opt_reshaped = q_opt.view(48, 4)
        greedy_actions = q_opt_reshaped.argmax(dim=1)
        Pi_opt = torch.zeros(48, 4)
        Pi_opt[torch.arange(48), greedy_actions] = 1.0

        temp_fig = plot_policy_and_value(
            q_opt_reshaped,
            Pi_opt,
            highlight_cliffs=True,
            goal_row=3,
            shape=(4, 12),
            min_prob=0.02,
            plot_all_trans=False,
            vmin=global_min,
            vmax=global_max
        )

        temp_fig.canvas.draw()
        img = np.frombuffer(temp_fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(temp_fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, :3]

        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'K={K}, Optimal', fontweight='bold', fontsize=10, color='green')
        plt.close(temp_fig)

        # Column 2: Architecture 2
        ax = axes[i, 2]
        df_subset = df_unrolls[
            (df_unrolls['K'] == K) &
            (df_unrolls['architecture_type'] == 2) &
            (df_unrolls['policy_file'].notna())
        ]

        if len(df_subset) > 0:
            policy_file = df_subset.iloc[0]['policy_file']
            policy_path = RESULTS_DIR / policy_file

            if policy_path.exists():
                data = np.load(policy_path)
                q_train = torch.from_numpy(data['q_train']).float()
                Pi_train = torch.from_numpy(data['Pi_train']).float()

                temp_fig = plot_policy_and_value(
                    q_train.view(48, 4),
                    Pi_train,
                    highlight_cliffs=True,
                    goal_row=3,
                    shape=(4, 12),
                    min_prob=0.02,
                    plot_all_trans=False,
                    vmin=global_min,
                    vmax=global_max
                )

                temp_fig.canvas.draw()
                img = np.frombuffer(temp_fig.canvas.buffer_rgba(), dtype=np.uint8)
                img = img.reshape(temp_fig.canvas.get_width_height()[::-1] + (4,))
                img = img[:, :, :3]

                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f'K={K}, Arch 2', fontweight='bold', fontsize=10)
                plt.close(temp_fig)
            else:
                ax.text(0.5, 0.5, 'Policy file\nnot found',
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.axis('off')

    fig.suptitle(f'Policy Maps (TRAIN, num_unrolls={num_unrolls}) - Universal Scale\n'
                f'(vmin={global_min:.1f}, vmax={global_max:.1f})',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'unrolls{num_unrolls}_v2_policy_maps_train_universal_scale.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: unrolls{num_unrolls}_v2_policy_maps_train_universal_scale.png")


# ============================================================================
# FIGURE 8b: POLICY MAPS TEST WITH UNIVERSAL SCALE
# ============================================================================

def plot_policy_maps_test_universal_scale(df, num_unrolls, q_opt_test):
    """Plot policy maps for TEST with UNIVERSAL SCALE across all maps (same vmin/vmax)."""

    df_unrolls = df[df['num_unrolls'] == num_unrolls].copy()

    K_values = sorted(df_unrolls['K'].unique())

    # Create figure: rows = K values, columns = Arch 1, Optimal, Arch 2
    n_rows = len(K_values)
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # ========================================================================
    # COMPUTE GLOBAL MIN/MAX across ALL policies for this num_unrolls (TEST)
    # ========================================================================

    global_min = float('inf')
    global_max = float('-inf')

    # Include optimal Q-values
    q_opt_reshaped = q_opt_test.view(48, 4)
    V_opt = q_opt_reshaped.max(dim=1).values
    global_min = min(global_min, float(V_opt.min()))
    global_max = max(global_max, float(V_opt.max()))

    # Include all trained policies (Arch 1 and Arch 2)
    for K in K_values:
        for arch in [1, 2]:
            df_subset = df_unrolls[
                (df_unrolls['K'] == K) &
                (df_unrolls['architecture_type'] == arch) &
                (df_unrolls['policy_file'].notna())
            ]

            if len(df_subset) > 0:
                policy_file = df_subset.iloc[0]['policy_file']
                policy_path = RESULTS_DIR / policy_file

                if policy_path.exists():
                    data = np.load(policy_path)
                    q_test = torch.from_numpy(data['q_test']).float()
                    q_test_reshaped = q_test.view(48, 4)
                    V_test = q_test_reshaped.max(dim=1).values
                    global_min = min(global_min, float(V_test.min()))
                    global_max = max(global_max, float(V_test.max()))

    print(f"    Universal scale for TEST num_unrolls={num_unrolls}: vmin={global_min:.2f}, vmax={global_max:.2f}")

    # ========================================================================
    # PLOT ALL MAPS WITH UNIVERSAL SCALE
    # ========================================================================

    for i, K in enumerate(K_values):
        # Column 0: Architecture 1
        ax = axes[i, 0]
        df_subset = df_unrolls[
            (df_unrolls['K'] == K) &
            (df_unrolls['architecture_type'] == 1) &
            (df_unrolls['policy_file'].notna())
        ]

        if len(df_subset) > 0:
            policy_file = df_subset.iloc[0]['policy_file']
            policy_path = RESULTS_DIR / policy_file

            if policy_path.exists():
                data = np.load(policy_path)
                q_test = torch.from_numpy(data['q_test']).float()
                Pi_test = torch.from_numpy(data['Pi_test']).float()

                temp_fig = plot_policy_and_value(
                    q_test.view(48, 4),
                    Pi_test,
                    highlight_cliffs=True,
                    goal_row=0,  # Mirrored environment
                    shape=(4, 12),
                    min_prob=0.02,
                    plot_all_trans=False,
                    vmin=global_min,
                    vmax=global_max
                )

                temp_fig.canvas.draw()
                img = np.frombuffer(temp_fig.canvas.buffer_rgba(), dtype=np.uint8)
                img = img.reshape(temp_fig.canvas.get_width_height()[::-1] + (4,))
                img = img[:, :, :3]

                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f'K={K}, Arch 1', fontweight='bold', fontsize=10)
                plt.close(temp_fig)
            else:
                ax.text(0.5, 0.5, 'Policy file\nnot found',
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.axis('off')

        # Column 1: Optimal Policy
        ax = axes[i, 1]

        q_opt_reshaped = q_opt_test.view(48, 4)
        greedy_actions = q_opt_reshaped.argmax(dim=1)
        Pi_opt = torch.zeros(48, 4)
        Pi_opt[torch.arange(48), greedy_actions] = 1.0

        temp_fig = plot_policy_and_value(
            q_opt_reshaped,
            Pi_opt,
            highlight_cliffs=True,
            goal_row=0,  # Mirrored environment
            shape=(4, 12),
            min_prob=0.02,
            plot_all_trans=False,
            vmin=global_min,
            vmax=global_max
        )

        temp_fig.canvas.draw()
        img = np.frombuffer(temp_fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(temp_fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, :3]

        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'K={K}, Optimal', fontweight='bold', fontsize=10, color='green')
        plt.close(temp_fig)

        # Column 2: Architecture 2
        ax = axes[i, 2]
        df_subset = df_unrolls[
            (df_unrolls['K'] == K) &
            (df_unrolls['architecture_type'] == 2) &
            (df_unrolls['policy_file'].notna())
        ]

        if len(df_subset) > 0:
            policy_file = df_subset.iloc[0]['policy_file']
            policy_path = RESULTS_DIR / policy_file

            if policy_path.exists():
                data = np.load(policy_path)
                q_test = torch.from_numpy(data['q_test']).float()
                Pi_test = torch.from_numpy(data['Pi_test']).float()

                temp_fig = plot_policy_and_value(
                    q_test.view(48, 4),
                    Pi_test,
                    highlight_cliffs=True,
                    goal_row=0,  # Mirrored environment
                    shape=(4, 12),
                    min_prob=0.02,
                    plot_all_trans=False,
                    vmin=global_min,
                    vmax=global_max
                )

                temp_fig.canvas.draw()
                img = np.frombuffer(temp_fig.canvas.buffer_rgba(), dtype=np.uint8)
                img = img.reshape(temp_fig.canvas.get_width_height()[::-1] + (4,))
                img = img[:, :, :3]

                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f'K={K}, Arch 2', fontweight='bold', fontsize=10)
                plt.close(temp_fig)
            else:
                ax.text(0.5, 0.5, 'Policy file\nnot found',
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.axis('off')

    fig.suptitle(f'Policy Maps (TEST - Mirrored, num_unrolls={num_unrolls}) - Universal Scale\n'
                f'(vmin={global_min:.1f}, vmax={global_max:.1f})',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'unrolls{num_unrolls}_v2_policy_maps_test_universal_scale.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: unrolls{num_unrolls}_v2_policy_maps_test_universal_scale.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("VISUALIZING UNROLLS EXPERIMENTAL RESULTS - V2")
    print("="*80)
    print()

    print("Loading experimental data...")
    df = load_results()
    print()

    # Compute optimal Q-values once (reuse for all num_unrolls)
    print("Computing optimal Q-values...")
    q_opt = get_optimal_q(
        mirror_env=False,
        use_logger=False,
        max_eval_iters=50,
        max_epochs=50
    )

    q_opt_test = get_optimal_q(
        mirror_env=True,
        use_logger=False,
        max_eval_iters=50,
        max_epochs=50
    )
    print(f"✓ Optimal Q computed")
    print(f"  Train env: ||q*|| = {torch.norm(q_opt):.2f}")
    print(f"  Test env:  ||q*|| = {torch.norm(q_opt_test):.2f}")
    print()

    num_unrolls_values = sorted(df['num_unrolls'].unique())

    print("Generating visualizations for each num_unrolls...")
    print()

    for num_unrolls in num_unrolls_values:
        print(f"Processing num_unrolls={num_unrolls}:")

        print(f"  1. Comprehensive 6-Metrics Figure (3x2)...")
        plot_comprehensive_6metrics(df, num_unrolls)

        print(f"  2. Policy Maps TRAIN with Universal Scale...")
        plot_policy_maps_train_universal_scale(df, num_unrolls, q_opt)

        print(f"  3. Policy Maps TEST with Universal Scale...")
        plot_policy_maps_test_universal_scale(df, num_unrolls, q_opt_test)

        print()

    print("="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print()
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print()
    print("Generated files (for each num_unrolls):")
    print("  - unrolls{N}_v2_comprehensive_6metrics.png")
    print("  - unrolls{N}_v2_policy_maps_train_universal_scale.png")
    print("  - unrolls{N}_v2_policy_maps_test_universal_scale.png")
    print()
    print(f"Total files: {len(num_unrolls_values) * 3} files")
    print("="*80)


if __name__ == "__main__":
    main()
