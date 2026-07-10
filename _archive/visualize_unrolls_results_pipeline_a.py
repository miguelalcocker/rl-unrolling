"""
Visualize Unrolls Experimental Results - Version 2
====================================================

Genera visualizaciones para cada num_unrolls basándose en las NUEVAS métricas del CSV.

MÉTRICAS ESPERADAS EN EL CSV (del script unrolls_experiments_analysis.py actualizado):
- bellman_optimality_error_train/test (normalizado)
- bellman_optimality_error_unnormalized_train/test
- optimality_gap_joint_train/test (usa q del modelo)
- optimality_gap_separate_train/test (usa q del modelo)
- optimality_gap_V_joint_train/test (usa V = max_a q(s,a) del modelo)
- optimality_gap_V_separate_train/test (usa V = max_a q(s,a) del modelo)
- policy_optimality_gap_joint_train/test (usa q_π, squared)
- policy_optimality_gap_separate_train/test (usa q_π, squared)

FIGURAS GENERADAS:
- unrolls{N}_v2_comprehensive_8metrics.png: 4x2 con todas las métricas
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

from src.plots import plot_policy_and_value
from src.utils import get_optimal_q

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = Path("unrolls_results_v3_3")
OUTPUT_DIR = RESULTS_DIR / "visualizations_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Professional color scheme
COLORS_ARCH = {1: '#0173B2', 2: '#DE8F05'}  # Azul y Naranja

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

    # Check if there's any valid (non-NaN) data before applying log scale
    has_data = any(
        not np.isnan(v)
        for mode_stats in [stats['train'], stats['test']]
        for v in mode_stats['m']
    )

    if use_log and has_data:
        ax.set_yscale('log')
        if use_scientific:
            ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        else:
            ax.yaxis.set_major_formatter(clean_formatter)
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.5, 1.0)))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
    elif not has_data:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='gray')
    else:
        ax.set_ylim(bottom=-0.01)
        ax.yaxis.set_major_formatter(clean_formatter)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))

    ax.legend(fontsize=8, loc='best', framealpha=0.9)


# ============================================================================
# FIGURE: COMPREHENSIVE 8-METRIC PLOT (4x2)
# ============================================================================

def plot_comprehensive_8metrics(df, num_unrolls):
    """
    Plot all 8 metrics in a 4x2 comprehensive figure:
    - Row 1: Bellman Optimality Error (Normalized) | Bellman Optimality Error (Unnormalized)
    - Row 2: Policy Optimality Gap (Joint Norm) | Policy Optimality Gap (Separate Norm)
    - Row 3: Optimality Gap q (Joint Norm) | Optimality Gap q (Separate Norm)
    - Row 4: Optimality Gap V (Joint Norm) | Optimality Gap V (Separate Norm)
    """

    df_unrolls = df[df['num_unrolls'] == num_unrolls].copy()

    fig, axes = plt.subplots(4, 2, figsize=(16, 24))

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

        # Row 3: Optimality Gap q (uses q from model directly)
        ('optimality_gap_joint_train', 'optimality_gap_joint_test',
         'Optimality Gap q (Joint Norm)',
         r'$\frac{\|q - q^*\|}{\|q^*\|}$',
         True, False),
        ('optimality_gap_separate_train', 'optimality_gap_separate_test',
         'Optimality Gap q (Separate Norm)',
         r'$\left\|\frac{q}{\|q\|} - \frac{q^*}{\|q^*\|}\right\|$',
         False, False),

        # Row 4: Optimality Gap V (uses V = max_a q(s,a) from model)
        ('optimality_gap_V_joint_train', 'optimality_gap_V_joint_test',
         'Optimality Gap V (Joint Norm)',
         r'$\frac{\|V - V^*\|}{\|V^*\|}$',
         True, False),
        ('optimality_gap_V_separate_train', 'optimality_gap_V_separate_test',
         'Optimality Gap V (Separate Norm)',
         r'$\left\|\frac{V}{\|V\|} - \frac{V^*}{\|V^*\|}\right\|$',
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
    plt.savefig(OUTPUT_DIR / f'unrolls{num_unrolls}_v2_comprehensive_8metrics.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: unrolls{num_unrolls}_v2_comprehensive_8metrics.png")


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
    # Use policy_file_train if available, otherwise fall back to policy_file
    for K in K_values:
        for arch in [1, 2]:
            # First try policy_file_train (median based on train metrics)
            policy_file = None
            if 'policy_file_train' in df_unrolls.columns:
                df_subset = df_unrolls[
                    (df_unrolls['K'] == K) &
                    (df_unrolls['architecture_type'] == arch) &
                    (df_unrolls['policy_file_train'].notna())
                ]
                if len(df_subset) > 0:
                    policy_file = df_subset.iloc[0]['policy_file_train']

            # Fall back to policy_file if policy_file_train not found
            if policy_file is None:
                df_subset = df_unrolls[
                    (df_unrolls['K'] == K) &
                    (df_unrolls['architecture_type'] == arch) &
                    (df_unrolls['policy_file'].notna())
                ]
                if len(df_subset) > 0:
                    policy_file = df_subset.iloc[0]['policy_file']

            if policy_file is not None:
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

    # Helper function to get policy file for TRAIN (prefers policy_file_train)
    def get_train_policy_file(df_unrolls, K, arch):
        """Get policy file for TRAIN visualization (prefers policy_file_train)."""
        policy_file = None
        if 'policy_file_train' in df_unrolls.columns:
            df_subset = df_unrolls[
                (df_unrolls['K'] == K) &
                (df_unrolls['architecture_type'] == arch) &
                (df_unrolls['policy_file_train'].notna())
            ]
            if len(df_subset) > 0:
                policy_file = df_subset.iloc[0]['policy_file_train']

        if policy_file is None:
            df_subset = df_unrolls[
                (df_unrolls['K'] == K) &
                (df_unrolls['architecture_type'] == arch) &
                (df_unrolls['policy_file'].notna())
            ]
            if len(df_subset) > 0:
                policy_file = df_subset.iloc[0]['policy_file']

        return policy_file

    for i, K in enumerate(K_values):
        # Column 0: Architecture 1
        ax = axes[i, 0]
        policy_file = get_train_policy_file(df_unrolls, K, 1)

        if policy_file is not None:
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
        policy_file = get_train_policy_file(df_unrolls, K, 2)

        if policy_file is not None:
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
# HELPER: Get policy file for a given configuration
# ============================================================================

def _get_policy_file(df_unrolls, K, arch, prefer_train=False):
    """Get policy .npz path for a given K and architecture."""
    policy_file = None
    if prefer_train and 'policy_file_train' in df_unrolls.columns:
        sub = df_unrolls[
            (df_unrolls['K'] == K) &
            (df_unrolls['architecture_type'] == arch) &
            (df_unrolls['policy_file_train'].notna())
        ]
        if len(sub) > 0:
            policy_file = sub.iloc[0]['policy_file_train']

    if policy_file is None:
        sub = df_unrolls[
            (df_unrolls['K'] == K) &
            (df_unrolls['architecture_type'] == arch) &
            (df_unrolls['policy_file'].notna())
        ]
        if len(sub) > 0:
            policy_file = sub.iloc[0]['policy_file']

    if policy_file is not None:
        path = RESULTS_DIR / policy_file
        if path.exists():
            return path
    return None


# ============================================================================
# FIGURE: Q-VALUE MATRIX HEATMAPS (48 x 4)
# ============================================================================

def plot_q_matrix_heatmaps(df, num_unrolls, q_opt, q_opt_test):
    """
    Plot q(s,a) matrices as (48 x 4) heatmaps for Arch 1, Optimal, Arch 2.

    For each num_unrolls: one figure for TRAIN and one for TEST.
    Rows = K values, Columns = [Arch 1, Optimal, Arch 2].
    Each cell is a (48, 4) heatmap showing q(s, a) for all state-action pairs.
    """
    ACTION_LABELS = ['UP', 'RIGHT', 'DOWN', 'LEFT']

    df_unrolls = df[df['num_unrolls'] == num_unrolls].copy()
    K_values = sorted(df_unrolls['K'].unique())

    for mode, q_optimal, goal_row in [('train', q_opt, 3), ('test', q_opt_test, 0)]:
        n_rows = len(K_values)
        n_cols = 3  # Arch 1, Optimal, Arch 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        q_opt_mat = q_optimal.view(48, 4).detach().cpu().numpy()

        # Compute universal scale across all panels
        global_min = float(q_opt_mat.min())
        global_max = float(q_opt_mat.max())

        for K in K_values:
            for arch in [1, 2]:
                path = _get_policy_file(df_unrolls, K, arch, prefer_train=(mode == 'train'))
                if path is not None:
                    data = np.load(path)
                    q_vec = data[f'q_{mode}']
                    q_mat = q_vec.reshape(48, 4)
                    global_min = min(global_min, float(q_mat.min()))
                    global_max = max(global_max, float(q_mat.max()))

        # Plot
        for i, K in enumerate(K_values):
            # Arch 1
            ax = axes[i, 0]
            path = _get_policy_file(df_unrolls, K, 1, prefer_train=(mode == 'train'))
            if path is not None:
                data = np.load(path)
                q_mat = data[f'q_{mode}'].reshape(48, 4)
                im = ax.imshow(q_mat, aspect='auto', cmap='viridis', vmin=global_min, vmax=global_max)
                ax.set_ylabel(f'K={K}\nState', fontweight='bold', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Arch 1' if i == 0 else '', fontweight='bold', fontsize=11)
            ax.set_xticks(range(4))
            ax.set_xticklabels(ACTION_LABELS if i == n_rows - 1 else [], fontsize=8)

            # Optimal
            ax = axes[i, 1]
            im = ax.imshow(q_opt_mat, aspect='auto', cmap='viridis', vmin=global_min, vmax=global_max)
            ax.set_title(f'Optimal' if i == 0 else '', fontweight='bold', fontsize=11, color='green')
            ax.set_xticks(range(4))
            ax.set_xticklabels(ACTION_LABELS if i == n_rows - 1 else [], fontsize=8)
            ax.set_yticks([])

            # Arch 2
            ax = axes[i, 2]
            path = _get_policy_file(df_unrolls, K, 2, prefer_train=(mode == 'train'))
            if path is not None:
                data = np.load(path)
                q_mat = data[f'q_{mode}'].reshape(48, 4)
                im = ax.imshow(q_mat, aspect='auto', cmap='viridis', vmin=global_min, vmax=global_max)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Arch 2' if i == 0 else '', fontweight='bold', fontsize=11)
            ax.set_xticks(range(4))
            ax.set_xticklabels(ACTION_LABELS if i == n_rows - 1 else [], fontsize=8)
            ax.set_yticks([])

        env_label = 'TRAIN' if mode == 'train' else 'TEST (Mirrored)'
        fig.suptitle(
            f'Q-value Matrix q(s,a) — {env_label} (Unrolls={num_unrolls})\n'
            f'Rows=states (0..47), Cols=actions | Universal scale [{global_min:.1f}, {global_max:.1f}]',
            fontsize=13, fontweight='bold')

        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='q(s, a)')

        fname = f'unrolls{num_unrolls}_v2_q_matrix_{mode}.png'
        plt.savefig(OUTPUT_DIR / fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {fname}")


# ============================================================================
# FIGURE: Q-VALUE SPATIAL MAPS PER ACTION (on the 4x12 grid)
# ============================================================================

def plot_q_spatial_per_action(df, num_unrolls, q_opt, q_opt_test):
    """
    Plot q(s,a) as spatial heatmaps on the 4x12 grid, one sub-plot per action.

    For each num_unrolls: one figure for TRAIN and one for TEST.
    Rows = K values × {Arch 1, Optimal, Arch 2}
    Columns = 4 actions (UP, RIGHT, DOWN, LEFT)
    """
    ACTION_NAMES = ['UP', 'RIGHT', 'DOWN', 'LEFT']

    df_unrolls = df[df['num_unrolls'] == num_unrolls].copy()
    K_values = sorted(df_unrolls['K'].unique())

    for mode, q_optimal, goal_row in [('train', q_opt, 3), ('test', q_opt_test, 0)]:
        q_opt_mat = q_optimal.view(48, 4).detach().cpu().numpy()  # (48, 4)

        # Compute universal scale
        global_min = float(q_opt_mat.min())
        global_max = float(q_opt_mat.max())

        for K in K_values:
            for arch in [1, 2]:
                path = _get_policy_file(df_unrolls, K, arch, prefer_train=(mode == 'train'))
                if path is not None:
                    data = np.load(path)
                    q_mat = data[f'q_{mode}'].reshape(48, 4)
                    global_min = min(global_min, float(q_mat.min()))
                    global_max = max(global_max, float(q_mat.max()))

        cliff_cells = [(goal_row, c) for c in range(1, 11)]

        # Layout: 3 rows per K (Arch1, Optimal, Arch2) × 4 action columns
        n_row_groups = len(K_values)
        rows_per_group = 3  # Arch 1, Optimal, Arch 2
        n_rows = n_row_groups * rows_per_group
        n_cols = 4  # actions

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3.2 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        arch_labels = ['Arch 1', 'Optimal', 'Arch 2']
        arch_colors = [COLORS_ARCH[1], 'green', COLORS_ARCH[2]]

        for ki, K in enumerate(K_values):
            # Collect q matrices for this K: [Arch1, Optimal, Arch2]
            q_matrices = [None, q_opt_mat, None]

            path_1 = _get_policy_file(df_unrolls, K, 1, prefer_train=(mode == 'train'))
            if path_1 is not None:
                q_matrices[0] = np.load(path_1)[f'q_{mode}'].reshape(48, 4)

            path_2 = _get_policy_file(df_unrolls, K, 2, prefer_train=(mode == 'train'))
            if path_2 is not None:
                q_matrices[2] = np.load(path_2)[f'q_{mode}'].reshape(48, 4)

            for ri, (q_mat, label, color) in enumerate(zip(q_matrices, arch_labels, arch_colors)):
                row_idx = ki * rows_per_group + ri

                for ai in range(4):
                    ax = axes[row_idx, ai]

                    if q_mat is not None:
                        # Reshape action slice to grid: q(s, a) for action ai
                        q_grid = q_mat[:, ai].reshape(4, 12)

                        # Mask cliff cells
                        q_masked = np.ma.array(q_grid)
                        for (cr, cc) in cliff_cells:
                            q_masked[cr, cc] = np.ma.masked

                        im = ax.imshow(q_masked, cmap='viridis', vmin=global_min, vmax=global_max)

                        # Mark cliffs
                        for (cr, cc) in cliff_cells:
                            ax.add_patch(plt.Rectangle((cc - 0.5, cr - 0.5), 1, 1, color='black'))
                    else:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                                transform=ax.transAxes, fontsize=9)

                    ax.set_xticks([])
                    ax.set_yticks([])

                    # Column headers (top row only)
                    if row_idx == 0:
                        ax.set_title(ACTION_NAMES[ai], fontweight='bold', fontsize=11)

                    # Row labels (left column only)
                    if ai == 0:
                        k_prefix = f'K={K}, ' if ri == 0 else ''
                        ax.set_ylabel(f'{k_prefix}{label}', fontweight='bold',
                                      fontsize=9, color=color)

            # Separator line between K groups
            if ki < n_row_groups - 1:
                sep_y = (ki * rows_per_group + rows_per_group) / n_rows
                fig.add_artist(plt.Line2D([0.05, 0.95], [1 - sep_y, 1 - sep_y],
                                          transform=fig.transFigure, color='gray',
                                          linewidth=0.5, linestyle='--'))

        env_label = 'TRAIN' if mode == 'train' else 'TEST (Mirrored)'
        fig.suptitle(
            f'Q-value Spatial Maps per Action — {env_label} (Unrolls={num_unrolls})\n'
            f'Universal scale [{global_min:.1f}, {global_max:.1f}]',
            fontsize=14, fontweight='bold')

        # Use a dedicated colorbar axis to avoid overlap
        fig.subplots_adjust(right=0.92, hspace=0.3, wspace=0.08)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='q(s, a)')

        fname = f'unrolls{num_unrolls}_v2_q_spatial_per_action_{mode}.png'
        plt.savefig(OUTPUT_DIR / fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {fname}")


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

        print(f"  1. Comprehensive 8-Metrics Figure (4x2)...")
        plot_comprehensive_8metrics(df, num_unrolls)

        print(f"  2. Policy Maps TRAIN with Universal Scale...")
        plot_policy_maps_train_universal_scale(df, num_unrolls, q_opt)

        print(f"  3. Policy Maps TEST with Universal Scale...")
        plot_policy_maps_test_universal_scale(df, num_unrolls, q_opt_test)

        print(f"  4. Q-value Matrix Heatmaps (48x4)...")
        plot_q_matrix_heatmaps(df, num_unrolls, q_opt, q_opt_test)

        print(f"  5. Q-value Spatial Maps per Action...")
        plot_q_spatial_per_action(df, num_unrolls, q_opt, q_opt_test)

        print()

    print("="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print()
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print()
    print("Generated files (for each num_unrolls):")
    print("  - unrolls{N}_v2_comprehensive_8metrics.png")
    print("  - unrolls{N}_v2_policy_maps_train_universal_scale.png")
    print("  - unrolls{N}_v2_policy_maps_test_universal_scale.png")
    print("  - unrolls{N}_v2_q_matrix_train.png")
    print("  - unrolls{N}_v2_q_matrix_test.png")
    print("  - unrolls{N}_v2_q_spatial_per_action_train.png")
    print("  - unrolls{N}_v2_q_spatial_per_action_test.png")
    print()
    print(f"Total files: {len(num_unrolls_values) * 7} files")
    print("="*80)


if __name__ == "__main__":
    main()
