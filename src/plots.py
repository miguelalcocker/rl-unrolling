"""Visualization functions for BellNet.

This module contains functions for plotting policies, values, 
and filter coefficients for analysis and debugging.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from typing import Tuple, Optional, List
import torch

# Canonical architecture colors — import these in all visualization scripts
# to ensure consistent appearance across every TFG figure.
ARCH_COLORS = {1: '#0173B2', 2: '#DE8F05'}


def plot_metric_vs_K(ax, df, col_tr, col_te, title, ylabel,
                     use_log=False, use_sci=False, transform_sqrt=False):
    """Plot train+test metric vs filter order K with median ± IQR bands.

    Shared by visualize_unrolls_results_tfg.py and visualize_cliff_variations.py.

    Args:
        transform_sqrt: Apply sqrt to stored-squared values (e.g. POG stored squared).
        use_log: Log scale on y-axis; clips values to 1e-10 before plotting.
        use_sci: Scientific-notation formatter on log y-axis ticks.
    """
    K_values = sorted(df['K'].unique())

    for arch in [1, 2]:
        color = ARCH_COLORS[arch]
        for mode, ls, mk, col in [('Train', '-', 'o', col_tr), ('Test', '--', 's', col_te)]:
            if col not in df.columns:
                continue
            med_vals, p25_vals, p75_vals = [], [], []
            for K in K_values:
                v = df[(df['architecture_type'] == arch) & (df['K'] == K)][col].dropna().values
                if transform_sqrt:
                    v = np.sqrt(np.maximum(v, 0.0))
                if use_log:
                    v = np.clip(v, 1e-10, None)
                med_vals.append(np.median(v)         if len(v) else np.nan)
                p25_vals.append(np.percentile(v, 25) if len(v) else np.nan)
                p75_vals.append(np.percentile(v, 75) if len(v) else np.nan)
            ax.plot(K_values, np.array(med_vals), ls, marker=mk, linewidth=2.5, markersize=8,
                    color=color, label=f'Arch {arch} — {mode}')
            ax.fill_between(K_values, np.array(p25_vals), np.array(p75_vals),
                            alpha=0.15, color=color)

    ax.set_xlabel('Filter Order K', fontweight='bold', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_xticks(K_values)
    ax.grid(True, alpha=0.2, linestyle=':')
    ax.legend(fontsize=8, framealpha=0.9)

    fmt = ticker.FormatStrFormatter('%g')
    if use_log:
        ax.set_yscale('log')
        if use_sci:
            ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        else:
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.5, 1.0)))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
    else:
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))


def plot_policy_and_value(q: torch.Tensor, Pi: torch.Tensor,
                         highlight_cliffs: bool = True, goal_row: int = 3,
                         shape: Tuple[int, int] = (4, 12), min_prob: float = 0.02,
                         plot_all_trans: bool = False, vmin: float = None, vmax: float = None,
                         cliff_cells: Optional[List[Tuple[int, int]]] = None,
                         goal_cell: Optional[Tuple[int, int]] = None,
                         start_cell: Optional[Tuple[int, int]] = None,
                         show_colorbar: bool = True,
                         show_title: bool = True) -> None:
    """Plot policy and value function for grid world environment.

    Args:
        q: Q-values tensor
        Pi: Policy tensor
        highlight_cliffs: Whether to highlight cliff states
        goal_row: Row containing the goal state (used only when cliff_cells/goal_cell are None)
        shape: Grid shape (rows, cols)
        min_prob: Minimum probability threshold for plotting transitions
        plot_all_trans: Whether to plot all transitions or just policy
        vmin: Minimum value for colormap (optional)
        vmax: Maximum value for colormap (optional)
        cliff_cells: Explicit list of (row, col) cliff positions. If None, derived from goal_row.
        goal_cell: Explicit (row, col) goal position. If None, derived from goal_row.
        show_colorbar: Whether to add a colorbar (set False when embedding into outer figure).
        show_title: Whether to show the diagnostic title (set False when embedding).
    """
    q = q.detach().cpu().numpy()
    Pi = Pi.detach().cpu().numpy()
    nS, nA = Pi.shape
    rows, cols = shape

    # Cliff and goal states — use explicit values if provided, else legacy formula
    if cliff_cells is None:
        cliff_cells = [(goal_row, c) for c in range(1, 11)]
    if goal_cell is None:
        goal_cell = (goal_row, 11)

    assert nS == rows * cols, "State count does not match grid shape"

    V = q.max(axis=1).reshape(rows, cols)
    greedy_actions = Pi.argmax(axis=1).reshape(rows, cols)

    V_masked = V.copy()
    if highlight_cliffs:
        for (r, c) in cliff_cells:
            V_masked[r, c] = float('nan')  # Will be ignored in imshow

    V_masked = np.ma.masked_invalid(V_masked)

    action_arrows = {
        0: (0, 1),
        1: (1, 0),
        2: (0, -1),
        3: (-1, 0),
    }

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(V_masked, cmap="viridis", vmin=vmin, vmax=vmax)

    # Arrow rendering parameters
    max_lw = 3.5           # maximum line width
    min_lw = 1           # minimum line width
    for s in range(nS):
        r, c = divmod(s, cols)

        # Skip rendering for cliff and goal cells
        if highlight_cliffs and (r, c) in cliff_cells:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="black"))
            continue

        if (r, c) == goal_cell:
            continue

        if plot_all_trans:
            for a, prob in enumerate(Pi[s]):
                if prob < min_prob:
                    continue
                dx, dy = action_arrows[a]
                ax.arrow(c, r, dx * 0.3, -dy * 0.3, linewidth=min_lw + (max_lw - min_lw) * prob,
                    alpha=prob, head_width=0.2, head_length=0.2, fc="white", ec="white")
        else:
            a = greedy_actions[r, c]
            dx, dy = action_arrows[a]
            ax.arrow(c, r, dx * 0.3, -dy * 0.3, head_width=0.2, head_length=0.2, fc='white', ec='white')

    # Mark goal cell with a gold star
    gr, gc = goal_cell
    ax.add_patch(plt.Rectangle((gc - 0.5, gr - 0.5), 1, 1,
                               facecolor='gold', edgecolor='black', linewidth=1.5,
                               alpha=0.85, zorder=3))
    ax.plot(gc, gr, marker='*', markersize=14, color='black',
            markerfacecolor='black', markeredgewidth=0, zorder=4)

    # Mark start cell with a green circle (skip if same as goal)
    if start_cell is not None and tuple(start_cell) != tuple(goal_cell):
        sr, sc = start_cell
        ax.add_patch(plt.Circle((sc, sr), 0.38,
                                facecolor='limegreen', edgecolor='darkgreen',
                                linewidth=1.5, alpha=0.85, zorder=3))
        ax.text(sc, sr, 'S', ha='center', va='center',
                fontsize=11, color='white', fontweight='bold', zorder=4)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if show_title:
        ax.set_title("Mean Max Trans Prob = {:.3f}".format(Pi.max(axis=1).mean()))
    if show_colorbar:
        fig.colorbar(im, ax=ax, label='V(s)')

    return fig


def plot_Pi(Pi, figsize=(12, 4), title='Prob'):
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(Pi, aspect='auto')
    fig.colorbar(cax, ax=ax)

    ax.set_xlabel("Next (a')")
    ax.set_ylabel("Current (s)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_Pi_train(list_Pi, ncols=5, freq_plots=10, figsize_per_plot=(12, 4)):
    num = len(list_Pi)
    nrows = int(np.ceil(num / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_plot[0]*ncols, figsize_per_plot[1]*nrows))

    # Flatten axes array for easy indexing
    axes = np.array(axes).reshape(-1)

    for i, Pi in enumerate(list_Pi):
        ax = axes[i]

        cax = ax.imshow(Pi, aspect='auto')
        fig.colorbar(cax, ax=ax)
        ax.set_title(fr'Prob (step {i*freq_plots})')
        ax.set_xlabel("Next (a')")
        ax.set_ylabel("Current (s)")

    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    return fig


def plot_filter_coefs(h_values):
    fig = plt.figure()
    plt.stem(range(h_values.size), h_values)
    plt.title("Shared filter coefficients h")
    plt.xlabel("k")
    plt.ylabel("h[k]")
    plt.grid(True)
    plt.tight_layout()
    return fig
            