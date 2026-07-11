"""Violin plot utilities for BO round analysis."""

import numpy as np
import matplotlib.pyplot as plt
from .defaults import PLOT_DEFAULTS


def plot_round_violin(round_values, round_labels=None, cum_max=None,
                      xlabel='Round', ylabel='Value',
                      shade_regions=None, save_path=None,
                      figsize=(10, 6), dpi=300,
                      fontsize_title=16, fontsize_label=14,
                      fontsize_tick=12, title=None):
    """
    Plot violin plots of values across BO rounds with optional cumulative max.

    Parameters
    ----------
    round_values : list of array-like
        Values for each round.
    round_labels : list of str or None
        Labels for each round. Default: ['R1', 'R2', ...].
    cum_max : array-like or None
        Cumulative maximum line to overlay.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    shade_regions : list of dict or None
        Each dict has 'ymin', 'ymax', 'color', 'alpha', 'label' for axhspan shading.
    save_path : str or None
        Path to save the figure.
    figsize : tuple
        Figure size.
    dpi : int
        DPI for saving.
    fontsize_title, fontsize_label, fontsize_tick : int
        Font sizes.
    title : str or None
        Figure title.

    Returns
    -------
    fig, ax
    """
    n_rounds = len(round_values)
    if round_labels is None:
        round_labels = [f'R{i+1}' for i in range(n_rounds)]

    colors = PLOT_DEFAULTS['round_colors']

    fig, ax = plt.subplots(figsize=figsize)

    # Shade regions
    if shade_regions is not None:
        for region in shade_regions:
            ax.axhspan(
                region['ymin'], region['ymax'],
                color=region.get('color', 'lightblue'),
                alpha=region.get('alpha', 0.2),
                label=region.get('label', ''),
            )

    # Violin plots
    positions = list(range(1, n_rounds + 1))
    parts = ax.violinplot(round_values, positions=positions, showmeans=True, showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        color = colors[i % len(colors)]
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    # Overlay individual points
    for i, vals in enumerate(round_values):
        color = colors[i % len(colors)]
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, size=len(vals))
        ax.scatter(positions[i] + jitter, vals, color=color, alpha=0.7,
                   s=30, zorder=3, edgecolor='white', linewidths=0.5)

    # Cumulative max line
    if cum_max is not None:
        ax.plot(positions[:len(cum_max)], cum_max, 'k--o', linewidth=2,
                markersize=8, label='Cumulative Max', zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels(round_labels, fontsize=fontsize_tick)
    ax.set_xlabel(xlabel, fontsize=fontsize_label)
    ax.set_ylabel(ylabel, fontsize=fontsize_label)
    ax.tick_params(labelsize=fontsize_tick)

    if title:
        ax.set_title(title, fontsize=fontsize_title)

    if cum_max is not None or shade_regions:
        ax.legend(fontsize=fontsize_tick)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax
