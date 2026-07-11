"""Default plot styling constants."""

PLOT_DEFAULTS = {
    # Font sizes
    'title_fontsize': 16,
    'label_fontsize': 14,
    'tick_fontsize': 12,

    # Figure
    'dpi': 300,

    # Colormaps
    'prediction_cmap': 'plasma',
    'acquisition_cmap': 'inferno',
    'categorical_cmap': 'viridis',

    # Round colors for BO campaigns
    'round_colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],

    # Target marker style (e.g., for optimal point)
    'target_marker': {
        'marker': '*',
        'color': 'lime',
        'edgecolor': 'black',
        's': 300,
        'zorder': 10,
        'linewidths': 1.0,
    },

    # Colorbar
    'cbar_ticks': [0.75, 1.25, 1.75, 2.25],
    'cbar_label_fontsize': 14,
}
