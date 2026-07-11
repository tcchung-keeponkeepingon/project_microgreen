"""Heatmap plotting utilities for prediction and acquisition surfaces."""

import numpy as np
import matplotlib.pyplot as plt
from .defaults import PLOT_DEFAULTS


def plot_prediction_heatmap(predict_fn, grid_fn, panel_values, x_col, y_col,
                            cbar_ticks=None, title=None, target_point=None,
                            round_data=None, save_path=None,
                            cmap='plasma', cbar_label=None,
                            figsize=None, dpi=300,
                            fontsize_title=16, fontsize_label=14,
                            fontsize_tick=12, feasibility_boundary=True):
    """
    Plot a prediction heatmap across panel slices.

    Parameters
    ----------
    predict_fn : callable
        predict_fn(X_features) -> predictions array.
    grid_fn : callable
        grid_fn(panel_value) -> (x_1d, y_1d, X_features, feasibility_mask).
        x_1d, y_1d are 1D arrays for the meshgrid axes.
        X_features is the full feature array for prediction.
        feasibility_mask is a boolean array (True = feasible).
    panel_values : list
        Values to create separate panels for (e.g., mass loadings).
    x_col : str
        Label for x-axis.
    y_col : str
        Label for y-axis.
    cbar_ticks : list or None
        Custom colorbar tick values.
    title : str or None
        Overall figure title.
    target_point : tuple or None
        (x, y) coordinates to mark with a star.
    round_data : list of dict or None
        Each dict has 'x', 'y', 'label', 'color', 'marker' for overlay points.
    save_path : str or None
        Path to save the figure.
    cmap : str, default='plasma'
        Matplotlib colormap.
    cbar_label : str or None
        Colorbar label.
    figsize : tuple or None
        Figure size. Auto-calculated if None.
    dpi : int, default=300
        DPI for saving.
    fontsize_title : int, default=16
        Title font size.
    fontsize_label : int, default=14
        Axis label font size.
    fontsize_tick : int, default=12
        Tick label font size.
    feasibility_boundary : bool, default=True
        Whether to show feasibility boundary as gray dashed line.

    Returns
    -------
    fig, axes
    """
    n_panels = len(panel_values)
    if figsize is None:
        figsize = (7 * n_panels, 5)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    for ax, pval in zip(axes, panel_values):
        x_1d, y_1d, X_features, feas_mask = grid_fn(pval)
        preds = predict_fn(X_features)

        # Reshape to grid
        nx, ny = len(x_1d), len(y_1d)
        pred_grid = np.full((ny, nx), np.nan)
        feas_grid = feas_mask.reshape(ny, nx) if feas_mask is not None else np.ones((ny, nx), dtype=bool)
        pred_grid[feas_grid] = preds[feas_mask] if feas_mask is not None else preds

        im = ax.pcolormesh(x_1d, y_1d, pred_grid, cmap=cmap, shading='auto')

        if feasibility_boundary and feas_mask is not None:
            ax.contour(x_1d, y_1d, feas_grid.astype(float),
                       levels=[0.5], colors='gray', linestyles='--', linewidths=1.5)

        if target_point is not None:
            defaults = PLOT_DEFAULTS['target_marker']
            ax.scatter(*target_point, **defaults)

        if round_data is not None:
            for rd in round_data:
                ax.scatter(rd['x'], rd['y'], label=rd.get('label', ''),
                           color=rd.get('color', 'white'),
                           marker=rd.get('marker', 'o'),
                           edgecolor=rd.get('edgecolor', 'black'),
                           s=rd.get('s', 80), zorder=5)

        cbar = fig.colorbar(im, ax=ax)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=fontsize_label)
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)

        ax.set_xlabel(x_col, fontsize=fontsize_label)
        ax.set_ylabel(y_col, fontsize=fontsize_label)
        ax.tick_params(labelsize=fontsize_tick)

    if title:
        fig.suptitle(title, fontsize=fontsize_title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, axes


def plot_acquisition_heatmap(predict_fn, grid_fn, panel_values, x_col, y_col,
                             cbar_ticks=None, title=None, target_point=None,
                             round_data=None, save_path=None,
                             cmap='inferno', cbar_label='Acquisition Value',
                             **kwargs):
    """Plot an acquisition function heatmap. Same API as plot_prediction_heatmap
    but with inferno colormap default."""
    return plot_prediction_heatmap(
        predict_fn, grid_fn, panel_values, x_col, y_col,
        cbar_ticks=cbar_ticks, title=title, target_point=target_point,
        round_data=round_data, save_path=save_path,
        cmap=cmap, cbar_label=cbar_label, **kwargs
    )
