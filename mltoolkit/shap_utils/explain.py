"""SHAP explanation utilities."""

import numpy as np


def compute_shap(predict_fn, X, feature_names=None, method='kernel',
                 nsamples=500, seed=42):
    """
    Compute SHAP values for a prediction function.

    Parameters
    ----------
    predict_fn : callable
        predict_fn(X) -> predictions array.
    X : np.ndarray
        Feature matrix for explanation.
    feature_names : list of str or None
        Feature names.
    method : str, default='kernel'
        'kernel' for KernelExplainer, 'tree' for TreeExplainer.
    nsamples : int, default=500
        Number of samples for KernelExplainer.
    seed : int, default=42
        Random seed.

    Returns
    -------
    shap_values : np.ndarray
        SHAP values of shape (n_samples, n_features).
    explainer : shap.Explainer
        Fitted SHAP explainer.
    """
    import shap

    np.random.seed(seed)

    if method == 'kernel':
        explainer = shap.KernelExplainer(predict_fn, X)
        shap_values = explainer.shap_values(X, nsamples=nsamples)
    elif method == 'tree':
        # predict_fn should be a tree-based model for TreeExplainer
        explainer = shap.TreeExplainer(predict_fn)
        shap_values = explainer.shap_values(X)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'kernel' or 'tree'.")

    return shap_values, explainer


def plot_force_top_n(shap_values, explainer, X_display, display_names,
                     y_pred=None, n=5, indices=None, sample_labels=None,
                     save_path=None, figsize_per_plot=(14, 3), dpi=300):
    """
    Create stitched force plots for the top-n predictions.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values.
    explainer : shap.Explainer
        Explainer with .expected_value attribute.
    X_display : np.ndarray
        Display-friendly feature values (e.g., percentages).
    display_names : list of str
        Feature display names.
    y_pred : np.ndarray or None
        Predicted values (for title annotation).
    n : int, default=5
        Number of top predictions to plot.
    indices : list of int or None
        Specific indices to plot. If None, uses top-n by predicted value.
    sample_labels : list of str or None
        Labels for each sample (e.g., sample IDs).
    save_path : str or None
        Path to save combined image.
    figsize_per_plot : tuple
        Size per individual force plot.
    dpi : int
        DPI for rendering.

    Returns
    -------
    combined : PIL.Image or None
        Combined image if PIL is available.
    """
    import shap
    import matplotlib.pyplot as plt
    from io import BytesIO

    if indices is None:
        if y_pred is not None:
            indices = np.argsort(y_pred)[::-1][:n]
        else:
            indices = list(range(min(n, len(shap_values))))

    force_images = []

    try:
        from PIL import Image
        has_pil = True
    except ImportError:
        has_pil = False

    for idx in indices:
        shap.force_plot(
            explainer.expected_value,
            shap_values[idx],
            X_display[idx],
            feature_names=display_names,
            matplotlib=True,
            show=False,
        )
        fig_tmp = plt.gcf()
        fig_tmp.set_size_inches(*figsize_per_plot)

        # Add title
        title_parts = []
        if sample_labels is not None:
            title_parts.append(sample_labels[idx])
        if y_pred is not None:
            title_parts.append(f'pred={y_pred[idx]:.3f}')
        if title_parts:
            fig_tmp.suptitle('  '.join(title_parts), fontsize=16, y=1.08)

        if has_pil:
            buf = BytesIO()
            fig_tmp.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
            plt.close(fig_tmp)
            buf.seek(0)
            force_images.append(Image.open(buf))
        else:
            plt.close(fig_tmp)

    if not has_pil or not force_images:
        return None

    # Stitch images vertically
    widths = [img.width for img in force_images]
    heights = [img.height for img in force_images]
    max_w = max(widths)
    total_h = sum(heights)

    combined = Image.new('RGBA', (max_w, total_h), (255, 255, 255, 255))
    y_offset = 0
    for img in force_images:
        combined.paste(img, (0, y_offset))
        y_offset += img.height

    if save_path:
        combined.save(save_path, dpi=(dpi, dpi))

    return combined


def compare_shap(shap_results_dict, feature_names, save_path=None,
                 figsize=None, dpi=300, fontsize_title=16, fontsize_label=14):
    """
    Side-by-side mean |SHAP| bar plots across models.

    Parameters
    ----------
    shap_results_dict : dict
        Mapping of model_name -> shap_values (np.ndarray of shape (n_samples, n_features)).
    feature_names : list of str
        Feature names.
    save_path : str or None
        Path to save the figure.
    figsize : tuple or None
        Figure size. Auto-calculated if None.
    dpi : int
        DPI for saving.
    fontsize_title, fontsize_label : int
        Font sizes.

    Returns
    -------
    fig, axes
    """
    import matplotlib.pyplot as plt

    n_models = len(shap_results_dict)
    if figsize is None:
        figsize = (6 * n_models, 5)

    fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (name, shap_values) in zip(axes, shap_results_dict.items()):
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Sort by importance
        sorted_idx = np.argsort(mean_abs_shap)
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_values = mean_abs_shap[sorted_idx]

        ax.barh(sorted_names, sorted_values)
        ax.set_xlabel('Mean |SHAP|', fontsize=fontsize_label)
        ax.set_title(name, fontsize=fontsize_title)
        ax.tick_params(labelsize=fontsize_label - 2)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, axes
