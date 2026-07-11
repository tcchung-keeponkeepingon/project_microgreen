"""Uniformity evaluation for sampled design spaces."""

import numpy as np
from scipy.spatial import distance_matrix


def evaluate_uniformity(samples, combo_labels):
    """Evaluate spacing quality per combo and overall.

    Parameters
    ----------
    samples : np.ndarray of shape (n_samples, n_dims)
        Sample points.
    combo_labels : np.ndarray of shape (n_samples,)
        Combo index for each sample.

    Returns
    -------
    dict
        Summary with 'per_combo' list and overall metrics.
    """
    unique_combos = np.unique(combo_labels)
    combo_metrics = []

    for combo_id in unique_combos:
        combo_samples = samples[combo_labels == combo_id]

        if len(combo_samples) < 2:
            continue

        dist_matrix = distance_matrix(combo_samples, combo_samples)
        np.fill_diagonal(dist_matrix, np.inf)
        nn_distances = dist_matrix.min(axis=1)

        combo_metrics.append({
            'combo_id': combo_id,
            'n_samples': len(combo_samples),
            'mean_nn_dist': nn_distances.mean(),
            'min_nn_dist': nn_distances.min(),
            'cv_nn_dist': nn_distances.std() / nn_distances.mean()
        })

    # Overall metrics
    all_cv = [m['cv_nn_dist'] for m in combo_metrics]
    all_min = [m['min_nn_dist'] for m in combo_metrics]

    summary = {
        'per_combo': combo_metrics,
        'mean_cv_across_combos': np.mean(all_cv) if all_cv else 0.0,
        'worst_cv': np.max(all_cv) if all_cv else 0.0,
        'mean_min_distance': np.mean(all_min) if all_min else 0.0,
        'worst_min_distance': np.min(all_min) if all_min else 0.0,
    }

    return summary
