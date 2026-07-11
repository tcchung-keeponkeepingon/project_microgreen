"""Batch selection with adaptive local penalization."""

import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist, pdist


def estimate_lipschitz(X_train, y_train, method='sample'):
    """Estimate the Lipschitz constant L of the objective function.

    Parameters
    ----------
    X_train : array-like, shape (n, d)
        Training input locations.
    y_train : array-like, shape (n,)
        Training target values.
    method : str
        'sample' — max |y_i - y_j| / ||x_i - x_j|| over all training pairs.

    Returns
    -------
    L : float
        Estimated Lipschitz constant (>= 1e-6 floor).
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).ravel()

    if method == 'sample':
        dists = pdist(X_train)
        y_diffs = pdist(y_train.reshape(-1, 1), metric='cityblock')
        nonzero = dists > 1e-12
        if nonzero.sum() == 0:
            return 1e-6
        ratios = y_diffs[nonzero] / dists[nonzero]
        return float(max(ratios.max(), 1e-6))
    else:
        raise ValueError(f"Unknown method: {method}")


def adaptive_local_penalization(acq_values, X_candidates, x_selected,
                                mu_selected, std_selected, f_min, L):
    """Adaptive local penalization (Gonzalez et al., 2016).

    For each already-selected point, computes a radius r and scale s
    from the GP predictions at that point:
        r = (mu_selected - f_min) / L
        s = std_selected / L
    Then the penalty is:
        penalty = Phi((distance - r) / s)
    which is ~0 near the selected point and ~1 far away.

    Parameters
    ----------
    acq_values : array, shape (n_candidates,)
        Current acquisition values.
    X_candidates : array, shape (n_candidates, d)
        Candidate points.
    x_selected : array, shape (d,)
        A single selected point.
    mu_selected : float
        GP mean at x_selected.
    std_selected : float
        GP std at x_selected.
    f_min : float
        Current best observed value.
    L : float
        Lipschitz constant estimate.

    Returns
    -------
    penalized : array, shape (n_candidates,)
        Acquisition values after multiplicative penalization.
    """
    acq_values = np.asarray(acq_values, dtype=np.float64).copy()
    X_candidates = np.asarray(X_candidates, dtype=np.float64)
    x_selected = np.asarray(x_selected, dtype=np.float64).reshape(1, -1)

    distances = cdist(X_candidates, x_selected).ravel()

    r = max((mu_selected - f_min) / L, 0.0)
    s = max(std_selected / L, 1e-12)

    penalty = norm.cdf((distances - r) / s)
    acq_values *= penalty

    return acq_values


def select_batch(X_candidates, acq_values, mu, std, y_best,
                 batch_size=10, L=None, X_train=None, y_train=None, seed=42):
    """Select a batch of points using adaptive local penalization.

    Parameters
    ----------
    X_candidates : array, shape (n, d)
        Candidate points.
    acq_values : array, shape (n,)
        Pre-computed acquisition values.
    mu : array, shape (n,)
        GP mean predictions.
    std : array, shape (n,)
        GP std predictions.
    y_best : float
        Current best observed value.
    batch_size : int
        Number of points to select.
    L : float or None
        Lipschitz constant. If None, estimated from X_train/y_train.
    X_train : array or None
        Training inputs (needed if L is None).
    y_train : array or None
        Training targets (needed if L is None).
    seed : int, default=42
        Random seed (reserved for future stochastic variants).

    Returns
    -------
    dict with keys:
        'points' : array, shape (batch_size, d)
        'indices' : list of int
        'acq_values' : list of float
        'diagnostics' : dict with per-step penalized landscapes
    """
    X_candidates = np.asarray(X_candidates, dtype=np.float64)
    acq_values = np.asarray(acq_values, dtype=np.float64).copy()
    mu = np.asarray(mu, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)

    if L is None:
        if X_train is None or y_train is None:
            raise ValueError("Must provide X_train and y_train if L is not given.")
        L = estimate_lipschitz(X_train, y_train)

    selected_indices = []
    selected_acq = []
    penalized = acq_values.copy()
    snapshots = []

    for i in range(batch_size):
        idx = int(np.argmax(penalized))
        selected_indices.append(idx)
        selected_acq.append(float(penalized[idx]))

        snapshots.append(penalized.copy())

        penalized = adaptive_local_penalization(
            penalized,
            X_candidates,
            X_candidates[idx],
            mu[idx],
            std[idx],
            y_best,
            L,
        )

    points = X_candidates[selected_indices]

    return {
        'points': points,
        'indices': selected_indices,
        'acq_values': selected_acq,
        'diagnostics': {
            'L': L,
            'snapshots': snapshots,
        },
    }


def calculate_batch_diversity(X_batch):
    """Diversity metrics for a batch of points.

    Returns dict with mean_distance, min_distance, max_distance, std_distance.
    """
    X_batch = np.asarray(X_batch, dtype=np.float64)
    if len(X_batch) < 2:
        return {'mean_distance': 0.0, 'min_distance': 0.0,
                'max_distance': 0.0, 'std_distance': 0.0}

    distances = pdist(X_batch)
    return {
        'mean_distance': float(np.mean(distances)),
        'min_distance': float(np.min(distances)),
        'max_distance': float(np.max(distances)),
        'std_distance': float(np.std(distances)),
    }


def acquisition_diagnostics(results_dict):
    """Structured summary of batch selection results.

    Parameters
    ----------
    results_dict : dict
        Output from select_batch().

    Returns
    -------
    dict with summary statistics.
    """
    points = results_dict['points']
    acq_vals = results_dict['acq_values']
    diag = results_dict['diagnostics']
    diversity = calculate_batch_diversity(points)

    return {
        'batch_size': len(points),
        'lipschitz_L': diag['L'],
        'acq_max': float(max(acq_vals)),
        'acq_min': float(min(acq_vals)),
        'acq_mean': float(np.mean(acq_vals)),
        **diversity,
    }
