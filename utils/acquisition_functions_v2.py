"""
Acquisition functions v2 — numerically stable implementations.

Improvements over v1:
- log-space EI computation avoids underflow in high-confidence regions
- Adaptive local penalization (González et al., 2016) replaces fixed-theta Gaussian
- Rank normalization for robust multi-objective combination
"""

import numpy as np
from scipy.stats import norm, rankdata
from scipy.spatial.distance import cdist, pdist


# ============================================================================
# STABLE PRIMITIVES
# ============================================================================

def _log_normal_cdf(z):
    """Log of the standard normal CDF, stable for large negative z.

    For z < -30, uses the asymptotic expansion:
        log Phi(z) ~ -z^2/2 - log(-z) - log(2*pi)/2
    which avoids log(0) from norm.cdf underflowing to 0.
    """
    z = np.asarray(z, dtype=np.float64)
    result = np.empty_like(z)

    safe = z >= -30
    result[safe] = np.log(np.maximum(norm.cdf(z[safe]), 1e-300))

    # Asymptotic expansion for the deep tail
    z_tail = z[~safe]
    result[~safe] = -0.5 * z_tail**2 - np.log(-z_tail) - 0.5 * np.log(2 * np.pi)

    return result


def _compute_z(mu, std, y_best, xi=0.01):
    """Standardized improvement Z = (mu - y_best - xi) / std.

    Clamps variance (std^2) at 1e-12 before taking the square root,
    so the returned std is always >= ~1e-6.
    """
    mu = np.asarray(mu, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    std = np.sqrt(np.maximum(std**2, 1e-12))
    improvement = mu - y_best - xi
    Z = improvement / std
    return Z, improvement, std


# ============================================================================
# CORE ACQUISITION FUNCTIONS
# ============================================================================

def log_expected_improvement(mu, std, y_best, xi=0.01):
    """Expected improvement in log-space — never underflows to -inf.

    Returns log(EI) where EI = (mu - y_best - xi)*Phi(Z) + std*phi(Z).

    Uses the log-sum-exp identity:
        log(a*Phi + b*phi) = log(Phi) + log(a + b * phi/Phi)
    where phi/Phi = exp(log(phi) - log(Phi)) is stable.
    """
    Z, improvement, std = _compute_z(mu, std, y_best, xi)

    log_phi = norm.logpdf(Z)          # always finite
    log_Phi = _log_normal_cdf(Z)

    # log(EI) = log(Phi(Z)) + log(improvement + std * exp(log_phi - log_Phi))
    # The inner term: improvement + std * (phi(Z) / Phi(Z))
    ratio = np.exp(log_phi - log_Phi)  # phi/Phi, Mills ratio inverse
    inner = improvement + std * ratio

    # Where inner <= 0, EI is effectively 0 → log(EI) = -inf
    log_ei = np.full_like(Z, -np.inf)
    pos = inner > 0
    log_ei[pos] = log_Phi[pos] + np.log(inner[pos])

    return log_ei


def expected_improvement(mu, std, y_best, xi=0.01):
    """Expected improvement via exp(log_EI). Drop-in replacement for v1."""
    log_ei = log_expected_improvement(mu, std, y_best, xi)
    return np.exp(log_ei)


def probability_improvement(mu, std, y_best, xi=0.01):
    """Probability of improvement using stable log-CDF internally."""
    Z, _, _ = _compute_z(mu, std, y_best, xi)
    return norm.cdf(Z)


def upper_confidence_bound(mu, std, beta=2.0):
    """Upper confidence bound: mu + beta * std."""
    mu = np.asarray(mu, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    return mu + beta * std


def lower_confidence_bound(mu, std, beta=2.0):
    """Lower confidence bound: mu - beta * std."""
    mu = np.asarray(mu, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    return mu - beta * std


# ============================================================================
# BATCH SELECTION — ADAPTIVE LOCAL PENALIZATION
# ============================================================================

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
        # Avoid division by zero for duplicate points
        nonzero = dists > 1e-12
        if nonzero.sum() == 0:
            return 1e-6
        ratios = y_diffs[nonzero] / dists[nonzero]
        return float(max(ratios.max(), 1e-6))
    else:
        raise ValueError(f"Unknown method: {method}")


def adaptive_local_penalization(acq_values, X_candidates, x_selected,
                                mu_selected, std_selected, f_min, L):
    """Adaptive local penalization (González et al., 2016).

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
        Current best observed value (for maximization, use negative).
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
                 batch_size=10, L=None, X_train=None, y_train=None):
    """Select a batch of points using adaptive local penalization.

    Computes acquisition values ONCE (passed in), then iteratively
    applies adaptive_local_penalization to discourage clustering.

    Parameters
    ----------
    X_candidates : array, shape (n, d)
        Candidate points.
    acq_values : array, shape (n,)
        Pre-computed acquisition values for all candidates.
    mu : array, shape (n,)
        GP mean predictions for all candidates.
    std : array, shape (n,)
        GP std predictions for all candidates.
    y_best : float
        Current best observed value.
    batch_size : int
        Number of points to select.
    L : float, optional
        Lipschitz constant. If None, estimated from X_train/y_train.
    X_train : array, optional
        Training inputs (needed if L is None).
    y_train : array, optional
        Training targets (needed if L is None).

    Returns
    -------
    dict with keys:
        'points' : array, shape (batch_size, d)
        'indices' : list of int
        'acq_values' : list of float (acquisition value at selection time)
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

        # Store snapshot before penalizing (for visualization)
        snapshots.append(penalized.copy())

        # Penalize around the selected point
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


# ============================================================================
# MULTI-OBJECTIVE COMBINATION
# ============================================================================

def rank_normalize(values):
    """Rank-normalize values to (0, 1) via rank / (n + 1).

    Robust to outliers — the result depends only on ordering.
    """
    values = np.asarray(values, dtype=np.float64)
    return rankdata(values) / (len(values) + 1)


def _minmax_normalize(values):
    """Min-max normalize to [0, 1]."""
    values = np.asarray(values, dtype=np.float64)
    vmin, vmax = values.min(), values.max()
    denom = vmax - vmin
    if denom < 1e-12:
        return np.zeros_like(values)
    return (values - vmin) / denom


def multi_objective_acquisition(mu, std, y_best,
                                weights=None,
                                xi=0.01, beta=2.0,
                                normalization='rank'):
    """Combine EI, UCB, PI acquisition functions with robust normalization.

    Parameters
    ----------
    mu, std : arrays of GP predictions.
    y_best : float, current best observed value.
    weights : dict, optional
        Keys from {'ei', 'ucb', 'pi'}, values are non-negative weights.
        Defaults to {'ei': 0.4, 'ucb': 0.3, 'pi': 0.3}.
    xi : float, exploration parameter for EI/PI.
    beta : float, exploration parameter for UCB.
    normalization : str
        'rank' — rank normalization (default, robust to outliers).
        'minmax' — standard min-max normalization.
        'none' — no normalization (raw values).

    Returns
    -------
    combined : array, combined acquisition values.
    """
    if weights is None:
        weights = {'ei': 0.4, 'ucb': 0.3, 'pi': 0.3}

    weight_sum = sum(weights.values())
    if not np.isclose(weight_sum, 1.0):
        weights = {k: v / weight_sum for k, v in weights.items()}

    # Compute shared Z once
    Z, improvement, std_safe = _compute_z(mu, std, y_best, xi)
    Phi_Z = norm.cdf(Z)
    phi_Z = norm.pdf(Z)

    components = {}
    if weights.get('ei', 0) > 0:
        components['ei'] = improvement * Phi_Z + std_safe * phi_Z
    if weights.get('ucb', 0) > 0:
        components['ucb'] = np.asarray(mu, dtype=np.float64) + beta * std_safe
    if weights.get('pi', 0) > 0:
        components['pi'] = Phi_Z

    # Normalize
    normalize_fn = {
        'rank': rank_normalize,
        'minmax': _minmax_normalize,
        'none': lambda x: np.asarray(x, dtype=np.float64),
    }[normalization]

    combined = np.zeros(len(mu), dtype=np.float64)
    for key, vals in components.items():
        combined += weights[key] * normalize_fn(vals)

    return combined


# ============================================================================
# UTILITIES
# ============================================================================

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
