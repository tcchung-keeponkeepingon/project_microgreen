"""
Acquisition functions — numerically stable implementations.

Includes log-space EI, UCB, PI, Thompson sampling, and multi-objective combination.
"""

import numpy as np
from scipy.stats import norm, rankdata


# ============================================================================
# STABLE PRIMITIVES
# ============================================================================

def _log_normal_cdf(z):
    """Log of the standard normal CDF, stable for large negative z.

    For z < -30, uses the asymptotic expansion:
        log Phi(z) ~ -z^2/2 - log(-z) - log(2*pi)/2
    """
    z = np.asarray(z, dtype=np.float64)
    result = np.empty_like(z)

    safe = z >= -30
    result[safe] = np.log(np.maximum(norm.cdf(z[safe]), 1e-300))

    z_tail = z[~safe]
    result[~safe] = -0.5 * z_tail**2 - np.log(-z_tail) - 0.5 * np.log(2 * np.pi)

    return result


def _compute_z(mu, std, y_best, xi=0.01):
    """Standardized improvement Z = (mu - y_best - xi) / std.

    Clamps variance (std^2) at 1e-12 before taking the square root.
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
    """
    Z, improvement, std = _compute_z(mu, std, y_best, xi)

    log_phi = norm.logpdf(Z)
    log_Phi = _log_normal_cdf(Z)

    ratio = np.exp(log_phi - log_Phi)
    inner = improvement + std * ratio

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


def thompson_sampling(mu, std, seed=42):
    """Draw from GP posterior for Thompson sampling.

    Parameters
    ----------
    mu : array-like
        GP mean predictions.
    std : array-like
        GP std predictions.
    seed : int, default=42
        Random seed.

    Returns
    -------
    samples : np.ndarray
        Random samples from N(mu, std).
    """
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    return rng.normal(mu, std)


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
    mu, std : arrays
        GP predictions.
    y_best : float
        Current best observed value.
    weights : dict or None
        Keys from {'ei', 'ucb', 'pi'}, values are non-negative weights.
        Defaults to {'ei': 0.4, 'ucb': 0.3, 'pi': 0.3}.
    xi : float
        Exploration parameter for EI/PI.
    beta : float
        Exploration parameter for UCB.
    normalization : str
        'rank', 'minmax', or 'none'.

    Returns
    -------
    combined : array
        Combined acquisition values.
    """
    if weights is None:
        weights = {'ei': 0.4, 'ucb': 0.3, 'pi': 0.3}

    weight_sum = sum(weights.values())
    if not np.isclose(weight_sum, 1.0):
        weights = {k: v / weight_sum for k, v in weights.items()}

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

    normalize_fn = {
        'rank': rank_normalize,
        'minmax': _minmax_normalize,
        'none': lambda x: np.asarray(x, dtype=np.float64),
    }[normalization]

    combined = np.zeros(len(mu), dtype=np.float64)
    for key, vals in components.items():
        combined += weights[key] * normalize_fn(vals)

    return combined
