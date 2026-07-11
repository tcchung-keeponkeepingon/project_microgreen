import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist

def expected_improvement(mu, std, y_best, xi=0.01):
    
    # Avoid division by zero
    std = np.maximum(std, 1e-9)
    
    # Calculate improvement
    improvement = mu - y_best - xi
    
    # Calculate Z score
    Z = improvement / std
    
    # Calculate EI
    ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
    
    # Set EI to 0 where std is 0
    ei[std <= 1e-9] = 0
    
    return ei


def upper_confidence_bound(mu, std, beta=2.0):

    return mu + beta * std


def probability_improvement(mu, std, y_best, xi=0.01):

    # Avoid division by zero
    std = np.maximum(std, 1e-9)
    
    # Calculate Z score
    Z = (mu - y_best - xi) / std
    
    return norm.cdf(Z)

def lower_confidence_bound(mu, std, beta=2.0):

    return mu - beta * std

# A-score
def acquisition_score(mu, std, w1, w2):

    # Normalize to [0, 1] range
    mu_norm = (mu - mu.min()) / (mu.max() - mu.min() + 1e-10)
    std_norm = (std - std.min()) / (std.max() - std.min() + 1e-10)
    
    return w1 * mu_norm + w2 * std_norm


def acquisition_score_dist(mu, dist, w1=0.5, w2=0.5):
    """
    Acquisition score combining model prediction (exploitation) with
    distance-based exploration.

    Args:
        mu: Model predictions for candidate points (exploitation signal)
        dist: Distance from candidates to observed data (exploration signal),
              e.g. min distance to nearest training point via cdist
        w1: Weight for exploitation (higher prediction = better)
        w2: Weight for exploration (farther from observed = better)

    Returns:
        Combined acquisition score, normalized to [0, 1]
    """
    mu = np.asarray(mu, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)

    # Normalize both to [0, 1]
    mu_norm = (mu - mu.min()) / (mu.max() - mu.min() + 1e-10)
    dist_norm = (dist - dist.min()) / (dist.max() - dist.min() + 1e-10)

    return w1 * mu_norm + w2 * dist_norm


# ============================================================================
# BATCH SELECTION STRATEGIES
# ============================================================================

def local_penalization(acquisition_values, X_candidates, X_selected, theta=0.3):
    """    
    Args:
        acquisition_values: Current acquisition function values
        X_candidates: Candidate points to select from
        X_selected: Already selected points in the batch
        theta: Penalization radius parameter
    
    Returns:
        Penalized acquisition values
    """
    if len(X_selected) == 0:
        return acquisition_values
    
    # Start with original values
    penalized_values = acquisition_values.copy()
    
    # Apply penalty for each selected point
    for x_selected in X_selected:
        # Calculate distances to selected point
        distances = cdist(X_candidates, [x_selected]).flatten()
        
        # Calculate Gaussian penalty
        penalty = np.exp(-distances**2 / (2 * theta**2))
        
        # Apply penalty (multiplicative)
        penalized_values = penalized_values * (1 - penalty)
    
    return penalized_values


def select_batch_with_penalization(X_candidates, gp, batch_size=8, 
                                   acquisition_func='ucb', beta=2.0, theta=0.3):
    """
    Select a batch of points using local penalization
    
    Args:
        X_candidates: Pool of candidate points
        gp: Fitted Gaussian Process model
        batch_size: Number of points to select
        acquisition_func: Which acquisition function to use ('ei', 'ucb', 'pi')
        beta: UCB parameter (if using UCB)
        theta: Penalization radius
    
    Returns:
        selected_points: Array of selected points
        selected_indices: Indices of selected points
    """
    selected_points = []
    selected_indices = []
    
    # Get current best from GP training data
    y_best = gp.y_train_.max()
    
    for i in range(batch_size):
        # Predict on candidates
        mu, std = gp.predict(X_candidates, return_std=True)
        
        # Calculate acquisition function
        if acquisition_func == 'ei':
            acq_values = expected_improvement(mu, std, y_best)
        elif acquisition_func == 'ucb':
            acq_values = upper_confidence_bound(mu, std, beta)
        elif acquisition_func == 'pi':
            acq_values = probability_improvement(mu, std, y_best)
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition_func}")
        
        # Apply penalization if we already selected points
        if i > 0:
            acq_values = local_penalization(
                acq_values, X_candidates, selected_points, theta
            )
        
        # Select next point
        next_idx = np.argmax(acq_values)
        selected_points.append(X_candidates[next_idx])
        selected_indices.append(next_idx)
        
        print(f"  Selected point {i+1}/{batch_size}: "
              f"acq_value={acq_values[next_idx]:.4f}, "
              f"μ={mu[next_idx]:.3f}, σ={std[next_idx]:.3f}")
    
    return np.array(selected_points), selected_indices


def calculate_batch_diversity(X_batch):
    """
    Calculate diversity metrics for a batch of points
    
    Args:
        X_batch: Array of selected points
    
    Returns:
        Dictionary with diversity metrics
    """
    from scipy.spatial.distance import pdist
    
    if len(X_batch) < 2:
        return {'mean_distance': 0, 'min_distance': 0, 'max_distance': 0}
    
    # Calculate pairwise distances
    distances = pdist(X_batch)
    
    return {
        'mean_distance': np.mean(distances),
        'min_distance': np.min(distances),
        'max_distance': np.max(distances),
        'std_distance': np.std(distances)
    }


# ============================================================================
# MULTI-OBJECTIVE ACQUISITION
# ============================================================================

def multi_objective_acquisition(mu, std, y_best, 
                               weights={'ei': 0.4, 'ucb': 0.3, 'pi': 0.3, 'ascore': 0.0},
                               xi=0.01, beta=2.0, w1=0.7, w2=0.3):
    """
    Combine EI, UCB, PI, and ascore acquisition functions
    
    Args:
        mu: GP mean predictions (array)
        std: GP standard deviation predictions (array)
        y_best: Current best observed value (scalar)
        weights: Dictionary of weights for each acquisition function
                 Keys: 'ei', 'ucb', 'pi', 'custom' (should sum to 1.0)
        xi: Exploration parameter for EI and PI (default 0.01)
        beta: Exploration parameter for UCB (default 2.0)
        w1: Exploitation weight for custom acquisition (default 0.7)
        w2: Exploration weight for custom acquisition (default 0.3)
    
    Returns:
        Combined acquisition values (array)
    """
    # Validate weights
    weight_sum = sum(weights.values())
    if not np.isclose(weight_sum, 1.0):
        print(f"Warning: Weights sum to {weight_sum:.3f}, not 1.0. Normalizing...")
        weights = {k: v/weight_sum for k, v in weights.items()}
    
    # Calculate each acquisition function
    ei = expected_improvement(mu, std, y_best, xi=xi)
    ucb = upper_confidence_bound(mu, std, beta=beta)
    pi = probability_improvement(mu, std, y_best, xi=xi)
    ascore = acquisition_score(mu, std, w1, w2)
    
    # Normalize each to [0, 1] range
    ei_normalized = (ei - ei.min()) / (ei.max() - ei.min() + 1e-9)
    ucb_normalized = (ucb - ucb.min()) / (ucb.max() - ucb.min() + 1e-9)
    pi_normalized = (pi - pi.min()) / (pi.max() - pi.min() + 1e-9)
    ascore_normalized = (ascore - ascore.min()) / (ascore.max() - ascore.min() + 1e-9)
    
    # Combine with weights
    combined = (weights['ei'] * ei_normalized + 
                weights['ucb'] * ucb_normalized + 
                weights['pi'] * pi_normalized +
                weights['ascore'] * ascore_normalized)
    
    return combined