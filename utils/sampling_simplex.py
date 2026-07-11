import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import qmc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import distance_matrix

def transform_to_simplex_exponential(uniform_points):
    """Transform uniform [0,1]^k points to simplex via exponential."""
    uniform_points = np.clip(uniform_points, 1e-10, 1 - 1e-10)
    exp_values = -np.log(uniform_points)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def transform_to_simplex_sort(uniform_points):
    """Transform uniform [0,1]^(k-1) points to k-simplex via sort transformation."""
    n_samples, k_minus_1 = uniform_points.shape
    
    with_boundaries = np.hstack([
        np.zeros((n_samples, 1)),
        uniform_points,
        np.ones((n_samples, 1))
    ])
    
    sorted_points = np.sort(with_boundaries, axis=1)
    return np.diff(sorted_points, axis=1)

def check_constraints_vectorized(simplex_samples, combo, material_constraints):
    """Check constraints for multiple samples at once."""
    n_samples = simplex_samples.shape[0]
    valid_mask = np.ones(n_samples, dtype=bool)
    
    for i, dim_idx in enumerate(combo):
        min_val, max_val = material_constraints[dim_idx]
        valid_mask &= (simplex_samples[:, i] >= min_val) & (simplex_samples[:, i] <= max_val)
    
    return valid_mask
    
def sample_sobol_simplex(n_dims, k_nonzero, n_samples, material_constraints=None, 
                         seed=None, oversample=False, oversample_factor=3.0, 
                         enforce_order_2=False, transformation='exponential'):
    """Systematically cover all combinations with Sobol sequence sampling."""
    
    if material_constraints is None:
        material_constraints = {i: (0.0, 1.0) for i in range(n_dims)}
    
    # Select transformation
    if transformation == 'exponential':
        sobol_dim = k_nonzero
        transform_func = transform_to_simplex_exponential
    elif transformation == 'sort':
        sobol_dim = k_nonzero - 1
        transform_func = transform_to_simplex_sort
    else:
        raise ValueError(f"transformation must be 'exponential' or 'sort', got '{transformation}'")
    
    all_combos = list(combinations(range(n_dims), k_nonzero))
    n_combos = len(all_combos)
    
    samples_per_combo = n_samples // n_combos
    extra_samples = n_samples % n_combos
    
    all_samples = []
    all_labels = []
    total_requested = 0
    total_generated = 0
    
    for combo_idx, combo in enumerate(all_combos):
        n_combo_samples = samples_per_combo + (1 if combo_idx < extra_samples else 0)
        
        if enforce_order_2 and n_combo_samples > 0:
            n_combo_samples = 2**int(np.round(np.log2(n_combo_samples)))
        
        total_requested += n_combo_samples
        
        combo_seed = (seed + combo_idx) if seed is not None else None
        sobol = qmc.Sobol(d=sobol_dim, scramble=True, seed=combo_seed)
        
        if oversample:
            n_generate = int(n_combo_samples * oversample_factor)
            if enforce_order_2:
                n_generate = 2**int(np.ceil(np.log2(n_generate)))
            
            sobol_points = sobol.random(n_generate)
            simplex_samples = transform_func(sobol_points)
            
            valid_mask = check_constraints_vectorized(simplex_samples, combo, material_constraints)
            valid_samples = simplex_samples[valid_mask][:n_combo_samples]
            
            for sample in valid_samples:
                sparse_sample = np.zeros(n_dims)
                sparse_sample[list(combo)] = sample
                all_samples.append(sparse_sample)
                all_labels.append(combo_idx)
            
            total_generated += len(valid_samples)
        else:
            batch_size = 100
            if enforce_order_2:
                batch_size = 128
            
            collected = 0
            while collected < n_combo_samples:
                sobol_points = sobol.random(batch_size)
                simplex_samples = transform_func(sobol_points)
                
                valid_mask = check_constraints_vectorized(simplex_samples, combo, material_constraints)
                valid_samples = simplex_samples[valid_mask]
                
                n_needed = n_combo_samples - collected
                take_samples = valid_samples[:n_needed]
                
                for sample in take_samples:
                    sparse_sample = np.zeros(n_dims)
                    sparse_sample[list(combo)] = sample
                    all_samples.append(sparse_sample)
                    all_labels.append(combo_idx)
                
                collected += len(take_samples)
            
            total_generated += collected
    
    if total_generated < 0.8 * total_requested:
        print(f"Warning: Only generated {total_generated}/{total_requested} samples "
              f"({100*total_generated/total_requested:.1f}%) due to tight constraints.")
    
    if enforce_order_2 and total_requested != n_samples:
        print(f"Info: Adjusted sample count from {n_samples} to {total_requested} "
              f"(power of 2 per combo for optimal Sobol properties)")
    
    return np.array(all_samples), np.array(all_labels)  

def sample_random_simplex(n_dims, k_nonzero, n_samples, material_constraints=None, seed=None, transformation='exponential'):
    """Systematically cover all combinations of non-zero dimensions with constraints."""
    
    # Create RNG once
    rng = np.random.default_rng(seed)
        
    # Select transformation
    if transformation == 'exponential':
        sobol_dim = k_nonzero
        transform_func = transform_to_simplex_exponential
    elif transformation == 'sort':
        sobol_dim = k_nonzero - 1
        transform_func = transform_to_simplex_sort
    else:
        raise ValueError(f"transformation must be 'exponential' or 'sort', got '{transformation}'")
        
    if material_constraints is None:
        material_constraints = {i: (0.0, 1.0) for i in range(n_dims)}
    
    all_combos = list(combinations(range(n_dims), k_nonzero))
    n_combos = len(all_combos)
    
    samples_per_combo = n_samples // n_combos
    extra_samples = n_samples % n_combos
    
    all_samples = []
    all_labels = []
    
    for combo_idx, combo in enumerate(all_combos):
        n_combo_samples = samples_per_combo + (1 if combo_idx < extra_samples else 0)
        
        for _ in range(n_combo_samples):
            valid_sample = False
            max_attempts = 1000
            attempts = 0
            
            while not valid_sample and attempts < max_attempts:
                attempts += 1
                
                uniform_points = rng.uniform(low=0.0, high=1.0, size=(1, sobol_dim))
                simplex_values = transform_func(uniform_points)[0]
                
                valid = True
                for i, dim_idx in enumerate(combo):
                    min_val, max_val = material_constraints[dim_idx]
                    if simplex_values[i] < min_val or simplex_values[i] > max_val:
                        valid = False
                        break
                
                if valid:
                    valid_sample = True
                    sparse_sample = np.zeros(n_dims)
                    sparse_sample[list(combo)] = simplex_values
                    all_samples.append(sparse_sample)
                    all_labels.append(combo_idx) 
            
            if not valid_sample:
                continue
    
    return np.array(all_samples), np.array(all_labels)

def evaluate_uniformity(samples, combo_labels):
    """Evaluate spacing quality per combo and overall."""
    from scipy.spatial import distance_matrix
    
    unique_combos = np.unique(combo_labels)
    combo_metrics = []
    
    for combo_id in unique_combos:
        combo_samples = samples[combo_labels == combo_id]
        
        if len(combo_samples) < 2:
            continue
        
        # Distance matrix for this combo only
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
        'mean_cv_across_combos': np.mean(all_cv),
        'worst_cv': np.max(all_cv),
        'mean_min_distance': np.mean(all_min),
        'worst_min_distance': np.min(all_min)
    }
    
    return summary

def create_2d_projections(samples, material_names, combo_labels=None):  # NEW parameter
    """Create interactive 2D projections using Plotly."""
    n_dims = samples.shape[1]
    all_pairs = list(combinations(range(n_dims), 2))
    total_pairs = len(all_pairs)
    
    max_plots = 20
    if total_pairs > max_plots:
        import random
        random.seed(42)
        dim_pairs = random.sample(all_pairs, max_plots)
        dim_pairs.sort()
        title_suffix = f" (showing {max_plots} of {total_pairs} projections)"
    else:
        dim_pairs = all_pairs
        title_suffix = ""
    
    n_pairs = len(dim_pairs)
    n_cols = min(4, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )
    
    for idx, (i, j) in enumerate(dim_pairs):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        # NEW: conditional marker coloring
        marker_dict = dict(size=6, opacity=0.6)
        if combo_labels is not None:
            marker_dict.update(dict(
                color=combo_labels,
                colorscale='Viridis',
                showscale=(idx == 0)
            ))
        else:
            marker_dict['color'] = '#42A0FF'
        
        fig.add_trace(
            go.Scatter(
                x=samples[:, i],
                y=samples[:, j],
                mode='markers',
                marker=marker_dict,
                showlegend=False,
                hovertemplate=f"{material_names[i]}: %{{x:.3f}}<br>{material_names[j]}: %{{y:.3f}}<extra></extra>"
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(
            title_text=material_names[i],
            range=[-0.05, 1.05],
            row=row, col=col,
            gridcolor='lightgray',
            showgrid=True
        )
        fig.update_yaxes(
            title_text=material_names[j],
            range=[-0.05, 1.05],
            row=row, col=col,
            gridcolor='lightgray',
            showgrid=True
        )
    
    fig.update_layout(
        title=f"{n_dims}D Sparse Simplex - 2D Projections{title_suffix}",
        height=250 * n_rows,
        showlegend=False,
        template="plotly_white"
    )
    
    return fig

def create_3d_plot(samples, material_names, combo_labels=None):
    """Create interactive 3D scatter plot using Plotly."""
    x = samples[:, 0]
    y = samples[:, 1]
    z = samples[:, 2]
    
    marker_dict = dict(size=4, opacity=0.7)
    if combo_labels is not None:
        marker_dict.update(dict(
            color=combo_labels,
            colorscale='Viridis',
            colorbar=dict(title="Combo")
        ))
    else:
        marker_dict['color'] = '#42A0FF'
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=marker_dict,
        hovertemplate=(
            f"{material_names[0]}: %{{x:.3f}}<br>"
            f"{material_names[1]}: %{{y:.3f}}<br>"
            f"{material_names[2]}: %{{z:.3f}}<extra></extra>"
        )
    )])
    
    fig.update_layout(
        title="3D Sparse Simplex Visualization",
        scene=dict(
            xaxis_title=material_names[0],
            yaxis_title=material_names[1],
            zaxis_title=material_names[2],
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            zaxis=dict(range=[0, 1])
        ),
        width=800,
        height=700
    )
    
    return fig