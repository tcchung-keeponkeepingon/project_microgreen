import numpy as np
import pandas as pd
import warnings

def oversample_by_hist(X, y, n_bins=100, random_state=None, expand_factor=10.0, 
                       shuffle=True, smoothing_coeff=1.0, return_df=False):  # Fixed typo
    """
    Oversamples a dataset based on the histogram of the target variable y,
    driving towards a uniform distribution as expand_factor increases.

    Uses simplified inverse proportionality with smoothing for sampling weights.
    """
    
    # Store DataFrame info before conversion
    X_is_df = isinstance(X, pd.DataFrame)
    y_is_df = isinstance(y, pd.DataFrame)
    
    if X_is_df:
        X_cols = X.columns.tolist()
    
    if y_is_df:
        y_cols = y.columns.tolist()
        y_shape = y.shape  # Store original shape
    
    # Convert to numpy arrays
    X = np.asarray(X)
    y_array = np.asarray(y)
    
    # Use flattened y for histogram calculation
    y_flat = y_array.flatten()
    
    if expand_factor < 1.0:
        warnings.warn(f"expand_factor={expand_factor} < 1.0. No oversampling will occur.", UserWarning)
        n_resampled = 0
    else:
        # Calculate number of samples to ADD
        n_resampled = int(round(len(y_flat) * (expand_factor - 1.0)))
    
    n_original = len(y_flat)
    
    if n_original == 0:
        warnings.warn("Input data is empty. Returning empty arrays.", UserWarning)
        if return_df and X_is_df:
            return pd.DataFrame(X.copy(), columns=X_cols), pd.DataFrame(y_array.copy(), columns=y_cols if y_is_df else None)
        return X.copy(), y_array.copy()
    
    if n_resampled <= 0:
        if return_df and X_is_df:
            return pd.DataFrame(X.copy(), columns=X_cols), pd.DataFrame(y_array.copy(), columns=y_cols if y_is_df else None)
        return X.copy(), y_array.copy()
    
    # --- Core Oversampling Logic ---
    # 1. Calculate histogram using flattened y
    try:
        hist, bin_edges = np.histogram(y_flat, bins=n_bins)
    except MemoryError:
        hist, bin_edges = np.histogram(y_flat, bins=100)
    
    # Ensure there's at least one bin
    if len(hist) == 0:
        warnings.warn("Histogram resulted in zero bins. Skipping oversampling.", UserWarning)
        if return_df and X_is_df:
            return pd.DataFrame(X.copy(), columns=X_cols), pd.DataFrame(y_array.copy(), columns=y_cols if y_is_df else None)
        return X.copy(), y_array.copy()
    
    # 2. Assign original data points to bins
    y_bin_indices = np.digitize(y_flat, bin_edges[:-1], right=False) - 1
    y_bin_indices = np.clip(y_bin_indices, 0, len(hist) - 1)
    
    # 3. Calculate sampling weights
    smoothed_hist = hist.astype(float) + float(smoothing_coeff)
    sampling_weights_per_bin = 1.0 / smoothed_hist
    
    # 4. Map sampling weights to each data point
    y_sampling_weights = sampling_weights_per_bin[y_bin_indices]
    
    # 5. Normalize weights
    sum_weights = np.sum(y_sampling_weights)
    if sum_weights <= 1e-9:
        warnings.warn("Sum of sampling weights is close to zero. Defaulting to uniform sampling.", UserWarning)
        y_sampling_probas = np.ones(n_original) / n_original
    else:
        y_sampling_probas = y_sampling_weights / sum_weights
    
    # 6. Perform resampling (using row indices)
    rng = np.random.default_rng(random_state)
    original_indices = np.arange(X.shape[0])  # Use X's first dimension
    resampled_indices = rng.choice(original_indices, size=n_resampled, replace=True, p=y_sampling_probas)
    
    # 7. Combine original and resampled data
    X_resampled = X[resampled_indices]
    y_resampled = y_array[resampled_indices]  # Use original shape
    
    X_combined = np.vstack([X, X_resampled])
    y_combined = np.concatenate([y_array, y_resampled])  # Preserve shape
    
    # 8. Shuffle if requested
    if shuffle:
        combined_indices = np.arange(len(y_combined))
        rng.shuffle(combined_indices)
        X_combined = X_combined[combined_indices]
        y_combined = y_combined[combined_indices]
    
    # 9. Return as DataFrame if requested
    if return_df:
        if X_is_df:
            X_combined = pd.DataFrame(X_combined, columns=X_cols)
        if y_is_df:
            y_combined = pd.DataFrame(y_combined, columns=y_cols)
    
    return X_combined, y_combined

# Oversampling V2
def oversample_by_hist_v2(X, y, n_bins=100, random_state=None, expand_factor=10.0, 
                       shuffle=True, smoothing_coeff=1.0, return_df=False,
                       y_col=None, oversample_column=None):  # NEW PARAMETERS
    """
    Oversamples a dataset based on the histogram of the target variable y,
    driving towards a uniform distribution as expand_factor increases.

    Uses simplified inverse proportionality with smoothing for sampling weights.
    """
    
    # Store DataFrame info before conversion
    X_is_df = isinstance(X, pd.DataFrame)
    y_is_df = isinstance(y, pd.DataFrame)
    
    if X_is_df:
        X_cols = X.columns.tolist()
    
    if y_is_df:
        y_cols = y.columns.tolist()
        y_shape = y.shape
    
    # ===== NEW: DataFrame handling branch =====
    if X_is_df:
        # Convert y to DataFrame if needed
        if not y_is_df:
            if y_col is None:
                warnings.warn("X is DataFrame but y is array. Please provide y_col parameter.", UserWarning)
                y_col = ['y'] if len(y.shape) == 1 else [f'y_{i}' for i in range(y.shape[1])]
            y_df = pd.DataFrame(y, columns=y_col if isinstance(y_col, list) else [y_col])
        else:
            y_df = y
            y_cols = y_df.columns.tolist()
        
        # Concatenate X and y
        combined_df = pd.concat([X, y_df], axis=1)
        
        # Determine oversample column
        if oversample_column is None or oversample_column not in combined_df.columns:
            if oversample_column is not None:
                warnings.warn(f"Column '{oversample_column}' not found. Using y column instead.", UserWarning)
            oversample_column = y_cols[0] if y_is_df else (y_col[0] if isinstance(y_col, list) else y_col)
        
        # Calculate distribution
        oversample_data = combined_df[oversample_column]
        
        # Handle categorical vs numerical
        if pd.api.types.is_numeric_dtype(oversample_data):
            # Numerical: use histogram
            try:
                hist, bin_edges = np.histogram(oversample_data, bins=n_bins)
            except MemoryError:
                hist, bin_edges = np.histogram(oversample_data, bins=100)
            
            bin_indices = np.digitize(oversample_data, bin_edges[:-1], right=False) - 1
            bin_indices = np.clip(bin_indices, 0, len(hist) - 1)
        else:
            # Categorical: use value counts
            value_counts = oversample_data.value_counts()
            hist = value_counts.values
            category_map = {cat: idx for idx, cat in enumerate(value_counts.index)}
            bin_indices = oversample_data.map(category_map).values
        
        # Calculate sampling weights
        smoothed_hist = hist.astype(float) + float(smoothing_coeff)
        sampling_weights_per_bin = 1.0 / smoothed_hist
        sampling_weights = sampling_weights_per_bin[bin_indices]
        
        # Normalize weights
        sum_weights = np.sum(sampling_weights)
        if sum_weights <= 1e-9:
            warnings.warn("Sum of sampling weights is close to zero. Defaulting to uniform sampling.", UserWarning)
            sampling_probas = np.ones(len(combined_df)) / len(combined_df)
        else:
            sampling_probas = sampling_weights / sum_weights
        
        # Calculate number of samples to add
        if expand_factor < 1.0:
            warnings.warn(f"expand_factor={expand_factor} < 1.0. No oversampling will occur.", UserWarning)
            n_resampled = 0
        else:
            n_resampled = int(round(len(combined_df) * (expand_factor - 1.0)))
        
        if n_resampled <= 0:
            if return_df:
                return X.copy(), y_df.copy()
            return X.values.copy(), y_df.values.copy()
        
        # Perform resampling
        rng = np.random.default_rng(random_state)
        original_indices = np.arange(len(combined_df))
        resampled_indices = rng.choice(original_indices, size=n_resampled, replace=True, p=sampling_probas)
        
        # Combine original and resampled
        resampled_df = combined_df.iloc[resampled_indices]
        combined_result = pd.concat([combined_df, resampled_df], ignore_index=True)
        
        # Shuffle if requested
        if shuffle:
            combined_result = combined_result.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Split back to X and y
        X_result = combined_result[X_cols]
        y_result = combined_result[y_cols if y_is_df else (y_col if isinstance(y_col, list) else [y_col])]
        
        if return_df:
            return X_result, y_result
        else:
            return X_result.values, y_result.values
    
    # ===== ORIGINAL CODE: numpy array handling =====
    # Convert to numpy arrays
    X = np.asarray(X)
    y_array = np.asarray(y)
    
    # Use flattened y for histogram calculation
    y_flat = y_array.flatten()
    
    if expand_factor < 1.0:
        warnings.warn(f"expand_factor={expand_factor} < 1.0. No oversampling will occur.", UserWarning)
        n_resampled = 0
    else:
        # Calculate number of samples to ADD
        n_resampled = int(round(len(y_flat) * (expand_factor - 1.0)))
    
    n_original = len(y_flat)
    
    if n_original == 0:
        warnings.warn("Input data is empty. Returning empty arrays.", UserWarning)
        if return_df and X_is_df:
            return pd.DataFrame(X.copy(), columns=X_cols), pd.DataFrame(y_array.copy(), columns=y_cols if y_is_df else None)
        return X.copy(), y_array.copy()
    
    if n_resampled <= 0:
        if return_df and X_is_df:
            return pd.DataFrame(X.copy(), columns=X_cols), pd.DataFrame(y_array.copy(), columns=y_cols if y_is_df else None)
        return X.copy(), y_array.copy()
    
    # --- Core Oversampling Logic ---
    # 1. Calculate histogram using flattened y
    try:
        hist, bin_edges = np.histogram(y_flat, bins=n_bins)
    except MemoryError:
        hist, bin_edges = np.histogram(y_flat, bins=100)
    
    # Ensure there's at least one bin
    if len(hist) == 0:
        warnings.warn("Histogram resulted in zero bins. Skipping oversampling.", UserWarning)
        if return_df and X_is_df:
            return pd.DataFrame(X.copy(), columns=X_cols), pd.DataFrame(y_array.copy(), columns=y_cols if y_is_df else None)
        return X.copy(), y_array.copy()
    
    # 2. Assign original data points to bins
    y_bin_indices = np.digitize(y_flat, bin_edges[:-1], right=False) - 1
    y_bin_indices = np.clip(y_bin_indices, 0, len(hist) - 1)
    
    # 3. Calculate sampling weights
    smoothed_hist = hist.astype(float) + float(smoothing_coeff)
    sampling_weights_per_bin = 1.0 / smoothed_hist
    
    # 4. Map sampling weights to each data point
    y_sampling_weights = sampling_weights_per_bin[y_bin_indices]
    
    # 5. Normalize weights
    sum_weights = np.sum(y_sampling_weights)
    if sum_weights <= 1e-9:
        warnings.warn("Sum of sampling weights is close to zero. Defaulting to uniform sampling.", UserWarning)
        y_sampling_probas = np.ones(n_original) / n_original
    else:
        y_sampling_probas = y_sampling_weights / sum_weights
    
    # 6. Perform resampling (using row indices)
    rng = np.random.default_rng(random_state)
    original_indices = np.arange(X.shape[0])
    resampled_indices = rng.choice(original_indices, size=n_resampled, replace=True, p=y_sampling_probas)
    
    # 7. Combine original and resampled data
    X_resampled = X[resampled_indices]
    y_resampled = y_array[resampled_indices]
    
    X_combined = np.vstack([X, X_resampled])
    y_combined = np.concatenate([y_array, y_resampled])
    
    # 8. Shuffle if requested
    if shuffle:
        combined_indices = np.arange(len(y_combined))
        rng.shuffle(combined_indices)
        X_combined = X_combined[combined_indices]
        y_combined = y_combined[combined_indices]
    
    # 9. Return as DataFrame if requested
    if return_df:
        if X_is_df:
            X_combined = pd.DataFrame(X_combined, columns=X_cols)
        if y_is_df:
            y_combined = pd.DataFrame(y_combined, columns=y_cols)
    
    return X_combined, y_combined