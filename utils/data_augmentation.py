import numpy as np
import pandas as pd


def data_aug_gaussian(X, y, feature_scale=0.05, label_scale=0.05, 
                      n_augment=5, seed=42, return_df=False):
    """
    Augment data by adding Gaussian noise.
    
    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Feature matrix
    y : np.ndarray or pd.Series/DataFrame
        Target values
    feature_scale : float
        Standard deviation of Gaussian noise for features
    label_scale : float
        Standard deviation of Gaussian noise for labels
    n_augment : int
        Number of augmented copies to generate
    seed : int
        Random seed for reproducibility
    return_df : bool
        If True and inputs are DataFrames, return DataFrames with original column names
    
    Returns
    -------
    X_aug : np.ndarray or pd.DataFrame
        Original data + n_augment augmented copies
    y_aug : np.ndarray or pd.Series/DataFrame
        Original labels + n_augment augmented copies
    """
    rng = np.random.default_rng(seed)
    
    # Check if inputs are DataFrames/Series and store metadata
    is_X_df = isinstance(X, pd.DataFrame)
    is_y_df = isinstance(y, (pd.Series, pd.DataFrame))
    
    # Store original column names and convert to numpy for processing
    if is_X_df:
        X_columns = X.columns
        X_np = X.values
    else:
        X_np = X
        X_columns = None
    
    if is_y_df:
        if isinstance(y, pd.Series):
            y_name = y.name
            y_columns = None
        else:
            y_columns = y.columns
            y_name = None
        y_np = y.values
    else:
        y_np = y
        y_columns = None
        y_name = None
    
    # Generate augmented data
    X_augmented = [X_np]
    y_augmented = [y_np]
    
    for _ in range(n_augment):
        X_noisy = rng.normal(loc=X_np, scale=feature_scale)
        y_noisy = rng.normal(loc=y_np, scale=label_scale)
        X_augmented.append(X_noisy)
        y_augmented.append(y_noisy)
    
    # Combine all augmented data
    X_aug = np.vstack(X_augmented)
    y_aug = np.concatenate(y_augmented)
    
    # Convert back to DataFrame if requested and inputs were DataFrames
    if return_df and is_X_df:
        X_aug = pd.DataFrame(X_aug, columns=X_columns)
    
    if return_df and is_y_df:
        if y_columns is not None:
            y_aug = pd.DataFrame(y_aug, columns=y_columns)
        else:
            y_aug = pd.Series(y_aug, name=y_name)
    
    return X_aug, y_aug