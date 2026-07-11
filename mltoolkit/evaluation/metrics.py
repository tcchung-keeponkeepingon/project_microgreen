"""Evaluation metrics for regression models."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
import warnings


def evaluation_metrics(y_true, y_pred, zero_handling='mask', epsilon=1e-3, clip_threshold=None, verbose=True):
    """
    Calculate comprehensive evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    zero_handling : str, default='mask'
        Strategy for handling zeros: 'epsilon' or 'mask'
    epsilon : float, default=1e-3
        Small value to avoid division by zero when zero_handling='epsilon'
    clip_threshold : float or None, default=None
        Clip relative errors above this threshold (e.g., 1.0 for 100%). None means no clipping.
    verbose : bool, default=True
        Whether to print results and warnings
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    def calculate_mre(y_true, y_pred, strategy, eps):
        """Calculate Mean Relative Error with configurable zero handling."""
        if strategy == 'epsilon':
            relative_errors = np.abs(y_true - y_pred) / (y_true + eps)
            if clip_threshold is not None:
                relative_errors = np.clip(relative_errors, None, clip_threshold)
            return np.mean(relative_errors) * 100

        elif strategy == 'mask':
            mask = y_true != 0

            if not np.any(mask):
                if verbose:
                    warnings.warn("All actual values are zero - MRE undefined")
                return np.nan

            relative_errors = np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]
            if clip_threshold is not None:
                relative_errors = np.clip(relative_errors, None, clip_threshold)
            n_skipped = len(y_true) - np.sum(mask)

            if n_skipped > 0 and verbose:
                print(f"  Note: Skipped {n_skipped}/{len(y_true)} samples with actual=0 in MRE")

            return np.mean(relative_errors) * 100
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def calculate_mape(y_true, y_pred, strategy, eps):
        """Calculate Mean Absolute Percentage Error with zero handling."""
        if strategy == 'epsilon':
            relative_errors = np.abs((y_true - y_pred) / (y_true + eps))
            if clip_threshold is not None:
                relative_errors = np.clip(relative_errors, None, clip_threshold)
            return np.mean(relative_errors) * 100

        elif strategy == 'mask':
            mask = y_true != 0

            if not np.any(mask):
                if verbose:
                    warnings.warn("All actual values are zero - MAPE undefined")
                return np.nan

            relative_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
            if clip_threshold is not None:
                relative_errors = np.clip(relative_errors, None, clip_threshold)
            mape = np.mean(relative_errors) * 100

            n_skipped = len(y_true) - np.sum(mask)
            if n_skipped > 0 and verbose:
                print(f"  Note: Skipped {n_skipped}/{len(y_true)} samples with actual=0 in MAPE")

            return mape
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def calculate_r2(y_true, y_pred):
        """Calculate R² with edge case handling."""
        if np.var(y_true) == 0:
            if verbose:
                warnings.warn("True values have zero variance - R² undefined")
            return np.nan

        return r2_score(y_true, y_pred)

    def format_metric(value):
        """Format metric value, handling NaN and None gracefully."""
        if value is None or np.isnan(value):
            return "N/A"
        return f"{value:.4f}"

    # Always computable metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    medae = median_absolute_error(y_true, y_pred)

    # Handle division-by-zero metrics
    mre_val = calculate_mre(y_true, y_pred, strategy=zero_handling, eps=epsilon)
    mape_val = calculate_mape(y_true, y_pred, strategy=zero_handling, eps=epsilon)

    # R² with edge case handling
    r2 = calculate_r2(y_true, y_pred)

    if verbose:
        print(f"  MAE:   {mae:.6f}")
        print(f"  RMSE:  {rmse:.6f}")
        print(f"  MedAE: {medae:.6f}")
        print(f"  MSE:   {mse:.6f}")
        print(f"  MRE:   {format_metric(mre_val)}%")
        print(f"  MAPE:  {format_metric(mape_val)}%")
        print(f"  R²:    {format_metric(r2)}")

    return {
        'mae': mae,
        'rmse': rmse,
        'medae': medae,
        'mse': mse,
        'mre': mre_val,
        'mape': mape_val,
        'r2': r2
    }


def compare_models(models_dict, X_test, y_test, verbose=False):
    """
    Compare multiple models on the same test set.

    Parameters
    ----------
    models_dict : dict
        Mapping of model_name -> model (must have .predict()).
    X_test : array-like
        Test features.
    y_test : array-like
        Test targets.
    verbose : bool, default=False
        Whether to print per-model metrics.

    Returns
    -------
    pd.DataFrame
        Comparison table with models as rows and metrics as columns.
    """
    results = {}
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        metrics = evaluation_metrics(y_test, y_pred, verbose=verbose)
        results[name] = metrics

    return pd.DataFrame(results).T
