"""Feature preparation utilities."""

import numpy as np
import pandas as pd


def prepare_data(df, feature_cols, target_col, feature_scaler=None, target_scaler=None, fit=True):
    """
    Extract features and target from a DataFrame, optionally scaling both.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    feature_cols : list of str
        Column names for features.
    target_col : str
        Column name for the target variable.
    feature_scaler : sklearn scaler or None
        Scaler for features (e.g., MinMaxScaler, StandardScaler).
        If None, no feature scaling is applied.
    target_scaler : sklearn scaler or None
        Scaler for target (e.g., MinMaxScaler, StandardScaler).
        If None, no target scaling is applied.
    fit : bool, default=True
        If True, fit the scalers on the data. If False, use pre-fitted scalers
        (for transforming test data with scalers fitted on training data).

    Returns
    -------
    X : np.ndarray
        Feature array (scaled if feature_scaler provided).
    y : np.ndarray
        Target array (scaled if target_scaler provided).
    feature_scaler : scaler or None
        Fitted feature scaler (or None if not provided).
    target_scaler : scaler or None
        Fitted target scaler (or None if not provided).
    """
    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)

    if feature_scaler is not None:
        if fit:
            X = feature_scaler.fit_transform(X)
        else:
            X = feature_scaler.transform(X)

    if target_scaler is not None:
        if fit:
            y = target_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        else:
            y = target_scaler.transform(y.reshape(-1, 1)).ravel()

    return X, y, feature_scaler, target_scaler
