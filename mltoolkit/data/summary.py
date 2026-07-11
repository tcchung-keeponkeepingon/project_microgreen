"""Data summary utilities."""

import numpy as np
import pandas as pd


def data_summary(df, feature_cols=None, target_col=None):
    """
    Generate a summary of a DataFrame including shape, dtypes, missing values,
    descriptive statistics, and correlations.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    feature_cols : list of str or None
        Feature columns to include in correlation analysis. If None, uses all numeric columns.
    target_col : str or None
        Target column for target-specific statistics.

    Returns
    -------
    dict
        Summary with keys: 'shape', 'dtypes', 'missing', 'stats', 'correlations'.
    """
    summary = {
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'stats': df.describe().to_dict(),
    }

    numeric_cols = feature_cols if feature_cols else df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col not in numeric_cols:
        numeric_cols = numeric_cols + [target_col]

    if len(numeric_cols) > 1:
        summary['correlations'] = df[numeric_cols].corr().to_dict()
    else:
        summary['correlations'] = {}

    if target_col and target_col in df.columns:
        target = df[target_col]
        summary['target'] = {
            'mean': float(target.mean()),
            'std': float(target.std()),
            'min': float(target.min()),
            'max': float(target.max()),
            'median': float(target.median()),
            'skew': float(target.skew()),
            'kurtosis': float(target.kurtosis()),
        }

    # Print formatted output
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    n_missing = sum(v for v in summary['missing'].values())
    if n_missing > 0:
        print(f"Missing values: {n_missing}")
        for col, count in summary['missing'].items():
            if count > 0:
                print(f"  {col}: {count}")
    else:
        print("Missing values: none")

    if target_col and 'target' in summary:
        t = summary['target']
        print(f"\nTarget '{target_col}':")
        print(f"  mean={t['mean']:.4f}, std={t['std']:.4f}, "
              f"min={t['min']:.4f}, max={t['max']:.4f}")
        print(f"  skew={t['skew']:.4f}, kurtosis={t['kurtosis']:.4f}")

    return summary
