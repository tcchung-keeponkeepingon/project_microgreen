import pandas as pd
import numpy as np

def detect_outliers_iqr(values, multiplier=1.5):
    """Detect outliers using IQR method."""
    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outlier_mask = (values < lower_bound) | (values > upper_bound)
    
    return outlier_mask, {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }


def detect_outliers_mad(values, multiplier=3):
    """Detect outliers using MAD (Median Absolute Deviation) method."""
    median = values.median()
    mad = np.median(np.abs(values - median))
    
    # Handle case where MAD is 0
    if mad == 0:
        mad = np.mean(np.abs(values - median))
    
    lower_bound = median - multiplier * mad
    upper_bound = median + multiplier * mad
    
    outlier_mask = (values < lower_bound) | (values > upper_bound)
    
    return outlier_mask, {
        'median': median,
        'MAD': mad,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }


def detect_outliers_percentile(values, lower_percentile=1, upper_percentile=99):
    """Detect outliers using percentile method."""
    lower_bound = values.quantile(lower_percentile / 100)
    upper_bound = values.quantile(upper_percentile / 100)
    
    outlier_mask = (values < lower_bound) | (values > upper_bound)
    
    return outlier_mask, {
        'lower_percentile': lower_percentile,
        'upper_percentile': upper_percentile,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }


def detect_outliers(data, column, method='iqr', 
                   iqr_multiplier=1.5, 
                   mad_multiplier=3,
                   lower_percentile=1, 
                   upper_percentile=99):
    """
    Detect outliers using various methods.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe
    column : str
        Column name to check for outliers
    method : str
        Method to use: 'iqr', 'mad', 'percentile', or 'all'
        If 'all', a row is considered an outlier if >=2 methods agree
    iqr_multiplier : float
        Multiplier for IQR method (default: 1.5)
    mad_multiplier : float
        Multiplier for MAD method (default: 3)
    lower_percentile : float
        Lower percentile threshold (default: 1)
    upper_percentile : float
        Upper percentile threshold (default: 99)
    
    Returns:
    --------
    outlier_info : dict
        Contains outlier indices, statistics, and method details
    removed_rows : pd.DataFrame
        Rows identified as outliers
    cleaned_df : pd.DataFrame
        Dataframe with outliers removed
    """
    values = data[column].dropna()
    
    if method == 'iqr':
        outlier_mask, stats = detect_outliers_iqr(values, iqr_multiplier)
        method_info = {'method': 'IQR', 'stats': stats}
        
    elif method == 'mad':
        outlier_mask, stats = detect_outliers_mad(values, mad_multiplier)
        method_info = {'method': 'MAD', 'stats': stats}
        
    elif method == 'percentile':
        outlier_mask, stats = detect_outliers_percentile(values, lower_percentile, upper_percentile)
        method_info = {'method': 'Percentile', 'stats': stats}
        
    elif method == 'all':
        # Apply all three methods
        iqr_mask, iqr_stats = detect_outliers_iqr(values, iqr_multiplier)
        mad_mask, mad_stats = detect_outliers_mad(values, mad_multiplier)
        pct_mask, pct_stats = detect_outliers_percentile(values, lower_percentile, upper_percentile)
        
        # Count votes for each data point
        vote_count = iqr_mask.astype(int) + mad_mask.astype(int) + pct_mask.astype(int)
        outlier_mask = vote_count >= 2
        
        method_info = {
            'method': 'All (>=2 votes)',
            'stats': {
                'IQR': iqr_stats,
                'MAD': mad_stats,
                'Percentile': pct_stats
            },
            'vote_breakdown': {
                'iqr_outliers': iqr_mask.sum(),
                'mad_outliers': mad_mask.sum(),
                'percentile_outliers': pct_mask.sum(),
                'consensus_outliers': outlier_mask.sum()
            }
        }
    else:
        raise ValueError(f"Invalid method: {method}. Choose from 'iqr', 'mad', 'percentile', or 'all'")
    
    # Map outlier mask back to original dataframe indices
    full_outlier_mask = pd.Series(False, index=data.index)
    full_outlier_mask.loc[values.index] = outlier_mask
    
    outlier_indices = data[full_outlier_mask].index.tolist()
    
    # Prepare outputs
    outlier_info = {
        'outlier_indices': outlier_indices,
        'n_outliers': len(outlier_indices),
        'column': column,
        **method_info
    }
    
    removed_rows = data.loc[full_outlier_mask].copy()
    cleaned_df = data[~full_outlier_mask].copy()
    
    return outlier_info, removed_rows, cleaned_df