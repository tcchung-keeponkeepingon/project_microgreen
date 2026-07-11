import numpy as np
import pandas as pd


def _extract_single(
    strain_arr: np.ndarray,
    stress_arr: np.ndarray,
    start_x: float,
    x_signal: float,
    tolerance: float
) -> tuple[float, float]:
    target_x = start_x + x_signal
    if target_x < strain_arr.min() or target_x > strain_arr.max():
        return target_x, np.nan
    differences = np.abs(strain_arr - target_x)
    closest_idx = np.argmin(differences)
    if differences[closest_idx] <= tolerance:
        return target_x, stress_arr[closest_idx]
    return target_x, np.nan


def extract_stress_at_target(
    strain: pd.Series,
    stress: pd.Series,
    start_x: float,
    x_signal: float | list[float],
    tolerance: float = 0.5
) -> tuple[float, float] | dict[float, tuple[float, float]]:
    """
    Extract stress value at target strain(s).
    If x_signal is float: returns (target_x, y_value).
    If x_signal is list: returns {x_signal: (target_x, y_value), ...}.
    Returns NaN for stress if target_x is outside data range or not found within tolerance.
    """
    if np.isnan(start_x):
        if isinstance(x_signal, list):
            return {xs: (np.nan, np.nan) for xs in x_signal}
        return np.nan, np.nan

    strain_arr = strain.values
    stress_arr = stress.values

    if isinstance(x_signal, list):
        return {xs: _extract_single(strain_arr, stress_arr, start_x, xs, tolerance)
                for xs in x_signal}

    return _extract_single(strain_arr, stress_arr, start_x, x_signal, tolerance)
