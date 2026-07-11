"""Configurable compositional design space grid generation."""

import numpy as np
import pandas as pd
from itertools import product


def create_design_space(compositional_cols, scalar_cols=None, n_points=21, constraints=None):
    """
    Generate a design space grid where compositional columns sum to 1.0,
    optionally crossed with discrete scalar columns.

    Parameters
    ----------
    compositional_cols : list of str
        Names of the compositional fraction columns (must sum to 1.0).
    scalar_cols : dict or None, default=None
        Mapping of column_name -> list of discrete values.
        Example: {'MassLoading': [15, 20]}
        Each combination is replicated via Cartesian product.
    n_points : int, default=21
        Number of grid points per compositional dimension (0 to 1 inclusive).
    constraints : dict or None, default=None
        Per-column constraints as {col_name: (min_val, max_val)}.
        Only compositions satisfying all constraints are kept.

    Returns
    -------
    pd.DataFrame
        Design space with compositional + scalar columns.
    """
    n_materials = len(compositional_cols)
    grid = np.linspace(0, 1, n_points)

    # Generate meshgrid for n-1 dimensions
    coords = np.stack(
        [g.flatten() for g in np.meshgrid(*[grid] * (n_materials - 1))],
    ).T

    # Keep compositions that sum to <= 1
    valid = coords[coords.sum(axis=1) <= 1.0 + 1e-10]
    last_fraction = (1 - valid.sum(axis=1)).reshape(-1, 1)
    # Clip to avoid tiny negative values from floating point
    last_fraction = np.clip(last_fraction, 0.0, 1.0)
    compositions = np.hstack([valid, last_fraction])

    df = pd.DataFrame(compositions, columns=compositional_cols)

    # Apply constraints
    if constraints:
        mask = np.ones(len(df), dtype=bool)
        for col, (lo, hi) in constraints.items():
            if col in df.columns:
                mask &= (df[col] >= lo - 1e-10) & (df[col] <= hi + 1e-10)
        df = df[mask].reset_index(drop=True)

    # Cross with scalar columns
    if scalar_cols:
        scalar_values = list(product(*scalar_cols.values()))
        scalar_names = list(scalar_cols.keys())
        grids = []
        for vals in scalar_values:
            g = df.copy()
            for name, val in zip(scalar_names, vals):
                g[name] = val
            grids.append(g)
        df = pd.concat(grids, ignore_index=True)

    return df
