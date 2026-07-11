"""Data splitting utilities."""

import pandas as pd
from typing import Dict, List


def split_by_ids(df, id_column, groups_dict):
    """
    Split a DataFrame into groups based on ID membership.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    id_column : str
        Column containing the IDs to split on.
    groups_dict : dict
        Mapping of group_name -> list of IDs.
        Example: {'train': ['A', 'B', 'C'], 'test': ['D', 'E']}

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping of group_name -> filtered DataFrame.
    """
    result = {}
    for name, ids in groups_dict.items():
        result[name] = df[df[id_column].isin(ids)].copy()
    return result
