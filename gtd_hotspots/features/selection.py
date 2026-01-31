"""
Feature selection: list of model features and X/y preparation.
"""

from typing import List, Optional, Tuple

import pandas as pd

from gtd_hotspots.config import FEATURE_COLUMNS


def get_feature_columns(
    df: pd.DataFrame,
    feature_list: Optional[List[str]] = None,
) -> List[str]:
    """
    Return list of feature column names that exist in df.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with engineered features.
    feature_list : list, optional
        Defaults to config.FEATURE_COLUMNS.

    Returns
    -------
    list
        Subset of feature_list that exists in df.
    """
    if feature_list is None:
        feature_list = FEATURE_COLUMNS
    return [c for c in feature_list if c in df.columns]


def prepare_X_y(
    df: pd.DataFrame,
    target_col: str = "is_hotspot",
    feature_list: Optional[List[str]] = None,
    drop_na: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and target y from cleaned/engineered df.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain target_col and all selected feature columns.
    target_col : str
        Name of target column (0/1).
    feature_list : list, optional
        Columns to use as features. Defaults to get_feature_columns(df).
    drop_na : bool
        If True, drop rows with NaN in any feature or target.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (aligned with df index where kept).
    y : pd.Series
        Target series.
    """
    cols = get_feature_columns(df, feature_list=feature_list)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in dataframe.")
    use = cols + [target_col]
    subset = df[use].copy()
    if drop_na:
        subset = subset.dropna()
    X = subset[cols]
    y = subset[target_col]
    return X, y
