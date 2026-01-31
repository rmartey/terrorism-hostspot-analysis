"""
Prepare X, y and apply temporal split; optional alignment check.
"""

from typing import Optional, Tuple

import pandas as pd

from gtd_hotspots.features.selection import prepare_X_y
from gtd_hotspots.modeling.split import temporal_split


def prepare_splits(
    df: pd.DataFrame,
    target_col: str = "is_hotspot",
    date_col: str = "date",
    future_window_days: int = 90,
    drop_na: bool = True,
    assert_aligned: bool = True,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series,
    pd.Series, pd.Series, pd.Series,
    pd.DataFrame, pd.Series,
]:
    """
    Build X, y from df and split into train/val/test by time.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned + engineered dataframe with is_hotspot and feature columns.
    target_col : str
        Target column name.
    date_col : str
        Date column for temporal split.
    future_window_days : int
        Passed to temporal_split.
    drop_na : bool
        Passed to prepare_X_y.
    assert_aligned : bool
        If True, assert len(df) == len(X) == len(y) before splitting.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test,
    train_mask, val_mask, test_mask,
    X, y
    """
    X, y = prepare_X_y(df, target_col=target_col, drop_na=drop_na)
    # After drop_na, X/y have subset of df index; align df to X.index for split
    df_aligned = df.loc[X.index].copy()
    if assert_aligned:
        assert len(df_aligned) == len(X) == len(y), (
            "df, X, y length mismatch after alignment."
        )

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        train_mask, val_mask, test_mask,
    ) = temporal_split(
        df_aligned, X, y,
        date_col=date_col,
        future_window_days=future_window_days,
    )
    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        train_mask, val_mask, test_mask,
        X, y,
    )
