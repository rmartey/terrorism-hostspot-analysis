"""
Temporal train/validation/test split with label-availability logic.

Splits respect FUTURE_WINDOW_DAYS: we only assign a row to train/val/test if
we can look that many days ahead without crossing the period boundary.
"""

from typing import Optional, Tuple

import pandas as pd

from gtd_hotspots.config import (
    FUTURE_WINDOW_DAYS,
    TRAIN_END,
    VAL_END,
    TEST_END,
    VAL_START,
    TEST_START,
)


def get_train_val_test_masks(
    df: pd.DataFrame,
    date_col: str = "date",
    future_window_days: int = FUTURE_WINDOW_DAYS,
    train_end: Optional[pd.Timestamp] = None,
    val_end: Optional[pd.Timestamp] = None,
    test_end: Optional[pd.Timestamp] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute boolean masks for train, validation, and test periods.

    Label-availability: a row is in train only if date <= train_end - future_window_days,
    so we have room to compute the future hotspot label.

    Parameters
    ----------
    df : pd.DataFrame
        Must have date_col as datetime.
    date_col : str
        Name of date column.
    future_window_days : int
        Days ahead used for hotspot label.
    train_end, val_end, test_end : pd.Timestamp, optional
        Period boundaries; default to config.

    Returns
    -------
    train_mask, val_mask, test_mask : pd.Series
        Boolean masks aligned with df.index.
    """
    train_end = train_end or TRAIN_END
    val_end = val_end or VAL_END
    test_end = test_end or TEST_END
    delta = pd.Timedelta(days=future_window_days)

    train_mask = df[date_col] <= train_end - delta
    val_mask = (
        (df[date_col] >= VAL_START)
        & (df[date_col] <= val_end - delta)
    )
    test_mask = (
        (df[date_col] >= TEST_START)
        & (df[date_col] <= test_end - delta)
    )

    return train_mask, val_mask, test_mask


def temporal_split(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    date_col: str = "date",
    future_window_days: int = FUTURE_WINDOW_DAYS,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series,
    pd.Series, pd.Series, pd.Series,
]:
    """
    Split df, X, y into train/val/test by temporal masks.

    Returns
    -------
    X_train, X_val, X_test : pd.DataFrame
    y_train, y_val, y_test : pd.Series
    train_mask, val_mask, test_mask : pd.Series
    """
    train_mask, val_mask, test_mask = get_train_val_test_masks(
        df, date_col=date_col, future_window_days=future_window_days
    )

    def _slice(data, mask):
        if hasattr(data, "loc"):
            return data.loc[mask].copy()
        return data[mask].copy()

    X_train = _slice(X, train_mask)
    X_val = _slice(X, val_mask)
    X_test = _slice(X, test_mask)
    y_train = _slice(y, train_mask)
    y_val = _slice(y, val_mask)
    y_test = _slice(y, test_mask)

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        train_mask, val_mask, test_mask,
    )


def split_percentages(
    df: pd.DataFrame,
    train_mask: pd.Series,
    val_mask: pd.Series,
    test_mask: pd.Series,
) -> Tuple[float, float, float]:
    """Return (train_pct, val_pct, test_pct) of len(df)."""
    n = len(df)
    train_pct = train_mask.sum() / n * 100
    val_pct = val_mask.sum() / n * 100
    test_pct = test_mask.sum() / n * 100
    return train_pct, val_pct, test_pct
