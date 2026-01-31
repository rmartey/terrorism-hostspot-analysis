"""
Create hotspot target labels: is_hotspot = 1 if ≥ threshold attacks in same grid in future window.
"""

from typing import Optional

import numpy as np
import pandas as pd


def create_hotspot_labels(
    df: pd.DataFrame,
    grid_col: str = "grid_cell",
    date_col: str = "date",
    future_window: int = 90,
    threshold: int = 3,
) -> pd.DataFrame:
    """
    Label each record as hotspot if the same grid cell has ≥ threshold attacks
    in (date, date + future_window].

    Parameters
    ----------
    df : pd.DataFrame
        Must have grid_col and date_col; date must be datetime.
    grid_col : str
        Column identifying grid cell.
    date_col : str
        Date column name.
    future_window : int
        Number of days ahead to count attacks.
    threshold : int
        Minimum future attacks in same grid to label as hotspot.

    Returns
    -------
    pd.DataFrame
        Copy of df with added column 'is_hotspot' (0/1).
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col, grid_col]).copy()
    out = out.sort_values([grid_col, date_col]).reset_index(drop=True)

    window = pd.Timedelta(days=future_window)
    is_hotspot = np.zeros(len(out), dtype=np.int8)

    for _g, idx in out.groupby(grid_col, sort=False).groups.items():
        dates = out.loc[idx, date_col].values.astype("datetime64[ns]")
        end_pos = np.searchsorted(
            dates, dates + window.to_timedelta64(), side="right"
        )
        future_counts = end_pos - np.arange(len(dates)) - 1
        is_hotspot[idx] = (future_counts >= threshold).astype(np.int8)

    out["is_hotspot"] = is_hotspot
    return out
