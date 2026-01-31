"""
Feature engineering: temporal, geographic, attack history, attack type, perpetrator.
"""

from typing import List, Optional

import pandas as pd

from gtd_hotspots.config import ROLLING_WINDOWS


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add year, month, day, day_of_year, quarter, decade, days_since_start, period."""
    out = df.copy()
    out["year"] = out["iyear"]
    out["month"] = out["imonth"]
    out["day"] = out["iday"]
    if "date" in out.columns:
        out["day_of_year"] = out["date"].dt.dayofyear
        out["quarter"] = out["date"].dt.quarter
        out["days_since_start"] = (out["date"] - out["date"].min()).dt.days
    out["decade"] = (out["year"] // 10) * 10
    out["period"] = pd.cut(
        out["year"],
        bins=[1969, 1980, 1990, 2000, 2010, 2018],
        labels=["1970s", "1980s", "1990s", "2000s", "2010s"],
    )
    return out


def add_geographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lat_grid, lon_grid, grid_cell, grid_cell_fine, region_code, country_code."""
    out = df.copy()
    out["lat_grid"] = (out["latitude"] // 1).astype(int)
    out["lon_grid"] = (out["longitude"] // 1).astype(int)
    out["grid_cell"] = (
        out["lat_grid"].astype(str) + "_" + out["lon_grid"].astype(str)
    )
    out["lat_grid_fine"] = (out["latitude"] // 0.5).astype(int)
    out["lon_grid_fine"] = (out["longitude"] // 0.5).astype(int)
    out["grid_cell_fine"] = (
        out["lat_grid_fine"].astype(str)
        + "_"
        + out["lon_grid_fine"].astype(str)
    )
    out["region_code"] = out["region"]
    out["country_code"] = out["country"]
    return out


def add_attack_history_features(
    df: pd.DataFrame,
    windows: Optional[List[int]] = None,
    count_col: str = "eventid",
) -> pd.DataFrame:
    """
    Add rolling attack counts and casualty sums per grid_cell over time.
    Requires date and grid_cell; df must be sorted by (grid_cell, date).
    """
    if windows is None:
        windows = ROLLING_WINDOWS
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).copy()
    out = out.sort_values(["grid_cell", "date"]).reset_index(drop=True)
    out["grid_cell"] = out["grid_cell"].astype("category")

    if count_col not in out.columns:
        out["_one"] = 1
        count_col = "_one"

    df_idx = out.set_index("date")
    for w in windows:
        col_name = f"attacks_last_{w}d"
        df_idx[col_name] = (
            df_idx.groupby("grid_cell")[count_col]
            .rolling(f"{w}D")
            .count()
            .shift(1)
            .reset_index(level=0, drop=True)
            .fillna(0)
            .astype("int64")
        )
    df_idx["total_casualties"] = pd.to_numeric(
        df_idx["total_casualties"], errors="coerce"
    ).fillna(0)
    for w in windows:
        col_name = f"casualties_last_{w}d"
        df_idx[col_name] = (
            df_idx.groupby("grid_cell")["total_casualties"]
            .rolling(f"{w}D")
            .sum()
            .shift(1)
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
    out = df_idx.reset_index()
    if "_one" in out.columns:
        out = out.drop(columns=["_one"])
    return out


def add_attack_type_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_bombing, is_armed_assault, is_assassination, targets_* dummies."""
    out = df.copy()
    out["is_bombing"] = (out["attacktype1"] == 3).astype(int)
    out["is_armed_assault"] = (out["attacktype1"] == 2).astype(int)
    out["is_assassination"] = (out["attacktype1"] == 1).astype(int)
    out["targets_civilians"] = (out["targtype1"] == 14).astype(int)
    out["targets_military"] = (out["targtype1"] == 4).astype(int)
    out["targets_government"] = (out["targtype1"] == 7).astype(int)
    return out


def add_perpetrator_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add group_region_attacks and group_avg_casualties (per group)."""
    out = df.copy()
    region_attacks = out.groupby("gname")["region"].transform("count")
    out["group_region_attacks"] = region_attacks
    group_casualties = out.groupby("gname")["total_casualties"].transform("mean")
    out["group_avg_casualties"] = group_casualties
    return out


def build_all_features(
    df: pd.DataFrame,
    rolling_windows: Optional[List[int]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run all feature engineering steps in order.

    Order: temporal → geographic → attack history → attack type → perpetrator.
    """
    out = add_temporal_features(df)
    if verbose:
        print("Temporal features added.")
    out = add_geographic_features(out)
    if verbose:
        print("Geographic features added.")
    out = add_attack_history_features(out, windows=rolling_windows)
    if verbose:
        print("Attack history features added.")
    out = add_attack_type_features(out)
    out = add_perpetrator_features(out)
    if verbose:
        print("Attack type and perpetrator features added.")
    return out
