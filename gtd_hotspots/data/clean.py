"""
Clean GTD data: date, geospatial, casualty, categorical, and binary handling.

Pipeline is reproducible and can be run step-by-step or via clean_gtd_data().
"""

from typing import List, Optional

import pandas as pd

from gtd_hotspots.config import CORE_FEATURES


def _create_date(row: pd.Series) -> pd.Timestamp:
    """Create date from iyear/imonth/iday; default unknowns to mid-period."""
    year = row["iyear"]
    month = row["imonth"] if row["imonth"] > 0 else 6
    day = row["iday"] if row["iday"] > 0 else 15
    try:
        return pd.to_datetime(f"{year}-{int(month):02d}-{int(day):02d}")
    except Exception:
        return pd.NaT


def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Add date column and date_uncertain flag. Does not drop rows."""
    out = df.copy()
    out["date"] = out.apply(_create_date, axis=1)
    out["date_uncertain"] = (
        (out["imonth"] == 0) | (out["iday"] == 0)
    ).astype(int)
    return out


def clean_geospatial(df: pd.DataFrame) -> pd.DataFrame:
    """Drop missing/invalid coordinates; add low_precision flag."""
    out = df.dropna(subset=["latitude", "longitude"]).copy()
    out = out[
        (out["latitude"] >= -90)
        & (out["latitude"] <= 90)
        & (out["longitude"] >= -180)
        & (out["longitude"] <= 180)
    ]
    out["low_precision"] = (out["specificity"] > 3).astype(int)
    return out


def _categorize_severity(casualties: float) -> str:
    if casualties == 0:
        return "None"
    if casualties <= 5:
        return "Low"
    if casualties <= 20:
        return "Medium"
    if casualties <= 50:
        return "High"
    return "Extreme"


def clean_casualties(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing nkill/nwound with 0; add total_casualties and severity."""
    out = df.copy()
    out["nkill"] = out["nkill"].fillna(0)
    out["nwound"] = out["nwound"].fillna(0)
    out["total_casualties"] = out["nkill"] + out["nwound"]
    out["severity"] = out["total_casualties"].apply(_categorize_severity)
    return out


def clean_categorical(
    df: pd.DataFrame,
    text_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Fill missing gname/city/provstate; standardize text columns."""
    if text_columns is None:
        text_columns = [
            "country_txt",
            "region_txt",
            "city",
            "provstate",
            "attacktype1_txt",
            "targtype1_txt",
            "weaptype1_txt",
            "gname",
        ]
    out = df.copy()
    out["gname"] = out["gname"].fillna("Unknown")
    out["city"] = out["city"].fillna("Unknown")
    out["provstate"] = out["provstate"].fillna("Unknown")
    for col in text_columns:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip().str.title()
    return out


def clean_binary(
    df: pd.DataFrame,
    binary_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Fill missing binary cols with 0 and cast to int."""
    if binary_cols is None:
        binary_cols = ["success", "suicide", "multiple", "extended", "doubtterr", "vicinity"]
    out = df.copy()
    for col in binary_cols:
        if col not in out.columns:
            continue
        if out[col].isnull().any():
            out[col] = out[col].fillna(0)
        out[col] = out[col].astype(int)
    return out


def clean_gtd_data(
    df: pd.DataFrame,
    core_columns: Optional[List[str]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run full cleaning pipeline: subset → date → geospatial → casualty → categorical → binary.

    Parameters
    ----------
    df : pd.DataFrame
        Raw GTD dataframe (or already subset to core columns).
    core_columns : list, optional
        Columns to keep before cleaning. Defaults to config.CORE_FEATURES.
    verbose : bool
        If True, print record counts after each step.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for feature engineering.
    """
    if core_columns is None:
        core_columns = CORE_FEATURES
    available = [c for c in core_columns if c in df.columns]
    out = df[available].copy()
    if verbose:
        print(f"Subset to {len(available)} columns, shape {out.shape}")

    out = clean_dates(out)
    if verbose:
        print(f"After date cleaning: {len(out)} rows")

    out = clean_geospatial(out)
    if verbose:
        print(f"After geospatial cleaning: {len(out)} rows")

    out = clean_casualties(out)
    out = clean_categorical(out)
    out = clean_binary(out)
    return out
