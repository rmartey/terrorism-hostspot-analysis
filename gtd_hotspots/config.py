"""
Configuration and constants for GTD terrorism hotspot prediction.

Centralizes paths, temporal boundaries, and feature definitions for reproducibility.
"""

from pathlib import Path
import pandas as pd

# -----------------------------------------------------------------------------
# Paths (relative to project root or configurable)
# -----------------------------------------------------------------------------
DEFAULT_DATA_PATH = Path("terrorism.csv")
DEFAULT_CLEANED_PATH = Path("gtd_cleaned.csv")
DEFAULT_MODEL_PATH = Path("terrorism_hotspot_model.pkl")
DEFAULT_SCALER_PATH = Path("feature_scaler.pkl")

# -----------------------------------------------------------------------------
# Temporal split (label-availability logic)
# -----------------------------------------------------------------------------
FUTURE_WINDOW_DAYS = 90
TRAIN_END = pd.Timestamp("2012-12-31")
VAL_END = pd.Timestamp("2014-12-31")
TEST_END = pd.Timestamp("2017-12-31")
VAL_START = pd.Timestamp("2013-01-01")
TEST_START = pd.Timestamp("2015-01-01")

# -----------------------------------------------------------------------------
# Hotspot target
# -----------------------------------------------------------------------------
HOTSPOT_THRESHOLD = 3  # ≥ this many attacks in future window → hotspot

# -----------------------------------------------------------------------------
# Core columns to keep from raw GTD (before cleaning)
# -----------------------------------------------------------------------------
CORE_FEATURES = [
    "eventid",
    "iyear",
    "imonth",
    "iday",
    "country",
    "country_txt",
    "region",
    "region_txt",
    "provstate",
    "city",
    "latitude",
    "longitude",
    "attacktype1",
    "attacktype1_txt",
    "targtype1",
    "targtype1_txt",
    "weaptype1",
    "weaptype1_txt",
    "nkill",
    "nwound",
    "success",
    "suicide",
    "multiple",
    "gname",
    "specificity",
    "vicinity",
    "doubtterr",
    "extended",
]

# -----------------------------------------------------------------------------
# Model feature columns (used for X)
# -----------------------------------------------------------------------------
TEMPORAL_FEATURE_NAMES = [
    "year",
    "month",
    "quarter",
    "day_of_year",
    "decade",
]
GEOGRAPHIC_FEATURE_NAMES = [
    "latitude",
    "longitude",
    "region_code",
    "country_code",
]
HISTORY_FEATURE_NAMES = [
    "attacks_last_30d",
    "attacks_last_90d",
    "attacks_last_180d",
    "attacks_last_365d",
    "casualties_last_30d",
    "casualties_last_90d",
    "casualties_last_180d",
    "casualties_last_365d",
]
ATTACK_FEATURE_NAMES = [
    "attacktype1",
    "targtype1",
    "weaptype1",
    "success",
    "suicide",
    "multiple",
    "is_bombing",
    "is_armed_assault",
    "is_assassination",
    "targets_civilians",
    "targets_military",
    "targets_government",
]
PERPETRATOR_FEATURE_NAMES = [
    "group_region_attacks",
    "group_avg_casualties",
]
IMPACT_FEATURE_NAMES = ["nkill", "nwound", "total_casualties"]

FEATURE_COLUMNS = (
    TEMPORAL_FEATURE_NAMES
    + GEOGRAPHIC_FEATURE_NAMES
    + HISTORY_FEATURE_NAMES
    + ATTACK_FEATURE_NAMES
    + PERPETRATOR_FEATURE_NAMES
    + IMPACT_FEATURE_NAMES
)

# Rolling windows (days) for attack/casualty history
ROLLING_WINDOWS = [30, 90, 180, 365]
