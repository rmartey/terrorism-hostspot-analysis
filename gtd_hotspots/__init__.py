"""
GTD Hotspots: modular package for terrorism hotspot prediction.

Use the pipeline from the notebook or run_pipeline.py, or import submodules:

  from gtd_hotspots.data import load_gtd_data, clean_gtd_data
  from gtd_hotspots.features import build_features, create_hotspot_labels
  from gtd_hotspots.modeling import temporal_split, prepare_xy
  from gtd_hotspots.config import FUTURE_WINDOW_DAYS, FEATURE_COLUMNS
"""

from gtd_hotspots.config import (
    FUTURE_WINDOW_DAYS,
    TRAIN_END,
    VAL_END,
    TEST_END,
    CORE_FEATURES,
    FEATURE_COLUMNS,
    HOTSPOT_THRESHOLD,
)

__all__ = [
    "FUTURE_WINDOW_DAYS",
    "TRAIN_END",
    "VAL_END",
    "TEST_END",
    "CORE_FEATURES",
    "FEATURE_COLUMNS",
    "HOTSPOT_THRESHOLD",
]
