"""
Utilities: display settings, warnings, and reproducibility.
"""

import warnings
from typing import Optional

import pandas as pd


def set_display_options(
    max_columns: Optional[int] = None,
    max_rows: int = 100,
    max_colwidth: Optional[int] = None,
) -> None:
    """Set pandas display options for notebooks/scripts."""
    if max_columns is not None:
        pd.set_option("display.max_columns", max_columns)
    pd.set_option("display.max_rows", max_rows)
    if max_colwidth is not None:
        pd.set_option("display.max_colwidth", max_colwidth)


def suppress_warnings(category: type = Warning) -> None:
    """Filter warnings (e.g. ignore)."""
    warnings.filterwarnings("ignore", category=category)
