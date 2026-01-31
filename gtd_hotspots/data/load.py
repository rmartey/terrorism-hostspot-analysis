"""
Load Global Terrorism Database (GTD) from CSV.

Handles encoding and optional column rename for BOM in eventid.
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from gtd_hotspots.config import CORE_FEATURES


def load_gtd_data(
    path: Optional[Union[str, Path]] = None,
    encoding: str = "ISO-8859-1",
    low_memory: bool = False,
    use_core_columns: bool = False,
) -> pd.DataFrame:
    """
    Load the GTD dataset from CSV.

    Parameters
    ----------
    path : str or Path, optional
        Path to terrorism.csv. Defaults to config.DEFAULT_DATA_PATH.
    encoding : str
        File encoding (GTD often uses ISO-8859-1).
    low_memory : bool
        Passed to pandas read_csv for large files.
    use_core_columns : bool
        If True, keep only CORE_FEATURES columns after load.

    Returns
    -------
    pd.DataFrame
        Raw or core-subset GTD dataframe.
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent.parent / "terrorism.csv"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GTD data not found: {path}")

    df = pd.read_csv(path, encoding=encoding, low_memory=low_memory)

    # Handle BOM in first column name (e.g. ï»¿eventid -> eventid)
    rename = {}
    for c in df.columns:
        if "eventid" in c and c != "eventid":
            rename[c] = "eventid"
            break
    if rename:
        df = df.rename(columns=rename)

    if use_core_columns:
        available = [c for c in CORE_FEATURES if c in df.columns]
        df = df[available].copy()

    return df
