#!/usr/bin/env python3
"""
Reproducible pipeline for GTD terrorism hotspot prediction.

Run from project root:
  python run_pipeline.py [--data terrorism.csv] [--out-dir .]

Steps: load → clean → features → target → split → (training in notebook).
"""

import argparse
from pathlib import Path

from gtd_hotspots.config import FUTURE_WINDOW_DAYS, HOTSPOT_THRESHOLD
from gtd_hotspots.data import load_gtd_data, clean_gtd_data
from gtd_hotspots.features import build_all_features, create_hotspot_labels, prepare_X_y
from gtd_hotspots.modeling import get_train_val_test_masks, split_percentages, prepare_splits
from gtd_hotspots.utils import set_display_options, suppress_warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="GTD hotspot pipeline: load, clean, featurize, split.")
    parser.add_argument("--data", type=Path, default=Path("terrorism.csv"), help="Path to terrorism.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="Output directory for gtd_cleaned.csv")
    parser.add_argument("--verbose", action="store_true", help="Print step summaries")
    args = parser.parse_args()

    suppress_warnings()
    set_display_options(max_columns=None, max_rows=100)

    # 1) Load
    df = load_gtd_data(path=args.data)
    if args.verbose:
        print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # 2) Clean
    df_clean = clean_gtd_data(df, verbose=args.verbose)

    # 3) Features
    df_clean = build_all_features(df_clean, verbose=args.verbose)

    # 4) Target
    df_clean = create_hotspot_labels(
        df_clean,
        future_window=FUTURE_WINDOW_DAYS,
        threshold=HOTSPOT_THRESHOLD,
    )
    if args.verbose:
        print(f"Hotspot rate: {df_clean['is_hotspot'].mean():.4f}")

    # 5) Prepare X, y and split
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        train_mask, val_mask, test_mask,
        X, y,
    ) = prepare_splits(
        df_clean,
        future_window_days=FUTURE_WINDOW_DAYS,
        drop_na=True,
        assert_aligned=True,
    )
    train_pct, val_pct, test_pct = split_percentages(
        df_clean.loc[X.index], train_mask, val_mask, test_mask
    )
    if args.verbose:
        print(f"Train: {len(y_train):,} ({train_pct:.1f}%)")
        print(f"Val:   {len(y_val):,} ({val_pct:.1f}%)")
        print(f"Test:  {len(y_test):,} ({test_pct:.1f}%)")

    # 6) Save cleaned data for notebook / downstream
    out_path = args.out_dir / "gtd_cleaned.csv"
    df_clean.to_csv(out_path, index=False)
    if args.verbose:
        print(f"Saved: {out_path}")

    print("Pipeline complete. Use GTD Analysis.ipynb for modeling and evaluation.")


if __name__ == "__main__":
    main()
