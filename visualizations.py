#!/usr/bin/env python3
"""
Run from the terminal to produce answers to the 6 analytical questions
and save all visualizations under an output directory.

Usage:
  python visualizations.py [--data PATH] [--out-dir DIR] [--figures-dir DIR]

Output layout (default):
  output/
    gtd_cleaned.csv       (cached cleaned data, if built from raw)
    visualization/        (all figure PNGs)

If gtd_cleaned.csv exists in --out-dir, it is loaded; otherwise the script
runs the full pipeline from --data (terrorism.csv) and trains a small model
for Q5/Q6.
"""

import argparse
import sys
from pathlib import Path

# Non-interactive backend for saving figures when run from terminal
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Project package
from gtd_hotspots.config import FUTURE_WINDOW_DAYS, HOTSPOT_THRESHOLD
from gtd_hotspots.data import load_gtd_data, clean_gtd_data
from gtd_hotspots.features import (
    build_all_features,
    create_hotspot_labels,
    get_feature_columns,
    prepare_X_y,
)
from gtd_hotspots.modeling import prepare_splits
from gtd_hotspots.utils import suppress_warnings

from sklearn.metrics import roc_curve

suppress_warnings()


def load_or_build_data(data_path: Path, out_dir: Path) -> pd.DataFrame:
    """Load gtd_cleaned.csv if it exists; otherwise build from raw data."""
    cleaned_path = out_dir / "gtd_cleaned.csv"
    if cleaned_path.exists():
        print(f"Loading cleaned data from {cleaned_path} ...")
        return pd.read_csv(cleaned_path, parse_dates=["date"], low_memory=False)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Neither {cleaned_path} nor {data_path} found. "
            "Run from project directory or provide --data terrorism.csv"
        )
    print(f"Building pipeline from {data_path} ...")
    df = load_gtd_data(path=data_path)
    df_clean = clean_gtd_data(df, verbose=True)
    df_clean = build_all_features(df_clean, verbose=True)
    df_clean = create_hotspot_labels(
        df_clean,
        future_window=FUTURE_WINDOW_DAYS,
        threshold=HOTSPOT_THRESHOLD,
    )
    df_clean.to_csv(cleaned_path, index=False)
    print(f"Saved {cleaned_path}")
    return df_clean


def train_minimal_model_for_q5_q6(df_clean: pd.DataFrame):
    """Prepare splits and train a Random Forest for feature importance and test predictions."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, auc

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        train_mask,
        val_mask,
        test_mask,
        X,
        y,
    ) = prepare_splits(
        df_clean,
        future_window_days=FUTURE_WINDOW_DAYS,
        drop_na=True,
        assert_aligned=True,
    )
    feature_columns = get_feature_columns(df_clean)
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    feature_importance = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Importance": rf.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)
    final_test_pred = rf.predict(X_test)
    final_test_proba = rf.predict_proba(X_test)[:, 1]
    return {
        "feature_importance": feature_importance,
        "y_test": y_test,
        "final_test_pred": final_test_pred,
        "final_test_proba": final_test_proba,
    }


def q1_answer_and_plot(df_clean: pd.DataFrame, figures_dir: Path) -> None:
    """Q1: How has global terrorism changed over time (1970–2017)?"""
    print("\n" + "=" * 60)
    print("Q1: How has global terrorism changed over time (1970–2017)?")
    print("=" * 60)
    year_col = "year" if "year" in df_clean.columns else "iyear"
    attacks_per_year = df_clean.groupby(year_col).size()
    casualties_per_year = df_clean.groupby(year_col)["total_casualties"].sum()
    print("Answer: Attack counts and total casualties rise sharply from the 2000s,")
    print("with peaks in the 2010s. The data show strong temporal variation.")
    print(
        f"Peak attack year: {attacks_per_year.idxmax()} ({attacks_per_year.max():,} attacks)"
    )
    print(
        f"Peak casualty year: {casualties_per_year.idxmax()} ({casualties_per_year.max():,.0f} casualties)"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    attacks_per_year.plot(kind="line", color="darkred", linewidth=2, ax=ax1)
    ax1.set_title("Global Terrorism Attacks Over Time (1970–2017)", fontweight="bold")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of Attacks")
    ax1.grid(alpha=0.3)
    casualties_per_year.plot(kind="line", color="darkblue", linewidth=2, ax=ax2)
    ax2.set_title("Total Casualties Over Time", fontweight="bold")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Total Casualties")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "q1_temporal_trends.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {figures_dir / 'q1_temporal_trends.png'}")


def q2_answer_and_plot(df_clean: pd.DataFrame, figures_dir: Path) -> None:
    """Q2: Which countries and regions have the most attacks and casualties?"""
    print("\n" + "=" * 60)
    print("Q2: Which countries and regions have the most attacks and casualties?")
    print("=" * 60)
    top_countries = df_clean["country_txt"].value_counts().head(20)
    top_casualties = (
        df_clean.groupby("country_txt")["total_casualties"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
    )
    region_attacks = df_clean["region_txt"].value_counts()
    print(
        "Answer: Top countries by attacks and by total casualties are shown in the figures."
    )
    print("Top 3 by attacks:", list(top_countries.index[:3]))
    print("Top 3 by casualties:", list(top_casualties.index[:3]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(
        y=top_countries.index, x=top_countries.values, palette="Reds_r", ax=axes[0]
    )
    axes[0].set_title("Top 20 Countries by Attack Frequency", fontweight="bold")
    axes[0].set_xlabel("Number of Attacks")
    sns.barplot(
        y=top_casualties.index, x=top_casualties.values, palette="Blues_r", ax=axes[1]
    )
    axes[1].set_title("Top 20 Countries by Total Casualties", fontweight="bold")
    axes[1].set_xlabel("Total Casualties")
    plt.tight_layout()
    plt.savefig(figures_dir / "q2_countries.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.barh(
        region_attacks.index, region_attacks.values, color="teal", edgecolor="black"
    )
    plt.title("Attacks by Region", fontweight="bold")
    plt.xlabel("Number of Attacks")
    plt.tight_layout()
    plt.savefig(figures_dir / "q2_regions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(
        f"  -> Saved {figures_dir / 'q2_countries.png'}, {figures_dir / 'q2_regions.png'}"
    )


def q3_answer_and_plot(df_clean: pd.DataFrame, figures_dir: Path) -> None:
    """Q3: What are the most common attack types and which cause the most deaths?"""
    print("\n" + "=" * 60)
    print("Q3: What are the most common attack types and which cause the most deaths?")
    print("=" * 60)
    attack_types = df_clean["attacktype1_txt"].value_counts().head(10)
    deadly_types = (
        df_clean.groupby("attacktype1_txt")["nkill"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    print("Answer: Bombing/Explosion and Armed Assault are among the most frequent;")
    print("the deadliest types by total deaths are shown in the second chart.")
    print(
        "Most common:",
        attack_types.index[0],
        "| Deadliest (nkill):",
        deadly_types.index[0],
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(y=attack_types.index, x=attack_types.values, ax=ax1, palette="viridis")
    ax1.set_title("Most Common Attack Types", fontweight="bold")
    ax1.set_xlabel("Number of Attacks")
    sns.barplot(y=deadly_types.index, x=deadly_types.values, ax=ax2, palette="Reds_r")
    ax2.set_title("Attack Types by Total Deaths (Deadliest)", fontweight="bold")
    ax2.set_xlabel("Total Deaths (nkill)")
    plt.tight_layout()
    plt.savefig(figures_dir / "q3_attack_types.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {figures_dir / 'q3_attack_types.png'}")


def q4_answer_and_plot(df_clean: pd.DataFrame, figures_dir: Path) -> None:
    """Q4: Where do terrorism hotspots form, and how are they defined?"""
    print("\n" + "=" * 60)
    print("Q4: Where do terrorism hotspots form, and how are they defined?")
    print("=" * 60)
    if "is_hotspot" not in df_clean.columns:
        print(
            "Answer: Hotspot labels not in data. Run pipeline with create_hotspot_labels."
        )
        return
    hotspot_share = df_clean["is_hotspot"].value_counts()
    hotspot_by_region = (
        df_clean.groupby("region_txt")["is_hotspot"].mean().sort_values(ascending=False)
    )
    pct = df_clean["is_hotspot"].mean() * 100
    print(
        "Answer: A hotspot is defined as a grid cell with ≥3 attacks in the subsequent 90 days."
    )
    print(
        f"Overall, {pct:.1f}% of events are hotspots. Share by region is in the figure."
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    hotspot_share.plot(
        kind="bar", ax=ax1, color=["lightcoral", "steelblue"], edgecolor="black"
    )
    ax1.set_title(
        "Hotspot vs Non-Hotspot Events (≥3 attacks in next 90 days in same grid)",
        fontweight="bold",
    )
    ax1.set_xlabel("Is Hotspot (0=No, 1=Yes)")
    ax1.set_ylabel("Count")
    ax1.set_xticklabels(["Non-hotspot", "Hotspot"], rotation=0)
    sns.barplot(
        y=hotspot_by_region.index,
        x=hotspot_by_region.values * 100,
        palette="Oranges_r",
        ax=ax2,
    )
    ax2.set_title("Share of Hotspot Events by Region (%)", fontweight="bold")
    ax2.set_xlabel("Hotspot %")
    plt.tight_layout()
    plt.savefig(figures_dir / "q4_hotspots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {figures_dir / 'q4_hotspots.png'}")


def q5_answer_and_plot(feature_importance: pd.DataFrame, figures_dir: Path) -> None:
    """Q5: What factors best predict future hotspots?"""
    print("\n" + "=" * 60)
    print("Q5: What factors best predict future hotspots?")
    print("=" * 60)
    top_n = min(15, len(feature_importance))
    fi = feature_importance.head(top_n)
    print(
        "Answer: Recent attack history (attacks in last 30–365 days) and casualty history"
    )
    print("are the strongest predictors. Top 3 features:", list(fi["Feature"].head(3)))

    plt.figure(figsize=(10, 6))
    sns.barplot(y=fi["Feature"], x=fi["Importance"], palette="viridis")
    plt.title("Top Predictive Features for Hotspot Formation (Q5)", fontweight="bold")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(figures_dir / "q5_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {figures_dir / 'q5_feature_importance.png'}")


def q6_answer_and_plot(
    y_test: pd.Series,
    final_test_pred: np.ndarray,
    final_test_proba: np.ndarray,
    figures_dir: Path,
) -> None:
    """Q6: How well can we predict hotspots on unseen time periods?"""
    print("\n" + "=" * 60)
    print("Q6: How well can we predict hotspots on unseen time periods?")
    print("=" * 60)
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        ConfusionMatrixDisplay,
    )

    precision = precision_score(y_test, final_test_pred, zero_division=0)
    recall = recall_score(y_test, final_test_pred, zero_division=0)
    f1 = f1_score(y_test, final_test_pred, zero_division=0)
    roc_auc = (
        roc_auc_score(y_test, final_test_proba) if len(np.unique(y_test)) >= 2 else 0.0
    )
    print("Answer: Test-set (2015–2017) performance:")
    print(
        f"  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}  ROC-AUC: {roc_auc:.4f}"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    cm = confusion_matrix(y_test, final_test_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Non-hotspot", "Hotspot"]).plot(
        ax=ax1, values_format="d"
    )
    ax1.set_title("Confusion Matrix (Test 2015–2017)", fontweight="bold")
    fpr, tpr, _ = roc_curve(y_test, final_test_proba)
    ax2.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax2.plot([0, 1], [0, 1], "k--", lw=1)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve (Test Set)", fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "q6_model_performance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {figures_dir / 'q6_model_performance.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce answers to the 6 analytical questions and save all visualizations."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("terrorism.csv"),
        help="Path to raw GTD CSV (used if gtd_cleaned.csv is missing)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output"),
        help="Output directory (contains gtd_cleaned.csv and visualization/)",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directory for figure files (default: <out-dir>/visualization)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = (
        args.figures_dir.resolve()
        if args.figures_dir is not None
        else out_dir / "visualization"
    )
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    print(f"Figures will be saved to {figures_dir}")

    # Load or build data
    df_clean = load_or_build_data(args.data, out_dir)
    if "date" in df_clean.columns and not pd.api.types.is_datetime64_any_dtype(
        df_clean["date"]
    ):
        df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")

    # Q1–Q4: from df_clean only
    q1_answer_and_plot(df_clean, figures_dir)
    q2_answer_and_plot(df_clean, figures_dir)
    q3_answer_and_plot(df_clean, figures_dir)
    q4_answer_and_plot(df_clean, figures_dir)

    # Q5–Q6: need model; train a minimal RF if we have features and target
    feature_columns = get_feature_columns(df_clean)
    if "is_hotspot" in df_clean.columns and len(feature_columns) >= 5:
        print(
            "\nTraining a small model for Q5 (feature importance) and Q6 (test performance) ..."
        )
        try:
            model_outputs = train_minimal_model_for_q5_q6(df_clean)
            q5_answer_and_plot(model_outputs["feature_importance"], figures_dir)
            q6_answer_and_plot(
                model_outputs["y_test"],
                model_outputs["final_test_pred"],
                model_outputs["final_test_proba"],
                figures_dir,
            )
        except Exception as e:
            print(f"Q5/Q6 skipped due to: {e}")
    else:
        print("\nSkipping Q5/Q6 (need is_hotspot and enough feature columns in data).")

    print("\n" + "=" * 60)
    print("Done. All answers printed above; figures saved in", figures_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
