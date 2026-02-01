# GTD Terrorism Hotspot Prediction

Predictive analytics for identifying future terrorism hotspots using the Global Terrorism Database (GTD), 1970–2017.

## Project structure

- **`GTD Analysis.ipynb`** – Full analysis, EDA, modeling, and evaluation (run this for the complete workflow).
- **`gtd_hotspots/`** – Refactored Python package for reproducibility and reuse.
- **`run_pipeline.py`** – Command-line script: load → clean → featurize → target → split (no training).
- **`visualizations.py`** – Run from terminal to answer the 6 analytical questions and save all visualizations to `output/visualization/`.

## Refactored package (`gtd_hotspots`)

The codebase is modular and reproducible:

| Module | Purpose |
|--------|--------|
| `gtd_hotspots.config` | Paths, constants, `FUTURE_WINDOW_DAYS`, `FEATURE_COLUMNS`, `CORE_FEATURES` |
| `gtd_hotspots.data` | `load_gtd_data()`, `clean_gtd_data()` (date, geo, casualty, categorical, binary) |
| `gtd_hotspots.features` | `build_all_features()`, `create_hotspot_labels()`, `prepare_X_y()` |
| `gtd_hotspots.modeling` | `get_train_val_test_masks()`, `temporal_split()`, `prepare_splits()`, `split_percentages()` |
| `gtd_hotspots.evaluation` | `classification_metrics()`, `print_classification_report()` |
| `gtd_hotspots.utils` | `set_display_options()`, `suppress_warnings()` |

### Use from Python or notebook

```python
from gtd_hotspots.data import load_gtd_data, clean_gtd_data
from gtd_hotspots.features import build_all_features, create_hotspot_labels, prepare_X_y
from gtd_hotspots.modeling import prepare_splits, split_percentages
from gtd_hotspots.config import FUTURE_WINDOW_DAYS, HOTSPOT_THRESHOLD

df = load_gtd_data("terrorism.csv")
df_clean = clean_gtd_data(df, verbose=True)
df_clean = build_all_features(df_clean, verbose=True)
df_clean = create_hotspot_labels(df_clean, future_window=FUTURE_WINDOW_DAYS, threshold=HOTSPOT_THRESHOLD)

X_train, X_val, X_test, y_train, y_val, y_test, train_mask, val_mask, test_mask, X, y = prepare_splits(
    df_clean, future_window_days=FUTURE_WINDOW_DAYS
)
```

### Run pipeline from command line

From the project directory:

```bash
pip install -r requirements.txt
python run_pipeline.py --data terrorism.csv --out-dir . --verbose
```

This writes `gtd_cleaned.csv`. Use **GTD Analysis.ipynb** for model training, threshold tuning, and evaluation.

### Run questions and visualizations from terminal

From the project directory (where `gtd_cleaned.csv` or `terrorism.csv` lives):

```bash
python visualizations.py [--data terrorism.csv] [--out-dir output] [--figures-dir output/visualization]
```

This prints answers to the 6 analytical questions and saves all graphs as PNGs in `output/visualization/` by default. If `gtd_cleaned.csv` is missing in `--out-dir`, the script builds it from `--data` and trains a small model for Q5/Q6.

## Data

- **terrorism.csv** – Raw GTD data (place in project directory or pass `--data path/to/terrorism.csv`).
- **gtd_cleaned.csv** – Output of `clean_gtd_data()` + feature engineering (created by notebook or `run_pipeline.py`).

## Requirements

See `requirements.txt`: pandas, numpy, scikit-learn, scipy, imbalanced-learn, joblib; optional matplotlib, seaborn, plotly, folium, jupyter, nbformat.
