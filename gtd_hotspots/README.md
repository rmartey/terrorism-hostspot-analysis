# gtd_hotspots

Modular Python package for **terrorism hotspot prediction** using the Global Terrorism Database (GTD). Designed for reproducibility and reuse.

## Structure

```
gtd_hotspots/
├── __init__.py       # Package API
├── config.py         # Paths, constants, feature lists
├── utils.py          # Display, warnings
├── data/
│   ├── load.py       # load_gtd_data()
│   └── clean.py      # clean_gtd_data(), clean_dates(), clean_geospatial(), ...
├── features/
│   ├── engineering.py # build_all_features(), temporal/geo/history/attack/perpetrator
│   ├── target.py     # create_hotspot_labels()
│   └── selection.py   # get_feature_columns(), prepare_X_y()
├── modeling/
│   ├── split.py      # get_train_val_test_masks(), temporal_split(), split_percentages()
│   └── prepare.py     # prepare_splits()
└── evaluation/
    └── metrics.py    # classification_metrics(), print_classification_report()
```

## Usage

### From script or notebook

```python
from gtd_hotspots.data import load_gtd_data, clean_gtd_data
from gtd_hotspots.features import build_all_features, create_hotspot_labels, prepare_X_y
from gtd_hotspots.modeling import prepare_splits, split_percentages
from gtd_hotspots.config import FUTURE_WINDOW_DAYS, HOTSPOT_THRESHOLD

# Load and clean
df = load_gtd_data("terrorism.csv")
df_clean = clean_gtd_data(df, verbose=True)
df_clean = build_all_features(df_clean, verbose=True)
df_clean = create_hotspot_labels(df_clean, future_window=FUTURE_WINDOW_DAYS, threshold=HOTSPOT_THRESHOLD)

# Split
splits = prepare_splits(df_clean, future_window_days=FUTURE_WINDOW_DAYS)
X_train, X_val, X_test, y_train, y_val, y_test, train_mask, val_mask, test_mask, X, y = splits
```

### Command-line pipeline

From project root:

```bash
python run_pipeline.py --data terrorism.csv --out-dir . --verbose
```

Produces `gtd_cleaned.csv`; use **GTD Analysis.ipynb** for model training, threshold tuning, and evaluation.

## Conventions

- **Temporal split**: Train ≤ 2012, Validation 2013–2014, Test 2015–2017, with label-availability window (90 days).
- **Hotspot**: ≥ 3 attacks in the same grid cell in the subsequent 90 days.
- **Features**: Defined in `config.FEATURE_COLUMNS`; selection uses only columns present in the dataframe.

## Requirements

See project root `requirements.txt`: pandas, numpy, scikit-learn, scipy, imbalanced-learn, joblib; optional matplotlib, seaborn, plotly, folium, jupyter, nbformat.
