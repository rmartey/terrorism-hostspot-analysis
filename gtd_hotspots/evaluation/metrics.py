"""
Classification metrics for hotspot prediction (precision, recall, F1, ROC-AUC, etc.).
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def classification_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_proba: Optional[Union[np.ndarray, pd.Series]] = None,
    zero_division: Union[int, str] = 0,
) -> Dict[str, float]:
    """
    Compute precision, recall, F1, accuracy; optionally ROC-AUC and PR-AUC.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted labels (0/1).
    y_proba : array-like, optional
        Predicted probabilities for positive class (for ROC/PR-AUC).
    zero_division : int or 'warn'
        Passed to precision_score, recall_score, f1_score.

    Returns
    -------
    dict
        Metric name -> value.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    out = {
        "precision": precision_score(
            y_true, y_pred, zero_division=zero_division
        ),
        "recall": recall_score(y_true, y_pred, zero_division=zero_division),
        "f1": f1_score(y_true, y_pred, zero_division=zero_division),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    if y_proba is not None:
        y_proba = np.asarray(y_proba).ravel()
        if len(np.unique(y_true)) >= 2:
            out["roc_auc"] = roc_auc_score(y_true, y_proba)
            out["pr_auc"] = average_precision_score(y_true, y_proba)
    return out


def print_classification_report(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    labels: Optional[list] = None,
    zero_division: Union[int, str] = 0,
) -> None:
    """Print sklearn classification_report and confusion matrix."""
    print(classification_report(y_true, y_pred, labels=labels, zero_division=zero_division))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
