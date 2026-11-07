"""Evaluation metrics and model comparison utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from .config import MODEL_COMPARISON_PATH


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE and R2 for prediction models."""

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Compute classification metrics for anomaly labels when available."""

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def metric_row(
    model_name: str,
    task: str,
    target_sensor: str,
    conclusion: str,
    rmse: float | None = None,
    mae: float | None = None,
    r2: float | None = None,
    precision: float | None = None,
    recall: float | None = None,
    f1_score_value: float | None = None,
) -> dict[str, Any]:
    """Create a normalized row for model_comparison.csv and SQL metrics."""

    return {
        "model_name": model_name,
        "task": task,
        "target_sensor": target_sensor,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score_value,
        "conclusion": conclusion,
    }


def save_model_comparison(
    rows: list[dict[str, Any]],
    output_path: Path = MODEL_COMPARISON_PATH,
) -> pd.DataFrame:
    """Save model comparison rows to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(rows)
    required_columns = [
        "model_name",
        "task",
        "target_sensor",
        "rmse",
        "mae",
        "r2",
        "precision",
        "recall",
        "f1_score",
        "conclusion",
    ]
    for column in required_columns:
        if column not in comparison.columns:
            comparison[column] = None
    comparison = comparison[required_columns]
    comparison.to_csv(output_path, index=False)
    return comparison

