from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def regression_metrics(y_true, y_pred, model_name):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"model": model_name, "rmse": rmse, "mae": mae, "r2": r2}


def anomaly_metrics(y_true=None, y_pred=None, reconstruction_error=None, threshold=None):
    if y_true is not None and y_pred is not None:
        cm = confusion_matrix(y_true, y_pred)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": cm.tolist(),
        }
    if reconstruction_error is None:
        return {}
    anomaly_count = int(np.sum(np.asarray(reconstruction_error) > threshold))
    total = int(len(reconstruction_error))
    return {
        "threshold": float(threshold),
        "anomaly_count": anomaly_count,
        "anomaly_percentage": float((anomaly_count / total) * 100) if total else 0.0,
    }


def save_metrics_report(prediction_metrics, anomaly_report, output_path):
    rows = list(prediction_metrics)
    if anomaly_report:
        rows.append({"model": "Autoencoder", **anomaly_report})
    report = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    return report

