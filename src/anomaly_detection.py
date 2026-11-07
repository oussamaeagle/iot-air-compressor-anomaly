"""Alert generation from anomaly detection outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ALERTS_PATH

ALERT_COLUMNS = [
    "timestamp",
    "sensor_name",
    "value",
    "anomaly_score",
    "threshold",
    "severity",
    "message",
]


def generate_alerts(
    anomalies: pd.DataFrame,
    output_path: Path = ALERTS_PATH,
) -> pd.DataFrame:
    """Create dashboard-ready alerts for detected anomalies."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if anomalies.empty:
        alerts = pd.DataFrame(columns=ALERT_COLUMNS)
        alerts.to_csv(output_path, index=False)
        return alerts

    detected = anomalies[anomalies["is_anomaly"].astype(bool)].copy()
    if detected.empty:
        alerts = pd.DataFrame(columns=ALERT_COLUMNS)
        alerts.to_csv(output_path, index=False)
        return alerts

    if "sensor_name" not in detected.columns:
        detected["sensor_name"] = "multivariate"
    if "value" not in detected.columns:
        detected["value"] = pd.NA

    detected["message"] = (
        "Anomalie detectee sur le compresseur: temperature d'huile ou pression anormale."
    )
    alerts = detected[ALERT_COLUMNS].sort_values("timestamp").reset_index(drop=True)
    alerts.to_csv(output_path, index=False)
    return alerts
