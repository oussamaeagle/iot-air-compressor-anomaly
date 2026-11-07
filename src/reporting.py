"""EDA and Power BI reporting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    EDA_TIME_SERIES_COLUMNS,
    FIGURES_DIR,
    KPI_SUMMARY_PATH,
    POWERBI_DATASET_PATH,
    RESULTS_DIR,
    SENSOR_COLUMNS,
    TIMESTAMP_COLUMN,
)


def _downsample_for_plot(df: pd.DataFrame, max_points: int = 5000) -> pd.DataFrame:
    """Return a deterministic downsample for readable plots."""

    if len(df) <= max_points:
        return df
    step = max(len(df) // max_points, 1)
    return df.iloc[::step].copy()


def run_eda(
    df: pd.DataFrame,
    output_dir: Path = RESULTS_DIR,
    figures_dir: Path = FIGURES_DIR,
) -> dict[str, Path]:
    """Generate EDA CSV summaries and PNG figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])

    summary_rows = [
        {"metric": "n_rows", "value": len(df)},
        {"metric": "n_columns", "value": df.shape[1]},
        {"metric": "start_date", "value": df[TIMESTAMP_COLUMN].min()},
        {"metric": "end_date", "value": df[TIMESTAMP_COLUMN].max()},
        {"metric": "duration_days", "value": (df[TIMESTAMP_COLUMN].max() - df[TIMESTAMP_COLUMN].min()).days},
        {"metric": "duplicated_rows", "value": int(df.duplicated().sum())},
    ]
    eda_summary = pd.DataFrame(summary_rows)
    eda_summary_path = output_dir / "eda_summary.csv"
    eda_summary.to_csv(eda_summary_path, index=False)

    dtypes_path = output_dir / "variable_types.csv"
    pd.DataFrame({"column": df.columns, "dtype": [str(dtype) for dtype in df.dtypes]}).to_csv(
        dtypes_path, index=False
    )

    descriptive_path = output_dir / "descriptive_statistics.csv"
    df.describe(include="all").transpose().to_csv(descriptive_path)

    missing_path = output_dir / "eda_missing_values.csv"
    missing = df.isna().sum().rename("missing_count").reset_index().rename(columns={"index": "column"})
    missing["missing_percentage"] = missing["missing_count"] / max(len(df), 1) * 100
    missing.to_csv(missing_path, index=False)

    comp_distribution_path = output_dir / "comp_distribution.csv"
    if "COMP" in df.columns:
        comp_distribution = df["COMP"].value_counts(dropna=False).rename_axis("COMP").reset_index(name="count")
        comp_distribution["percentage"] = comp_distribution["count"] / max(len(df), 1) * 100
    else:
        comp_distribution = pd.DataFrame(columns=["COMP", "count", "percentage"])
    comp_distribution.to_csv(comp_distribution_path, index=False)

    numeric_columns = [column for column in SENSOR_COLUMNS if column in df.columns]
    correlation_path = output_dir / "correlation_matrix.csv"
    correlation = df[numeric_columns].corr(numeric_only=True)
    correlation.to_csv(correlation_path)

    plot_df = _downsample_for_plot(df[[TIMESTAMP_COLUMN, *numeric_columns]])
    for column in EDA_TIME_SERIES_COLUMNS:
        if column not in plot_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(plot_df[TIMESTAMP_COLUMN], plot_df[column], linewidth=0.8)
        ax.set_title(f"Evolution temporelle - {column}")
        ax.set_xlabel("timestamp")
        ax.set_ylabel(column)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(figures_dir / f"timeseries_{column}.png", dpi=140)
        plt.close(fig)

    if not correlation.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(correlation.columns)))
        ax.set_yticks(range(len(correlation.index)))
        ax.set_xticklabels(correlation.columns, rotation=90)
        ax.set_yticklabels(correlation.index)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Matrice de correlation")
        fig.tight_layout()
        fig.savefig(figures_dir / "correlation_matrix.png", dpi=140)
        plt.close(fig)

    statistical_anomalies_path = output_dir / "statistical_anomalies.csv"
    if numeric_columns:
        zscores = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std(ddof=0).replace(0, np.nan)
        abs_zscores = zscores.abs().fillna(0)
        max_scores = abs_zscores.max(axis=1)
        max_sensors = abs_zscores.idxmax(axis=1)
        mask = max_scores > 3
        statistical_anomalies = df.loc[mask, [TIMESTAMP_COLUMN]].copy()
        statistical_anomalies["sensor_name"] = max_sensors.loc[mask].to_numpy()
        statistical_anomalies["z_score"] = max_scores.loc[mask].to_numpy()
        statistical_anomalies["value"] = [
            df.at[index, sensor]
            for index, sensor in zip(statistical_anomalies.index, statistical_anomalies["sensor_name"])
        ]
    else:
        statistical_anomalies = pd.DataFrame(columns=[TIMESTAMP_COLUMN, "sensor_name", "z_score", "value"])
    statistical_anomalies.to_csv(statistical_anomalies_path, index=False)

    return {
        "eda_summary": eda_summary_path,
        "variable_types": dtypes_path,
        "descriptive_statistics": descriptive_path,
        "missing_values": missing_path,
        "comp_distribution": comp_distribution_path,
        "correlation_matrix": correlation_path,
        "statistical_anomalies": statistical_anomalies_path,
    }


def _prediction_tag(model_name: str) -> str:
    """Return a stable lowercase tag for prediction columns."""

    mapping = {
        "NaiveBaseline": "baseline",
        "LSTM": "lstm",
        "Transformer": "transformer",
    }
    return mapping.get(model_name, model_name.lower().replace(" ", "_"))


def export_powerbi_dataset(
    clean_df: pd.DataFrame,
    prediction_frames: Iterable[pd.DataFrame],
    anomalies: pd.DataFrame,
    output_path: Path = POWERBI_DATASET_PATH,
) -> pd.DataFrame:
    """Create the consolidated CSV used by Power BI."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = clean_df.copy()
    dataset[TIMESTAMP_COLUMN] = pd.to_datetime(dataset[TIMESTAMP_COLUMN])

    for predictions in prediction_frames:
        if predictions is None or predictions.empty:
            continue
        predictions = predictions.copy()
        predictions[TIMESTAMP_COLUMN] = pd.to_datetime(predictions[TIMESTAMP_COLUMN])
        model_name = str(predictions["model_name"].iloc[0])
        tag = _prediction_tag(model_name)
        columns = predictions[
            [TIMESTAMP_COLUMN, "predicted_value", "prediction_error"]
        ].rename(
            columns={
                "predicted_value": f"{tag}_prediction",
                "prediction_error": f"{tag}_error",
            }
        )
        dataset = dataset.merge(columns, on=TIMESTAMP_COLUMN, how="left")

    anomaly_columns = [
        TIMESTAMP_COLUMN,
        "anomaly_score",
        "threshold",
        "is_anomaly",
        "severity",
        "sensor_name",
    ]
    if anomalies is not None and not anomalies.empty:
        anomaly_data = anomalies.copy()
        anomaly_data[TIMESTAMP_COLUMN] = pd.to_datetime(anomaly_data[TIMESTAMP_COLUMN])
        available = [column for column in anomaly_columns if column in anomaly_data.columns]
        dataset = dataset.merge(anomaly_data[available], on=TIMESTAMP_COLUMN, how="left")
    else:
        for column in anomaly_columns[1:]:
            dataset[column] = pd.NA

    if "is_anomaly" in dataset.columns:
        dataset["is_anomaly"] = dataset["is_anomaly"].where(dataset["is_anomaly"].notna(), False).astype(bool)

    dataset.to_csv(output_path, index=False)
    return dataset


def save_kpi_summary(
    clean_df: pd.DataFrame,
    anomalies: pd.DataFrame,
    model_comparison: pd.DataFrame,
    output_path: Path = KPI_SUMMARY_PATH,
) -> pd.DataFrame:
    """Save high-level dashboard KPIs."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_measurements = int(len(clean_df))

    detected = pd.DataFrame()
    if anomalies is not None and not anomalies.empty:
        detected = anomalies[anomalies["is_anomaly"].astype(bool)].copy()
    total_anomalies = int(len(detected))
    anomaly_percentage = float(total_anomalies / max(total_measurements, 1) * 100)

    prediction_metrics = model_comparison[
        (model_comparison["task"] == "prediction") & model_comparison["rmse"].notna()
    ].copy()
    if prediction_metrics.empty:
        best_prediction_model = None
        best_rmse = None
    else:
        best_row = prediction_metrics.sort_values("rmse").iloc[0]
        best_prediction_model = best_row["model_name"]
        best_rmse = best_row["rmse"]

    last_anomaly_date = None
    most_problematic_sensor = None
    if not detected.empty:
        last_anomaly_date = pd.to_datetime(detected["timestamp"]).max()
        if "sensor_name" in detected.columns:
            most_problematic_sensor = detected["sensor_name"].value_counts().idxmax()

    kpis = pd.DataFrame(
        [
            {
                "total_measurements": total_measurements,
                "total_anomalies": total_anomalies,
                "anomaly_percentage": anomaly_percentage,
                "best_prediction_model": best_prediction_model,
                "best_rmse": best_rmse,
                "last_anomaly_date": last_anomaly_date,
                "most_problematic_sensor": most_problematic_sensor,
            }
        ]
    )
    kpis.to_csv(output_path, index=False)
    return kpis


