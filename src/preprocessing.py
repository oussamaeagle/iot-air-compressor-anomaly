"""Data loading, cleaning, scaling and sequence utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import (
    MISSING_REPORT_PATH,
    PROCESSED_DATA_PATH,
    SCALED_DATA_PATH,
    SENSOR_COLUMNS,
    TIMESTAMP_COLUMN,
    ensure_directories,
    resolve_raw_data_path,
)


@dataclass
class PreprocessingResult:
    """Container returned by the preprocessing pipeline."""

    clean_data: pd.DataFrame
    scaled_data: pd.DataFrame
    scaler: StandardScaler
    missing_summary: pd.DataFrame
    numeric_columns: list[str]


def _dtype_map() -> dict[str, str]:
    """Return memory-friendly dtypes for known numeric sensors."""

    return {column: "float32" for column in SENSOR_COLUMNS}


def load_raw_data(
    raw_data_path: str | Path | None = None,
    sample_size: int | None = None,
) -> pd.DataFrame:
    """Load the MetroPT3 CSV with optional row sampling.

    sample_size reads the first N chronological rows from the CSV. This is
    deterministic and appropriate for fast experiments on a time series.
    """

    path = resolve_raw_data_path(raw_data_path)
    read_kwargs: dict[str, object] = {
        "dtype": _dtype_map(),
        "low_memory": False,
    }
    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size must be a positive integer.")
        read_kwargs["nrows"] = sample_size

    return pd.read_csv(path, **read_kwargs)


def clean_data(
    df: pd.DataFrame,
    filter_comp_active: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean raw MetroPT3 records and return data plus missing-value report."""

    if df.empty:
        raise ValueError("The input dataframe is empty.")

    df = df.copy()
    unnamed_columns = [column for column in df.columns if column.startswith("Unnamed")]
    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    if TIMESTAMP_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {TIMESTAMP_COLUMN}")

    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP_COLUMN])
    df = df.sort_values(TIMESTAMP_COLUMN).drop_duplicates().reset_index(drop=True)

    numeric_columns = [column for column in SENSOR_COLUMNS if column in df.columns]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float32")

    if filter_comp_active:
        if "COMP" not in df.columns:
            raise ValueError("filter_comp_active=True requires the COMP column.")
        df = df[df["COMP"] == 1].reset_index(drop=True)

    missing_summary = (
        df.isna()
        .sum()
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missing_summary["missing_percentage"] = (
        missing_summary["missing_count"] / max(len(df), 1) * 100
    )

    if numeric_columns:
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
        df[numeric_columns] = df[numeric_columns].interpolate(
            method="linear", limit_direction="both"
        )
        df[numeric_columns] = df[numeric_columns].ffill().bfill()
        for column in numeric_columns:
            if df[column].isna().any():
                df[column] = df[column].fillna(df[column].median())
            df[column] = df[column].astype("float32")

    return df.reset_index(drop=True), missing_summary


def scale_numeric_data(
    df: pd.DataFrame,
    numeric_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Normalize numeric sensor columns with StandardScaler."""

    numeric_columns = numeric_columns or [column for column in SENSOR_COLUMNS if column in df]
    scaled = df.copy()
    scaler = StandardScaler()
    if numeric_columns:
        scaled_values = scaler.fit_transform(scaled[numeric_columns]).astype("float32")
        scaled[numeric_columns] = scaled_values
    return scaled, scaler


def create_time_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int = 30,
    target_column: str | None = None,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray | None, pd.Series]:
    """Create chronological sliding-window sequences.

    Returns X with shape (samples, sequence_length, features), optional y, and
    timestamps aligned with the predicted/reconstructed point.
    """

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if target_column is not None and horizon <= 0:
        raise ValueError("horizon must be positive.")
    effective_horizon = horizon if target_column is not None else 0

    missing_columns = [column for column in feature_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing feature columns: {missing_columns}")

    n_samples = len(df) - sequence_length - effective_horizon + 1
    if n_samples <= 0:
        raise ValueError(
            "Not enough rows to create sequences. Increase sample_size or reduce sequence_length."
        )

    values = df[feature_columns].to_numpy(dtype=np.float32, copy=True)
    windows = np.lib.stride_tricks.sliding_window_view(
        values, window_shape=sequence_length, axis=0
    )
    x = np.moveaxis(windows[:n_samples], -1, 1)
    x = np.ascontiguousarray(x, dtype=np.float32)

    target_index_start = sequence_length + effective_horizon - 1
    timestamps = df[TIMESTAMP_COLUMN].iloc[target_index_start:].reset_index(drop=True)

    if target_column is None:
        return x, None, timestamps

    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")
    y = (
        df[target_column]
        .iloc[target_index_start:]
        .to_numpy(dtype=np.float32, copy=True)
    )
    return x, y, timestamps


def preprocess_data(
    raw_data_path: str | Path | None = None,
    sample_size: int | None = None,
    filter_comp_active: bool = False,
    output_path: Path = PROCESSED_DATA_PATH,
    scaled_output_path: Path = SCALED_DATA_PATH,
    missing_report_path: Path = MISSING_REPORT_PATH,
) -> PreprocessingResult:
    """Run the complete preprocessing pipeline and save cleaned CSV files."""

    ensure_directories()
    raw = load_raw_data(raw_data_path=raw_data_path, sample_size=sample_size)
    clean, missing_summary = clean_data(raw, filter_comp_active=filter_comp_active)
    numeric_columns = [column for column in SENSOR_COLUMNS if column in clean.columns]
    scaled, scaler = scale_numeric_data(clean, numeric_columns)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False)
    scaled.to_csv(scaled_output_path, index=False)
    missing_summary.to_csv(missing_report_path, index=False)

    return PreprocessingResult(
        clean_data=clean,
        scaled_data=scaled,
        scaler=scaler,
        missing_summary=missing_summary,
        numeric_columns=numeric_columns,
    )





