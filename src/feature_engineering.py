"""Feature engineering helpers for time-series models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import (
    AUTOENCODER_FEATURE_COLUMNS,
    PREDICTION_FEATURE_COLUMNS,
    SENSOR_COLUMNS,
    TIMESTAMP_COLUMN,
)
from .preprocessing import create_time_sequences


@dataclass
class SupervisedSequenceData:
    """Chronological train/validation/test arrays for prediction models."""

    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    y_train_raw: np.ndarray
    y_val_raw: np.ndarray
    y_test_raw: np.ndarray
    baseline_test: np.ndarray
    timestamps_test: pd.Series
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    feature_columns: list[str]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features useful for exploratory analysis."""

    if TIMESTAMP_COLUMN not in df.columns:
        raise ValueError(f"Missing timestamp column: {TIMESTAMP_COLUMN}")

    enriched = df.copy()
    timestamp = pd.to_datetime(enriched[TIMESTAMP_COLUMN], errors="coerce")
    enriched["hour"] = timestamp.dt.hour.astype("int16")
    enriched["dayofweek"] = timestamp.dt.dayofweek.astype("int16")
    enriched["month"] = timestamp.dt.month.astype("int16")
    return enriched


def prediction_feature_columns(target_sensor: str) -> list[str]:
    """Return prediction features, excluding the selected target if needed."""

    if target_sensor == "Oil_temperature":
        return [column for column in PREDICTION_FEATURE_COLUMNS if column != target_sensor]
    return [column for column in SENSOR_COLUMNS if column != target_sensor]


def autoencoder_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return available columns for the multivariate autoencoder."""

    return [column for column in AUTOENCODER_FEATURE_COLUMNS if column in df.columns]


def chronological_indices(
    n_samples: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[slice, slice, slice]:
    """Build chronological train, validation and test slices."""

    if n_samples < 10:
        raise ValueError("At least 10 sequences are required for a chronological split.")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be lower than 1.")

    train_end = max(int(n_samples * train_ratio), 1)
    val_end = max(int(n_samples * (train_ratio + val_ratio)), train_end + 1)
    if val_end >= n_samples:
        val_end = n_samples - 1
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n_samples)


def _scale_3d(
    array: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    """Scale a 3D sequence array without changing its shape."""

    original_shape = array.shape
    flat = array.reshape(-1, original_shape[-1])
    scaled = scaler.transform(flat).reshape(original_shape)
    return scaled.astype("float32")


def build_supervised_sequence_data(
    df: pd.DataFrame,
    target_sensor: str,
    sequence_length: int = 30,
    horizon: int = 1,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> SupervisedSequenceData:
    """Create scaled chronological arrays for LSTM and Transformer models."""

    if target_sensor not in df.columns:
        raise ValueError(f"Target sensor not found: {target_sensor}")

    feature_columns = prediction_feature_columns(target_sensor)
    missing_columns = [column for column in feature_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing feature columns for prediction: {missing_columns}")

    x_raw, y_raw, timestamps = create_time_sequences(
        df=df,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        target_column=target_sensor,
        horizon=horizon,
    )
    if y_raw is None:
        raise ValueError("Target sequence generation failed.")

    target_indices = np.arange(sequence_length + horizon - 1, len(df))
    baseline_indices = target_indices - horizon
    baseline = df[target_sensor].iloc[baseline_indices].to_numpy(dtype=np.float32)

    train_slice, val_slice, test_slice = chronological_indices(
        len(x_raw), train_ratio=train_ratio, val_ratio=val_ratio
    )

    feature_scaler = StandardScaler()
    feature_scaler.fit(x_raw[train_slice].reshape(-1, len(feature_columns)))

    target_scaler = StandardScaler()
    target_scaler.fit(y_raw[train_slice].reshape(-1, 1))

    x_train = _scale_3d(x_raw[train_slice], feature_scaler)
    x_val = _scale_3d(x_raw[val_slice], feature_scaler)
    x_test = _scale_3d(x_raw[test_slice], feature_scaler)

    y_train = target_scaler.transform(y_raw[train_slice].reshape(-1, 1)).astype("float32")
    y_val = target_scaler.transform(y_raw[val_slice].reshape(-1, 1)).astype("float32")
    y_test = target_scaler.transform(y_raw[test_slice].reshape(-1, 1)).astype("float32")

    return SupervisedSequenceData(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        y_train_raw=y_raw[train_slice],
        y_val_raw=y_raw[val_slice],
        y_test_raw=y_raw[test_slice],
        baseline_test=baseline[test_slice],
        timestamps_test=timestamps.iloc[test_slice].reset_index(drop=True),
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_columns=feature_columns,
    )

