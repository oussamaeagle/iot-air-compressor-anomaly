"""Prediction helpers for trained Keras models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .config import LSTM_MODEL_PATH, TARGET_SENSOR
from .feature_engineering import build_supervised_sequence_data
from .train_lstm import _prediction_frame


def load_model_and_metadata(
    model_path: Path = LSTM_MODEL_PATH,
    metadata_path: Path | None = None,
) -> tuple[tf.keras.Model, dict[str, Any] | None]:
    """Load a saved Keras model and optional joblib metadata."""

    model = tf.keras.models.load_model(model_path, compile=False)
    metadata = None
    if metadata_path is not None and metadata_path.exists():
        metadata = joblib.load(metadata_path)
    return model, metadata


def predict_test_split(
    df: pd.DataFrame,
    model: tf.keras.Model,
    target_sensor: str = TARGET_SENSOR,
    sequence_length: int = 30,
    batch_size: int = 512,
    model_name: str = "LoadedModel",
) -> pd.DataFrame:
    """Predict the chronological test split of a dataframe with a trained model."""

    data = build_supervised_sequence_data(
        df=df,
        target_sensor=target_sensor,
        sequence_length=sequence_length,
    )
    pred_scaled = model.predict(data.x_test, batch_size=batch_size, verbose=0)
    predictions = data.target_scaler.inverse_transform(np.asarray(pred_scaled).reshape(-1, 1)).ravel()
    return _prediction_frame(
        data.timestamps_test,
        model_name,
        target_sensor,
        data.y_test_raw,
        predictions,
    )
