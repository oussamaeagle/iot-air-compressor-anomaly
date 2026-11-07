"""LSTM training for Oil_temperature forecasting."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM

from .config import (
    BASELINE_PREDICTIONS_PATH,
    LSTM_MODEL_PATH,
    LSTM_PREDICTIONS_PATH,
    MODELS_DIR,
    TARGET_SENSOR,
)
from .evaluation import metric_row, regression_metrics
from .feature_engineering import build_supervised_sequence_data


def build_lstm_model(sequence_length: int, n_features: int) -> tf.keras.Model:
    """Create a compact LSTM model for one-step-ahead forecasting."""

    model = Sequential(
        [
            Input(shape=(sequence_length, n_features)),
            LSTM(16, return_sequences=False),
            Dropout(0.10),
            Dense(8, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _prediction_frame(
    timestamps: pd.Series,
    model_name: str,
    target_sensor: str,
    y_true: Any,
    y_pred: Any,
) -> pd.DataFrame:
    """Build the normalized prediction output dataframe."""

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "model_name": model_name,
            "target_sensor": target_sensor,
            "real_value": y_true,
            "predicted_value": y_pred,
        }
    )
    frame["prediction_error"] = frame["real_value"] - frame["predicted_value"]
    return frame


def train_lstm_model(
    df: pd.DataFrame,
    target_sensor: str = TARGET_SENSOR,
    sequence_length: int = 30,
    epochs: int = 1,
    batch_size: int = 512,
    model_path: Path = LSTM_MODEL_PATH,
    predictions_path: Path = LSTM_PREDICTIONS_PATH,
    baseline_predictions_path: Path = BASELINE_PREDICTIONS_PATH,
) -> dict[str, Any]:
    """Train LSTM, save predictions and compare it with a naive baseline."""

    data = build_supervised_sequence_data(
        df=df,
        target_sensor=target_sensor,
        sequence_length=sequence_length,
    )
    model = build_lstm_model(sequence_length, data.x_train.shape[-1])

    callbacks = []
    if epochs > 1:
        callbacks.append(EarlyStopping(patience=2, restore_best_weights=True))

    history = model.fit(
        data.x_train,
        data.y_train,
        validation_data=(data.x_val, data.y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
        shuffle=False,
    )

    pred_scaled = model.predict(data.x_test, batch_size=batch_size, verbose=0)
    predictions = data.target_scaler.inverse_transform(pred_scaled).ravel()

    lstm_metrics = regression_metrics(data.y_test_raw, predictions)
    baseline_metrics = regression_metrics(data.y_test_raw, data.baseline_test)

    baseline_better = baseline_metrics["rmse"] <= lstm_metrics["rmse"]
    lstm_conclusion = (
        "La baseline naive est meilleure ou equivalente au LSTM sur le test."
        if baseline_better
        else "Le LSTM ameliore la baseline naive sur le test."
    )
    baseline_conclusion = "Reference: prediction(t) = valeur reelle(t-1)."

    prediction_df = _prediction_frame(
        data.timestamps_test,
        "LSTM",
        target_sensor,
        data.y_test_raw,
        predictions,
    )
    baseline_df = _prediction_frame(
        data.timestamps_test,
        "NaiveBaseline",
        target_sensor,
        data.y_test_raw,
        data.baseline_test,
    )

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(predictions_path, index=False)
    baseline_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_df.to_csv(baseline_predictions_path, index=False)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path, include_optimizer=False)
    joblib.dump(
        {
            "feature_scaler": data.feature_scaler,
            "target_scaler": data.target_scaler,
            "feature_columns": data.feature_columns,
            "target_sensor": target_sensor,
            "sequence_length": sequence_length,
        },
        MODELS_DIR / "lstm_scalers.joblib",
    )

    result = {
        "predictions": prediction_df,
        "baseline_predictions": baseline_df,
        "metrics_row": metric_row(
            model_name="LSTM",
            task="prediction",
            target_sensor=target_sensor,
            conclusion=lstm_conclusion,
            **lstm_metrics,
        ),
        "baseline_metrics_row": metric_row(
            model_name="NaiveBaseline",
            task="prediction",
            target_sensor=target_sensor,
            conclusion=baseline_conclusion,
            **baseline_metrics,
        ),
        "history": history.history,
    }

    tf.keras.backend.clear_session()
    gc.collect()
    return result
