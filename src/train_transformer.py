"""Transformer training for time-series forecasting."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Add,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model

from .config import MODELS_DIR, TARGET_SENSOR, TRANSFORMER_MODEL_PATH, TRANSFORMER_PREDICTIONS_PATH
from .evaluation import metric_row, regression_metrics
from .feature_engineering import build_supervised_sequence_data
from .train_lstm import _prediction_frame


def build_transformer_model(
    sequence_length: int,
    n_features: int,
    d_model: int = 32,
    num_heads: int = 2,
    dropout: float = 0.10,
) -> tf.keras.Model:
    """Create a small Transformer encoder for one-step forecasting."""

    inputs = Input(shape=(sequence_length, n_features))
    x = Dense(d_model)(inputs)

    attention = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=max(d_model // num_heads, 1),
        dropout=dropout,
    )(x, x)
    x = Add()([x, attention])
    x = LayerNormalization(epsilon=1e-6)(x)

    feed_forward = Dense(d_model * 2, activation="relu")(x)
    feed_forward = Dense(d_model)(feed_forward)
    x = Add()([x, feed_forward])
    x = LayerNormalization(epsilon=1e-6)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs, name="simple_time_series_transformer")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_transformer_model(
    df: pd.DataFrame,
    target_sensor: str = TARGET_SENSOR,
    sequence_length: int = 30,
    epochs: int = 1,
    batch_size: int = 512,
    model_path: Path = TRANSFORMER_MODEL_PATH,
    predictions_path: Path = TRANSFORMER_PREDICTIONS_PATH,
) -> dict[str, Any]:
    """Train Transformer, save predictions and compare it with the baseline."""

    data = build_supervised_sequence_data(
        df=df,
        target_sensor=target_sensor,
        sequence_length=sequence_length,
    )
    model = build_transformer_model(sequence_length, data.x_train.shape[-1])

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

    transformer_metrics = regression_metrics(data.y_test_raw, predictions)
    baseline_metrics = regression_metrics(data.y_test_raw, data.baseline_test)
    baseline_better = baseline_metrics["rmse"] <= transformer_metrics["rmse"]
    conclusion = (
        "La baseline naive est meilleure ou equivalente au Transformer sur le test."
        if baseline_better
        else "Le Transformer ameliore la baseline naive sur le test."
    )

    prediction_df = _prediction_frame(
        data.timestamps_test,
        "Transformer",
        target_sensor,
        data.y_test_raw,
        predictions,
    )

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(predictions_path, index=False)

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
        MODELS_DIR / "transformer_scalers.joblib",
    )

    result = {
        "predictions": prediction_df,
        "metrics_row": metric_row(
            model_name="Transformer",
            task="prediction",
            target_sensor=target_sensor,
            conclusion=conclusion,
            **transformer_metrics,
        ),
        "history": history.history,
    }

    tf.keras.backend.clear_session()
    gc.collect()
    return result
