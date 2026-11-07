"""Unsupervised autoencoder anomaly detection."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Reshape
from tensorflow.keras.models import Model

from .config import AUTOENCODER_ANOMALIES_PATH, AUTOENCODER_MODEL_PATH, MODELS_DIR, TARGET_SENSOR
from .evaluation import classification_metrics, metric_row
from .feature_engineering import autoencoder_feature_columns, chronological_indices
from .preprocessing import create_time_sequences


def build_autoencoder_model(sequence_length: int, n_features: int) -> tf.keras.Model:
    """Create a dense sequence autoencoder for multivariate reconstruction."""

    inputs = Input(shape=(sequence_length, n_features))
    x = Flatten()(inputs)
    x = Dense(96, activation="relu")(x)
    x = Dropout(0.10)(x)
    x = Dense(32, activation="relu", name="bottleneck")(x)
    x = Dense(96, activation="relu")(x)
    x = Dense(sequence_length * n_features)(x)
    outputs = Reshape((sequence_length, n_features))(x)

    model = Model(inputs=inputs, outputs=outputs, name="sequence_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def _severity(score: float, threshold: float) -> str:
    """Map an anomaly score to a severity level."""

    if score <= threshold:
        return "normal"
    if threshold <= 0:
        return "elevee"
    ratio = score / threshold
    if ratio < 1.20:
        return "faible"
    if ratio < 1.80:
        return "moyenne"
    return "elevee"


def _compute_reconstruction_errors(
    model: tf.keras.Model,
    sequences: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute sequence scores and dominant feature indices by batch."""

    errors = np.empty(len(sequences), dtype=np.float32)
    dominant_indices = np.empty(len(sequences), dtype=np.int32)

    for start in range(0, len(sequences), batch_size):
        end = min(start + batch_size, len(sequences))
        batch = sequences[start:end]
        reconstructed = model.predict(batch, batch_size=batch_size, verbose=0)
        squared_error = np.square(batch - reconstructed)
        errors[start:end] = squared_error.mean(axis=(1, 2))
        dominant_indices[start:end] = squared_error.mean(axis=1).argmax(axis=1)

    return errors, dominant_indices


def train_autoencoder_model(
    df: pd.DataFrame,
    sequence_length: int = 30,
    epochs: int = 1,
    batch_size: int = 512,
    threshold_method: str = "percentile95",
    model_path: Path = AUTOENCODER_MODEL_PATH,
    anomalies_path: Path = AUTOENCODER_ANOMALIES_PATH,
) -> dict[str, Any]:
    """Train an unsupervised autoencoder and save anomaly scores."""

    feature_columns = autoencoder_feature_columns(df)
    if not feature_columns:
        raise ValueError("No sensor columns available for autoencoder training.")

    train_row_end = max(int(len(df) * 0.70), sequence_length + 1)
    scaler = StandardScaler()
    scaler.fit(df[feature_columns].iloc[:train_row_end])

    scaled_df = df.copy()
    scaled_df[feature_columns] = scaler.transform(df[feature_columns]).astype("float32")
    sequences, _, timestamps = create_time_sequences(
        scaled_df,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        target_column=None,
    )
    train_slice, val_slice, _ = chronological_indices(len(sequences), train_ratio=0.70, val_ratio=0.15)

    model = build_autoencoder_model(sequence_length, len(feature_columns))
    callbacks = []
    if epochs > 1:
        callbacks.append(EarlyStopping(patience=2, restore_best_weights=True))

    history = model.fit(
        sequences[train_slice],
        sequences[train_slice],
        validation_data=(sequences[val_slice], sequences[val_slice]),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
        shuffle=False,
    )

    errors, dominant_indices = _compute_reconstruction_errors(model, sequences, batch_size)
    train_errors = errors[train_slice]
    if threshold_method == "mean_plus_3std":
        threshold = float(train_errors.mean() + 3 * train_errors.std())
    elif threshold_method == "percentile95":
        threshold = float(np.percentile(train_errors, 95))
    else:
        raise ValueError("threshold_method must be percentile95 or mean_plus_3std.")

    is_anomaly = errors > threshold
    end_indices = np.arange(sequence_length - 1, sequence_length - 1 + len(sequences))
    raw_values = df[feature_columns].to_numpy(dtype=np.float32, copy=False)
    sensor_names = np.asarray(feature_columns)[dominant_indices]
    sensor_values = raw_values[end_indices, dominant_indices]
    severities = [_severity(float(score), threshold) for score in errors]

    anomalies = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "model_name": "Autoencoder",
            "anomaly_score": errors,
            "threshold": threshold,
            "is_anomaly": is_anomaly,
            "severity": severities,
            "sensor_name": sensor_names,
            "value": sensor_values,
        }
    )

    anomalies_path.parent.mkdir(parents=True, exist_ok=True)
    anomalies.to_csv(anomalies_path, index=False)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path, include_optimizer=False)
    joblib.dump(
        {
            "scaler": scaler,
            "feature_columns": feature_columns,
            "sequence_length": sequence_length,
            "threshold": threshold,
            "threshold_method": threshold_method,
        },
        MODELS_DIR / "autoencoder_scaler.joblib",
    )

    precision = recall = f1 = None
    confusion = None
    if "anomaly_label" in df.columns:
        labels = df["anomaly_label"].iloc[end_indices].to_numpy(dtype=int)
        metrics = classification_metrics(labels, is_anomaly.astype(int))
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1_score"]
        confusion = metrics["confusion_matrix"]

    total_anomalies = int(is_anomaly.sum())
    anomaly_percentage = float(total_anomalies / max(len(anomalies), 1) * 100)
    conclusion = (
        f"Mode non supervise: {total_anomalies} anomalies detectees "
        f"({anomaly_percentage:.2f}%)."
    )

    result = {
        "anomalies": anomalies,
        "metrics_row": metric_row(
            model_name="Autoencoder",
            task="anomaly_detection",
            target_sensor=TARGET_SENSOR,
            conclusion=conclusion,
            precision=precision,
            recall=recall,
            f1_score_value=f1,
        ),
        "history": history.history,
        "threshold": threshold,
        "confusion_matrix": confusion,
    }

    tf.keras.backend.clear_session()
    gc.collect()
    return result
