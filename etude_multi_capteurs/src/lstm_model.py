from pathlib import Path

import numpy as np

from src.evaluation import regression_metrics
from src.visualization import plot_actual_vs_predicted


def _tf():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow est requis. Installe-le avec: pip install tensorflow") from exc
    return tf


def build_lstm_model(input_shape, units=64, dropout=0.2):
    tf = _tf()
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(units, return_sequences=False),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def _temporal_validation_split(X_train, y_train, validation_fraction=0.2):
    split_index = int(len(X_train) * (1 - validation_fraction))
    if split_index <= 0 or split_index >= len(X_train):
        raise ValueError("Not enough training sequences for temporal validation split.")
    return X_train[:split_index], y_train[:split_index], X_train[split_index:], y_train[split_index:]


def train_lstm(
    X_train,
    y_train,
    X_test,
    y_test,
    timestamps_test,
    output_model_path,
    figures_dir,
    epochs=20,
    batch_size=32,
    validation_fraction=0.2,
    y_test_for_metrics=None,
    prediction_transform=None,
    plot_name="LSTM",
):
    tf = _tf()
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    X_fit, y_fit, X_val, y_val = _temporal_validation_split(X_train, y_train, validation_fraction)
    model.fit(
        X_fit,
        y_fit,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=0,
    )
    predictions = model.predict(X_test, verbose=0).reshape(-1)
    metric_y = y_test_for_metrics if y_test_for_metrics is not None else y_test
    metric_predictions = prediction_transform(predictions) if prediction_transform else predictions
    metrics = regression_metrics(metric_y, metric_predictions, "LSTM")
    Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_model_path)
    plot_actual_vs_predicted(timestamps_test, metric_y, metric_predictions, plot_name, figures_dir)
    return model, predictions, metrics
