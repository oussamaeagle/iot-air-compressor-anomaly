from pathlib import Path

import numpy as np
import pandas as pd

from src.visualization import plot_anomaly_scores


def _tf():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow est requis. Installe-le avec: pip install tensorflow") from exc
    return tf


def build_dense_autoencoder(input_shape, latent_dim=32):
    tf = _tf()
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    encoded = tf.keras.layers.Dense(latent_dim, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(encoded)
    x = tf.keras.layers.Dense(input_shape[0] * input_shape[1], activation="linear")(x)
    outputs = tf.keras.layers.Reshape(input_shape)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def train_autoencoder(
    X_train,
    X_all,
    timestamps_all,
    output_model_path,
    anomaly_output_path,
    figures_dir,
    epochs=10,
    batch_size=64,
    percentile=95,
    target_values_all=None,
):
    tf = _tf()
    model = build_dense_autoencoder(input_shape=(X_train.shape[1], X_train.shape[2]))
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
    model.fit(
        X_train,
        X_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=0,
    )

    train_reconstruction = model.predict(X_train, verbose=0)
    train_errors = np.mean(np.square(X_train - train_reconstruction), axis=(1, 2))
    threshold = float(np.percentile(train_errors, percentile))

    reconstruction = model.predict(X_all, verbose=0)
    errors = np.mean(np.square(X_all - reconstruction), axis=(1, 2))
    anomalies = (errors > threshold).astype(int)

    results = pd.DataFrame(
        {
            "timestamp": pd.Series(timestamps_all).reset_index(drop=True),
            "reconstruction_error": errors,
            "anomaly_threshold": threshold,
            "anomaly_score": errors,
            "is_anomaly": anomalies,
        }
    )
    if target_values_all is not None:
        results["target_value"] = pd.Series(target_values_all).reset_index(drop=True)
    Path(anomaly_output_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(anomaly_output_path, index=False)
    Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_model_path)
    plot_anomaly_scores(results["timestamp"], results["anomaly_score"], threshold, figures_dir)
    return model, results, threshold
