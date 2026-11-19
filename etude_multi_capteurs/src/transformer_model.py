from pathlib import Path

from src.evaluation import regression_metrics
from src.visualization import plot_actual_vs_predicted


def _tf():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow est requis. Installe-le avec: pip install tensorflow") from exc
    return tf


def transformer_encoder(inputs, head_size=64, num_heads=2, ff_dim=64, dropout=0.1):
    tf = _tf()
    x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + inputs)
    ff = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    ff = tf.keras.layers.Dropout(dropout)(ff)
    ff = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(ff)
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff)


def build_transformer_model(input_shape, head_size=64, num_heads=2, ff_dim=64, num_blocks=2, dropout=0.1):
    tf = _tf()
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    for _ in range(num_blocks):
        x = transformer_encoder(x, head_size=head_size, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def _temporal_validation_split(X_train, y_train, validation_fraction=0.2):
    split_index = int(len(X_train) * (1 - validation_fraction))
    if split_index <= 0 or split_index >= len(X_train):
        raise ValueError("Not enough training sequences for temporal validation split.")
    return X_train[:split_index], y_train[:split_index], X_train[split_index:], y_train[split_index:]


def train_transformer(
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
    plot_name="Transformer",
):
    tf = _tf()
    model = build_transformer_model(input_shape=(X_train.shape[1], X_train.shape[2]))
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
    metrics = regression_metrics(metric_y, metric_predictions, "Transformer")
    Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_model_path)
    plot_actual_vs_predicted(timestamps_test, metric_y, metric_predictions, plot_name, figures_dir)
    return model, predictions, metrics
