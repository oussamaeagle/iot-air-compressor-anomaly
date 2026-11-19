import numpy as np


def create_sequences(data, target_column, sequence_length):
    """Transforme un DataFrame numerique en X 3D et y 1D pour LSTM/Transformer."""
    if target_column not in data.columns:
        raise ValueError(f"Colonne cible absente: {target_column}")
    if sequence_length < 2:
        raise ValueError("sequence_length doit etre >= 2")
    if len(data) <= sequence_length:
        raise ValueError("Pas assez de lignes pour creer les sequences.")

    values = data.to_numpy(dtype="float32")
    target_idx = data.columns.get_loc(target_column)
    X, y = [], []
    for i in range(len(values) - sequence_length):
        X.append(values[i : i + sequence_length])
        y.append(values[i + sequence_length, target_idx])
    return np.asarray(X, dtype="float32"), np.asarray(y, dtype="float32")


def sequence_timestamps(timestamps, sequence_length):
    return timestamps.iloc[sequence_length:].reset_index(drop=True)

