from pathlib import Path

import pandas as pd


def load_iot_data(file_path, timestamp_column, max_rows=None, drop_unnamed=True):
    """Charge un CSV IoT, convertit la date et trie les mesures chronologiquement."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path.resolve()}")

    df = pd.read_csv(path, nrows=max_rows)
    if drop_unnamed:
        unnamed = [col for col in df.columns if str(col).startswith("Unnamed") or str(col) == ""]
        df = df.drop(columns=unnamed, errors="ignore")

    if timestamp_column not in df.columns:
        raise ValueError(f"Colonne temps absente: {timestamp_column}. Colonnes disponibles: {list(df.columns)}")

    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors="coerce")
    invalid_dates = df[timestamp_column].isna().sum()
    if invalid_dates:
        print(f"[INFO] {invalid_dates} lignes avec date invalide supprimees.")
        df = df.dropna(subset=[timestamp_column])

    df = df.sort_values(timestamp_column).reset_index(drop=True)
    print(f"Dimensions du dataset: {df.shape}")
    print("\nTypes des colonnes:")
    print(df.dtypes)
    print("\nPremieres lignes:")
    print(df.head())
    return df


def validate_columns(df, required_columns):
    missing = [col for col in required_columns if col and col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

