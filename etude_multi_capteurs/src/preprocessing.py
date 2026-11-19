from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Doublons supprimes: {before - len(df)}")
    return df


def handle_missing_values(df, timestamp_column, numeric_columns, method="time_interpolation"):
    df = df.copy()
    if method == "time_interpolation":
        df = df.set_index(timestamp_column)
        df[numeric_columns] = df[numeric_columns].interpolate(method="time", limit_direction="both")
        df = df.reset_index()
    elif method == "mean":
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    else:
        raise ValueError("method doit etre 'time_interpolation' ou 'mean'")
    return df


def temporal_train_test_split(df, train_ratio=0.8):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio doit etre compris entre 0 et 1.")
    split_index = int(len(df) * train_ratio)
    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()


def fit_transform_scaler(train_df, test_df, numeric_columns, scaler_path):
    scaler = MinMaxScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[numeric_columns] = scaler.fit_transform(train_df[numeric_columns])
    test_scaled[numeric_columns] = scaler.transform(test_df[numeric_columns])
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    return train_scaled, test_scaled, scaler


def preprocess_data(
    df,
    timestamp_column,
    numeric_columns,
    scaler_path,
    clean_data_path,
    train_ratio=0.8,
    missing_method="time_interpolation",
):
    df = remove_duplicates(df)
    df = handle_missing_values(df, timestamp_column, numeric_columns, method=missing_method)
    train_df, test_df = temporal_train_test_split(df, train_ratio=train_ratio)
    train_scaled, test_scaled, scaler = fit_transform_scaler(train_df, test_df, numeric_columns, scaler_path)
    clean_scaled = pd.concat([train_scaled, test_scaled], axis=0).reset_index(drop=True)
    Path(clean_data_path).parent.mkdir(parents=True, exist_ok=True)
    clean_scaled.to_csv(clean_data_path, index=False)
    return clean_scaled, train_scaled, test_scaled, scaler

