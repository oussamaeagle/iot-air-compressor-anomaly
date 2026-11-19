import numpy as np


def simple_outlier_report(df, numeric_columns, z_threshold=3.0):
    report = {}
    for col in numeric_columns:
        std = df[col].std()
        if std == 0 or np.isnan(std):
            report[col] = 0
            continue
        z_scores = np.abs((df[col] - df[col].mean()) / std)
        report[col] = int((z_scores > z_threshold).sum())
    return report


def missing_values_report(df):
    return df.isna().sum().sort_values(ascending=False)

