from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _prepare_output(output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    return output


def plot_time_series(df, timestamp_column, sensor_columns, output_dir):
    output = _prepare_output(output_dir)
    for col in sensor_columns:
        if col not in df.columns:
            continue
        plt.figure(figsize=(13, 5))
        plt.plot(df[timestamp_column], df[col], label=col, linewidth=1)
        plt.title(f"Evolution temporelle - {col}")
        plt.xlabel("Temps")
        plt.ylabel(col)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / f"time_series_{col}.png", dpi=150)
        plt.close()


def plot_correlation_matrix(df, numeric_columns, output_dir):
    output = _prepare_output(output_dir)
    if len(numeric_columns) < 2:
        return
    plt.figure(figsize=(11, 8))
    sns.heatmap(df[numeric_columns].corr(), cmap="viridis", annot=False)
    plt.title("Matrice de correlation")
    plt.tight_layout()
    plt.savefig(output / "correlation_matrix.png", dpi=150)
    plt.close()


def plot_actual_vs_predicted(timestamps, actual, predicted, model_name, output_dir):
    output = _prepare_output(output_dir)
    plt.figure(figsize=(13, 5))
    plt.plot(timestamps, actual, label="Reel", linewidth=1)
    plt.plot(timestamps, predicted, label="Predit", linewidth=1)
    plt.title(f"Valeurs reelles vs predites - {model_name}")
    plt.xlabel("Temps")
    plt.ylabel("Valeur cible normalisee")
    plt.legend()
    plt.tight_layout()
    safe_name = str(model_name).lower().replace(" ", "_").replace("/", "_")
    plt.savefig(output / f"actual_vs_predicted_{safe_name}.png", dpi=150)
    plt.close()


def plot_anomaly_scores(timestamps, scores, threshold, output_dir):
    output = _prepare_output(output_dir)
    plt.figure(figsize=(13, 5))
    plt.plot(timestamps, scores, label="Score d'anomalie", linewidth=1)
    plt.axhline(threshold, color="red", linestyle="--", label="Seuil")
    plt.title("Scores d'anomalies")
    plt.xlabel("Temps")
    plt.ylabel("Erreur de reconstruction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "anomaly_scores.png", dpi=150)
    plt.close()


def plot_anomalies_on_series(df, timestamp_column, value_column, anomaly_column, output_dir):
    output = _prepare_output(output_dir)
    anomalies = df[df[anomaly_column] == 1]
    plt.figure(figsize=(13, 5))
    plt.plot(df[timestamp_column], df[value_column], label=value_column, linewidth=1)
    plt.scatter(anomalies[timestamp_column], anomalies[value_column], color="red", s=12, label="Anomalies")
    plt.title(f"Anomalies detectees - {value_column}")
    plt.xlabel("Temps")
    plt.ylabel(value_column)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "anomalies_on_series.png", dpi=150)
    plt.close()


def plot_model_comparison(metrics_df, output_dir):
    output = _prepare_output(output_dir)
    if metrics_df.empty:
        return
    df = metrics_df.copy()
    rmse_col = "RMSE" if "RMSE" in df.columns else "rmse"
    model_col = "model_name" if "model_name" in df.columns else "model"
    if rmse_col not in df.columns or model_col not in df.columns:
        return
    plt.figure(figsize=(8, 5))
    if "target_column" in df.columns:
        sns.barplot(data=df, x="target_column", y=rmse_col, hue=model_col, palette="Set2")
        plt.xticks(rotation=30, ha="right")
        plt.xlabel("Variable cible")
    else:
        sns.barplot(data=df, x=model_col, y=rmse_col, hue=model_col, palette="Set2", legend=False)
        plt.xlabel("Modele")
    plt.title("Comparaison des modeles par RMSE")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(output / "model_comparison.png", dpi=150)
    plt.close()


def export_dataframe_plot_data(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
