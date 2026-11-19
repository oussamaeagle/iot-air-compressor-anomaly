from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from src.anomaly_detection import missing_values_report, simple_outlier_report
from src.autoencoder_model import train_autoencoder
from src.baseline_model import evaluate_previous_value_baseline
from src.data_loader import load_iot_data, validate_columns
from src.evaluation import regression_metrics
from src.lstm_model import train_lstm
from src.preprocessing import preprocess_data
from src.sequence_builder import create_sequences, sequence_timestamps
from src.transformer_model import train_transformer
from src.visualization import (
    plot_actual_vs_predicted,
    plot_anomalies_on_series,
    plot_correlation_matrix,
    plot_model_comparison,
    plot_time_series,
)


# Configuration principale: adaptez ces variables a votre dataset.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR.parent / "data" / "raw" / "MetroPT3(AirCompressor).csv"
DATASET_NAME = "MetroPT3 Air Compressor"
TIMESTAMP_COLUMN = "timestamp"
PRIMARY_TARGET_COLUMN = "Oil_temperature"
CANDIDATE_TARGETS = ["Motor_current", "TP2", "TP3", "Reservoirs", "Oil_temperature"]
ANOMALY_LABEL_COLUMN = None
MAX_ROWS = 50000  # Mettre None pour utiliser tout le fichier.

SEQUENCE_LENGTH = 60
EPOCHS = 20
BATCH_SIZE = 32
TEST_SIZE = 0.2
TRAIN_RATIO = 1 - TEST_SIZE
TOP_K_FEATURES = 10
VALIDATION_FRACTION = 0.2
ANOMALY_THRESHOLD_PERCENTILE = 95

DATA_PROCESSED = PROJECT_DIR / "data" / "processed"
OUTPUTS = PROJECT_DIR / "outputs"
FIGURES_DIR = OUTPUTS / "figures"
MODELS_DIR = OUTPUTS / "models"
REPORTS_DIR = OUTPUTS / "reports"
POWERBI_DIR = OUTPUTS / "powerbi"
DATABASE_PATH = PROJECT_DIR / "database" / "iot_data.db"


def ensure_directories():
    for path in [DATA_PROCESSED, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, POWERBI_DIR, DATABASE_PATH.parent]:
        path.mkdir(parents=True, exist_ok=True)


def safe_name(value):
    return str(value).lower().replace(" ", "_").replace("/", "_")


def inverse_target_values(values, scaler, numeric_columns, target_column):
    """Inverse MinMax scaling for one target without reconstructing all feature columns."""
    target_idx = numeric_columns.index(target_column)
    data_min = scaler.data_min_[target_idx]
    data_max = scaler.data_max_[target_idx]
    return np.asarray(values) * (data_max - data_min) + data_min


def select_top_features(train_df, numeric_columns, target_column, top_k):
    correlations = train_df[numeric_columns].corr(numeric_only=True)[target_column].abs().dropna()
    correlations = correlations.drop(labels=[target_column], errors="ignore").sort_values(ascending=False)
    selected_predictors = correlations.head(max(top_k - 1, 0)).index.tolist()
    selected_features = [target_column] + [col for col in selected_predictors if col != target_column]

    rows = [
        {
            "target_column": target_column,
            "feature": target_column,
            "abs_correlation": 1.0,
            "rank": 0,
            "is_target_history": True,
        }
    ]
    for rank, (feature, corr_value) in enumerate(correlations.loc[selected_predictors].items(), start=1):
        rows.append(
            {
                "target_column": target_column,
                "feature": feature,
                "abs_correlation": float(corr_value),
                "rank": rank,
                "is_target_history": False,
            }
        )
    return selected_features, rows


def normalize_metric_row(metrics, target_column, model_name, train_size, test_size):
    return {
        "target_column": target_column,
        "model_name": model_name,
        "RMSE": metrics["rmse"],
        "MAE": metrics["mae"],
        "R2": metrics["r2"],
        "sequence_length": SEQUENCE_LENGTH,
        "epochs": EPOCHS if model_name != "NaivePreviousValue" else 0,
        "train_size": train_size,
        "test_size": test_size,
    }


def evaluate_target(target_column, train_df, test_df, clean_df, numeric_columns, scaler):
    selected_features, selected_rows = select_top_features(train_df, numeric_columns, target_column, TOP_K_FEATURES)
    print(f"\n[INFO] Target: {target_column}")
    print(f"[INFO] Selected features: {selected_features}")

    X_train, y_train = create_sequences(train_df[selected_features], target_column, SEQUENCE_LENGTH)
    X_test, y_test = create_sequences(test_df[selected_features], target_column, SEQUENCE_LENGTH)
    test_timestamps = sequence_timestamps(test_df[TIMESTAMP_COLUMN], SEQUENCE_LENGTH)
    target_index = selected_features.index(target_column)

    y_test_original = inverse_target_values(y_test, scaler, numeric_columns, target_column)

    def inverse_prediction(predictions):
        return inverse_target_values(predictions, scaler, numeric_columns, target_column)

    baseline_scaled, _ = evaluate_previous_value_baseline(X_test, y_test, target_index)
    baseline_original = inverse_prediction(baseline_scaled)
    baseline_metrics = regression_metrics(y_test_original, baseline_original, "NaivePreviousValue")

    _, lstm_scaled, lstm_metrics = train_lstm(
        X_train,
        y_train,
        X_test,
        y_test,
        test_timestamps,
        MODELS_DIR / f"lstm_model_{safe_name(target_column)}.keras",
        FIGURES_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_fraction=VALIDATION_FRACTION,
        y_test_for_metrics=y_test_original,
        prediction_transform=inverse_prediction,
        plot_name=f"LSTM_{target_column}",
    )
    lstm_original = inverse_prediction(lstm_scaled)

    _, transformer_scaled, transformer_metrics = train_transformer(
        X_train,
        y_train,
        X_test,
        y_test,
        test_timestamps,
        MODELS_DIR / f"transformer_model_{safe_name(target_column)}.keras",
        FIGURES_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_fraction=VALIDATION_FRACTION,
        y_test_for_metrics=y_test_original,
        prediction_transform=inverse_prediction,
        plot_name=f"Transformer_{target_column}",
    )
    transformer_original = inverse_prediction(transformer_scaled)

    plot_actual_vs_predicted(
        test_timestamps,
        y_test_original,
        baseline_original,
        f"NaivePreviousValue_{target_column}",
        FIGURES_DIR,
    )

    train_size = len(X_train)
    test_size = len(X_test)
    metric_rows = [
        normalize_metric_row(lstm_metrics, target_column, "LSTM", train_size, test_size),
        normalize_metric_row(transformer_metrics, target_column, "Transformer", train_size, test_size),
        normalize_metric_row(baseline_metrics, target_column, "NaivePreviousValue", train_size, test_size),
    ]

    prediction_df = pd.DataFrame(
        {
            "target_column": target_column,
            "timestamp": test_timestamps,
            "actual": y_test_original,
            "lstm_prediction": lstm_original,
            "transformer_prediction": transformer_original,
            "naive_previous_value_prediction": baseline_original,
        }
    )
    return prediction_df, metric_rows, selected_rows, selected_features


def compare_against_baseline(target_comparison_df):
    rows = []
    for target, group in target_comparison_df.groupby("target_column"):
        baseline = group[group["model_name"] == "NaivePreviousValue"].iloc[0]
        for _, row in group.iterrows():
            rmse_delta = row["RMSE"] - baseline["RMSE"]
            rows.append(
                {
                    **row.to_dict(),
                    "baseline_RMSE": baseline["RMSE"],
                    "RMSE_minus_baseline": rmse_delta,
                    "beats_naive_baseline": bool(row["RMSE"] < baseline["RMSE"]),
                }
            )
    return pd.DataFrame(rows)


def build_anomaly_summary(anomaly_df, threshold):
    total = len(anomaly_df)
    anomalies = int(anomaly_df["is_anomaly"].sum()) if total else 0
    return pd.DataFrame(
        [
            {
                "threshold": threshold,
                "total_points": total,
                "anomalies_detected": anomalies,
                "anomaly_percentage": round((anomalies / total) * 100, 4) if total else 0,
                "threshold_method": f"{ANOMALY_THRESHOLD_PERCENTILE}th percentile of training reconstruction error",
            }
        ]
    )


def build_kpi_summary(clean_df, anomaly_df, model_comparison_df):
    best_overall = model_comparison_df.sort_values("RMSE", ascending=True).iloc[0]
    deep_learning = model_comparison_df[model_comparison_df["model_name"].isin(["LSTM", "Transformer"])]
    best_deep = deep_learning.sort_values("RMSE", ascending=True).iloc[0]
    anomaly_count = int(anomaly_df["is_anomaly"].sum()) if not anomaly_df.empty else 0
    total = len(clean_df)
    return pd.DataFrame(
        [
            {
                "dataset_name": DATASET_NAME,
                "total_measurements": total,
                "anomalies_detected": anomaly_count,
                "anomaly_percentage": round((anomaly_count / total) * 100, 4) if total else 0,
                "best_overall_model": best_overall["model_name"],
                "best_overall_target": best_overall["target_column"],
                "best_overall_rmse": best_overall["RMSE"],
                "best_deep_learning_model": best_deep["model_name"],
                "best_deep_learning_target": best_deep["target_column"],
                "best_deep_learning_rmse": best_deep["RMSE"],
                "deep_learning_beats_naive_baseline": bool(
                    model_comparison_df[
                        model_comparison_df["model_name"].isin(["LSTM", "Transformer"])
                    ]["beats_naive_baseline"].any()
                ),
                "start_date": clean_df[TIMESTAMP_COLUMN].min(),
                "end_date": clean_df[TIMESTAMP_COLUMN].max(),
            }
        ]
    )


def generate_academic_report(target_comparison_df, model_comparison_df, anomaly_summary_df):
    best_rows = model_comparison_df.sort_values("RMSE").head(5)
    any_dl_beats_baseline = model_comparison_df[
        model_comparison_df["model_name"].isin(["LSTM", "Transformer"])
    ]["beats_naive_baseline"].any()
    oil_rows = target_comparison_df[target_comparison_df["target_column"] == "Oil_temperature"]

    lines = [
        "# Academic Interpretation Report",
        "",
        f"Dataset: **{DATASET_NAME}**.",
        "",
        "## Prediction Results",
        "",
        "The project compares LSTM, Transformer and a previous-value naive baseline on the same temporal test split. "
        "The scaler is fitted only on the training period, then applied to the test period, which avoids data leakage.",
        "",
        "The naive baseline predicts the next value using the last observed value in the input sequence. "
        "For smooth IoT signals this is a very strong reference model and must be included for academic honesty.",
        "",
    ]

    if not oil_rows.empty:
        lines.extend(
            [
                "## Oil Temperature",
                "",
                "Oil_temperature is a smooth variable in this dataset. When a signal changes slowly, the previous value is often already very close to the next value. "
                "Therefore, LSTM and Transformer models must be judged against this baseline, not only against their absolute RMSE.",
                "",
                "```text",
                oil_rows.to_string(index=False),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Negative R2",
            "",
            "A negative R2 means that the model explains less variance than a simple mean-based predictor on the test set. "
            "When this happens, the model should not be described as a strong forecasting model, even if it produces visually plausible curves.",
            "",
            "## Best Comparisons",
            "",
            "```text",
            best_rows[
                ["target_column", "model_name", "RMSE", "MAE", "R2", "baseline_RMSE", "RMSE_minus_baseline", "beats_naive_baseline"]
            ].to_string(index=False),
            "```",
            "",
            "## Scientific Conclusion",
            "",
        ]
    )

    if any_dl_beats_baseline:
        lines.append(
            "At least one deep learning model beats the naive previous-value baseline for one target. "
            "Those target/model combinations are the most defensible candidates for predictive analysis."
        )
    else:
        lines.append(
            "No LSTM or Transformer model beats the naive previous-value baseline in the current run. "
            "The deep learning models should therefore be presented as experimental models, not as strong predictors."
        )

    anomaly_row = anomaly_summary_df.iloc[0]
    lines.extend(
        [
            "",
            "## Anomaly Detection",
            "",
            "The autoencoder anomaly detection is unsupervised because no confirmed anomaly labels are provided. "
            "The threshold is defined as the 95th percentile of the training reconstruction error.",
            "",
            f"Detected anomalies: **{int(anomaly_row['anomalies_detected'])}** "
            f"out of **{int(anomaly_row['total_points'])}** sequence points "
            f"(**{anomaly_row['anomaly_percentage']}%**).",
            "",
            "These detected points are potential anomalies, not confirmed faults. They should be validated with maintenance logs, expert review, or known failure periods.",
            "",
            "## Recommendations",
            "",
            "- Keep the naive baseline in all future comparisons.",
            "- Try targets with stronger dynamics such as Motor_current, TP2 or Reservoirs.",
            "- Add lag features, rolling statistics and operating-state features.",
            "- Tune sequence length, model size, learning rate and selected feature count.",
            "- Validate anomaly detections against real failure logs if available.",
            "- Consider classical models such as ARIMA, XGBoost, Random Forest or Isolation Forest as additional baselines.",
        ]
    )

    report_path = REPORTS_DIR / "academic_interpretation.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def export_to_sqlite(clean_df, prediction_df, anomaly_df, metrics_df):
    with sqlite3.connect(DATABASE_PATH) as conn:
        clean_df.to_sql("clean_iot_data", conn, if_exists="replace", index=False)
        prediction_df.to_sql("prediction_results", conn, if_exists="replace", index=False)
        anomaly_df.to_sql("anomaly_results", conn, if_exists="replace", index=False)
        metrics_df.to_sql("model_metrics", conn, if_exists="replace", index=False)


def main():
    ensure_directories()

    df = load_iot_data(DATA_FILE, TIMESTAMP_COLUMN, max_rows=MAX_ROWS)
    available_targets = [target for target in CANDIDATE_TARGETS if target in df.columns]
    if not available_targets:
        raise ValueError(f"No candidate targets found. Available columns: {list(df.columns)}")
    validate_columns(df, [TIMESTAMP_COLUMN, PRIMARY_TARGET_COLUMN, ANOMALY_LABEL_COLUMN])

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    if ANOMALY_LABEL_COLUMN in numeric_columns:
        numeric_columns.remove(ANOMALY_LABEL_COLUMN)

    print("\nStatistiques descriptives:")
    print(df[numeric_columns].describe())
    print("\nValeurs manquantes:")
    print(missing_values_report(df))
    print("\nValeurs aberrantes simples par z-score:")
    print(simple_outlier_report(df, numeric_columns))

    plot_time_series(df, TIMESTAMP_COLUMN, numeric_columns[:6], FIGURES_DIR)
    plot_correlation_matrix(df, numeric_columns, FIGURES_DIR)

    clean_df, train_df, test_df, scaler = preprocess_data(
        df,
        timestamp_column=TIMESTAMP_COLUMN,
        numeric_columns=numeric_columns,
        scaler_path=MODELS_DIR / "minmax_scaler.joblib",
        clean_data_path=DATA_PROCESSED / "clean_data.csv",
        train_ratio=TRAIN_RATIO,
    )

    all_prediction_frames = []
    all_metric_rows = []
    all_selected_feature_rows = []
    selected_features_by_target = {}

    for target_column in available_targets:
        prediction_df, metric_rows, selected_rows, selected_features = evaluate_target(
            target_column, train_df, test_df, clean_df, numeric_columns, scaler
        )
        all_prediction_frames.append(prediction_df)
        all_metric_rows.extend(metric_rows)
        all_selected_feature_rows.extend(selected_rows)
        selected_features_by_target[target_column] = selected_features

    prediction_results = pd.concat(all_prediction_frames, ignore_index=True)
    target_comparison_df = pd.DataFrame(all_metric_rows)
    model_comparison_df = compare_against_baseline(target_comparison_df)
    selected_features_df = pd.DataFrame(all_selected_feature_rows)

    primary_anomaly_features = selected_features_by_target.get(PRIMARY_TARGET_COLUMN, numeric_columns[:TOP_K_FEATURES])
    X_train_anomaly, _ = create_sequences(train_df[primary_anomaly_features], PRIMARY_TARGET_COLUMN, SEQUENCE_LENGTH)
    X_all_anomaly, _ = create_sequences(clean_df[primary_anomaly_features], PRIMARY_TARGET_COLUMN, SEQUENCE_LENGTH)
    all_timestamps = sequence_timestamps(clean_df[TIMESTAMP_COLUMN], SEQUENCE_LENGTH)
    target_values_all = inverse_target_values(
        clean_df[PRIMARY_TARGET_COLUMN].iloc[SEQUENCE_LENGTH:].to_numpy(),
        scaler,
        numeric_columns,
        PRIMARY_TARGET_COLUMN,
    )

    _, anomaly_df, threshold = train_autoencoder(
        X_train_anomaly,
        X_all_anomaly,
        all_timestamps,
        MODELS_DIR / "autoencoder_model.keras",
        POWERBI_DIR / "anomaly_results.csv",
        FIGURES_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        percentile=ANOMALY_THRESHOLD_PERCENTILE,
        target_values_all=target_values_all,
    )

    anomaly_summary_df = build_anomaly_summary(anomaly_df, threshold)
    kpi_df = build_kpi_summary(clean_df, anomaly_df, model_comparison_df)

    clean_df.to_csv(POWERBI_DIR / "clean_iot_data.csv", index=False)
    prediction_results.to_csv(POWERBI_DIR / "prediction_results.csv", index=False)
    model_comparison_df.to_csv(POWERBI_DIR / "model_comparison.csv", index=False)
    kpi_df.to_csv(POWERBI_DIR / "kpi_summary.csv", index=False)
    anomaly_df.to_csv(POWERBI_DIR / "anomaly_results.csv", index=False)

    target_comparison_df.to_csv(REPORTS_DIR / "target_comparison.csv", index=False)
    selected_features_df.to_csv(REPORTS_DIR / "selected_features.csv", index=False)
    anomaly_summary_df.to_csv(REPORTS_DIR / "anomaly_summary.csv", index=False)
    model_comparison_df.to_csv(REPORTS_DIR / "metrics_report.csv", index=False)
    generate_academic_report(target_comparison_df, model_comparison_df, anomaly_summary_df)

    anomaly_plot_df = clean_df.merge(anomaly_df[["timestamp", "is_anomaly"]], on="timestamp", how="left")
    anomaly_plot_df["is_anomaly"] = anomaly_plot_df["is_anomaly"].fillna(0).astype(int)
    plot_anomalies_on_series(anomaly_plot_df, TIMESTAMP_COLUMN, PRIMARY_TARGET_COLUMN, "is_anomaly", FIGURES_DIR)
    plot_model_comparison(model_comparison_df, FIGURES_DIR)

    export_to_sqlite(clean_df, prediction_results, anomaly_df, model_comparison_df)

    any_dl_beats_baseline = model_comparison_df[
        model_comparison_df["model_name"].isin(["LSTM", "Transformer"])
    ]["beats_naive_baseline"].any()
    print("\nPipeline termine avec succes.")
    print(f"Resultats Power BI: {POWERBI_DIR}")
    print(f"Rapports: {REPORTS_DIR}")
    print(f"Base SQLite: {DATABASE_PATH}")
    if any_dl_beats_baseline:
        print("Au moins un modele deep learning bat la baseline naive.")
    else:
        print("Aucun modele deep learning ne bat la baseline naive dans cette execution.")


if __name__ == "__main__":
    main()
