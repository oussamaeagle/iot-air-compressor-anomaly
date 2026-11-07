"""Main executable pipeline for the PFE project."""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from src.anomaly_detection import generate_alerts
from src.config import TARGET_SENSOR, ensure_directories
from src.database import (
    create_tables,
    get_engine,
    insert_anomalies,
    insert_measurements,
    insert_model_metrics,
    insert_predictions,
)
from src.evaluation import save_model_comparison
from src.preprocessing import preprocess_data
from src.reporting import export_powerbi_dataset, run_eda, save_kpi_summary
from src.train_autoencoder import train_autoencoder_model
from src.train_lstm import train_lstm_model
from src.train_transformer import train_transformer_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Pipeline IA/Data pour MetroPT3 Air Compressor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-data-path", default=None, help="Optional custom path to MetroPT3 CSV.")
    parser.add_argument("--sample-size", type=int, default=None, help="Read only the first N rows.")
    parser.add_argument(
        "--filter-comp-active",
        action="store_true",
        help="Keep only rows where COMP == 1.",
    )
    parser.add_argument("--target", default=TARGET_SENSOR, help="Target sensor to forecast.")
    parser.add_argument("--sequence-length", type=int, default=30, help="Temporal sequence length.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs for deep models.")
    parser.add_argument("--batch-size", type=int, default=512, help="Keras batch size.")
    parser.add_argument(
        "--threshold-method",
        choices=["percentile95", "mean_plus_3std"],
        default="percentile95",
        help="Autoencoder anomaly threshold strategy.",
    )
    parser.add_argument(
        "--no-reset-db",
        action="store_true",
        help="Append to existing SQLite tables instead of resetting them.",
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    """Execute the complete project workflow."""

    start = time.time()
    ensure_directories()

    print("[1/9] Preprocessing data...")
    preprocessing = preprocess_data(
        raw_data_path=args.raw_data_path,
        sample_size=args.sample_size,
        filter_comp_active=args.filter_comp_active,
    )
    clean_df = preprocessing.clean_data
    print(f"      Clean rows: {len(clean_df):,}")

    print("[2/9] Creating SQLite database and inserting measurements...")
    engine = get_engine()
    create_tables(engine, drop_existing=not args.no_reset_db)
    insert_measurements(engine, clean_df)

    print("[3/9] Running EDA...")
    eda_outputs = run_eda(clean_df)

    print("[4/9] Training LSTM and naive baseline...")
    lstm_output = train_lstm_model(
        clean_df,
        target_sensor=args.target,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print("[5/9] Training Transformer...")
    transformer_output = train_transformer_model(
        clean_df,
        target_sensor=args.target,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print("[6/9] Training Autoencoder and detecting anomalies...")
    autoencoder_output = train_autoencoder_model(
        clean_df,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        threshold_method=args.threshold_method,
    )

    print("[7/9] Saving predictions, anomalies and metrics to SQLite...")
    prediction_frames = [
        lstm_output["baseline_predictions"],
        lstm_output["predictions"],
        transformer_output["predictions"],
    ]
    predictions_for_db = pd.concat(prediction_frames, ignore_index=True)
    insert_predictions(engine, predictions_for_db)
    insert_anomalies(engine, autoencoder_output["anomalies"])

    comparison_rows = [
        lstm_output["baseline_metrics_row"],
        lstm_output["metrics_row"],
        transformer_output["metrics_row"],
        autoencoder_output["metrics_row"],
    ]
    comparison = save_model_comparison(comparison_rows)
    insert_model_metrics(engine, comparison)

    print("[8/9] Generating alerts...")
    alerts = generate_alerts(autoencoder_output["anomalies"])

    print("[9/9] Exporting Power BI datasets and KPIs...")
    powerbi_dataset = export_powerbi_dataset(
        clean_df,
        prediction_frames=prediction_frames,
        anomalies=autoencoder_output["anomalies"],
    )
    kpis = save_kpi_summary(clean_df, autoencoder_output["anomalies"], comparison)

    elapsed = time.time() - start
    print(f"Pipeline completed in {elapsed:.1f} seconds.")
    print(f"Generated {len(alerts):,} alerts and {len(powerbi_dataset):,} Power BI rows.")

    return {
        "preprocessing": preprocessing,
        "eda_outputs": eda_outputs,
        "comparison": comparison,
        "alerts": alerts,
        "kpis": kpis,
    }


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    try:
        run_pipeline(args)
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
