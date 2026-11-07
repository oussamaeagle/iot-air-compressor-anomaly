"""Central configuration for the IoT air compressor project."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
MODELS_DIR = PROJECT_ROOT / "models"
DATABASE_DIR = PROJECT_ROOT / "database"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

RAW_DATA_PATH = RAW_DATA_DIR / "MetroPT3(AirCompressor).csv"
FALLBACK_RAW_DATA_PATH = PROJECT_ROOT.parent / "MetroPT3(AirCompressor).csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "preprocessed_data.csv"
SCALED_DATA_PATH = PROCESSED_DATA_DIR / "preprocessed_scaled_data.csv"
MISSING_REPORT_PATH = RESULTS_DIR / "missing_values.csv"

DATABASE_PATH = DATABASE_DIR / "iot_air_compressor.db"

LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.h5"
TRANSFORMER_MODEL_PATH = MODELS_DIR / "transformer_model.h5"
AUTOENCODER_MODEL_PATH = MODELS_DIR / "autoencoder_model.h5"

LSTM_PREDICTIONS_PATH = RESULTS_DIR / "lstm_predictions.csv"
TRANSFORMER_PREDICTIONS_PATH = RESULTS_DIR / "transformer_predictions.csv"
BASELINE_PREDICTIONS_PATH = RESULTS_DIR / "baseline_predictions.csv"
AUTOENCODER_ANOMALIES_PATH = RESULTS_DIR / "autoencoder_anomalies.csv"
MODEL_COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"

POWERBI_DATASET_PATH = DASHBOARD_DIR / "powerbi_dataset.csv"
ALERTS_PATH = DASHBOARD_DIR / "alerts.csv"
KPI_SUMMARY_PATH = DASHBOARD_DIR / "kpi_summary.csv"

TIMESTAMP_COLUMN = "timestamp"
TARGET_SENSOR = "Oil_temperature"

SENSOR_COLUMNS = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]

PREDICTION_FEATURE_COLUMNS = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Motor_current",
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]

AUTOENCODER_FEATURE_COLUMNS = SENSOR_COLUMNS.copy()

EDA_TIME_SERIES_COLUMNS = [
    "Oil_temperature",
    "Motor_current",
    "TP2",
    "TP3",
    "DV_pressure",
    "Reservoirs",
]


def ensure_directories() -> None:
    """Create all project directories required by the pipeline."""

    for directory in [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        DASHBOARD_DIR,
        MODELS_DIR,
        DATABASE_DIR,
        NOTEBOOKS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def resolve_raw_data_path(raw_data_path: str | Path | None = None) -> Path:
    """Return the available MetroPT3 CSV path.

    The expected location is data/raw, but the fallback keeps the project easy
    to run from the initial workspace where the file was provided at root.
    """

    if raw_data_path is not None:
        candidate = Path(raw_data_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Raw data file not found: {candidate}")

    if RAW_DATA_PATH.exists():
        return RAW_DATA_PATH
    if FALLBACK_RAW_DATA_PATH.exists():
        return FALLBACK_RAW_DATA_PATH
    raise FileNotFoundError(
        "MetroPT3(AirCompressor).csv was not found in data/raw or project parent."
    )

