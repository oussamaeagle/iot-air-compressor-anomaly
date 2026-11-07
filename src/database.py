"""SQLite database management for measurements, predictions and metrics."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
)
from sqlalchemy.engine import Engine

from .config import DATABASE_PATH, SENSOR_COLUMNS, TIMESTAMP_COLUMN


def get_engine(database_path: str | Path = DATABASE_PATH) -> Engine:
    """Create a SQLAlchemy engine for the SQLite database."""

    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{database_path}", future=True)


def _metadata() -> MetaData:
    """Define the database schema."""

    metadata = MetaData()
    Table(
        "iot_measurements",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("timestamp", DateTime, index=True, nullable=False),
        *(Column(column, Float) for column in SENSOR_COLUMNS),
    )
    Table(
        "predictions",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("timestamp", DateTime, index=True, nullable=False),
        Column("model_name", String(80), nullable=False),
        Column("target_sensor", String(80), nullable=False),
        Column("real_value", Float),
        Column("predicted_value", Float),
        Column("prediction_error", Float),
    )
    Table(
        "anomalies",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("timestamp", DateTime, index=True, nullable=False),
        Column("model_name", String(80), nullable=False),
        Column("anomaly_score", Float),
        Column("threshold", Float),
        Column("is_anomaly", Boolean),
        Column("severity", String(40)),
    )
    Table(
        "model_metrics",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("model_name", String(80), nullable=False),
        Column("task", String(80), nullable=False),
        Column("rmse", Float),
        Column("mae", Float),
        Column("r2", Float),
        Column("precision", Float),
        Column("recall", Float),
        Column("f1_score", Float),
        Column("created_at", DateTime, default=datetime.utcnow),
    )
    return metadata


def create_tables(engine: Engine, drop_existing: bool = False) -> None:
    """Create all project tables, optionally resetting existing tables."""

    metadata = _metadata()
    if drop_existing:
        metadata.drop_all(engine)
    metadata.create_all(engine)


def _prepare_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure timestamp values can be inserted into SQLite."""

    prepared = df.copy()
    if TIMESTAMP_COLUMN in prepared.columns:
        prepared[TIMESTAMP_COLUMN] = pd.to_datetime(prepared[TIMESTAMP_COLUMN])
    return prepared


def insert_measurements(
    engine: Engine,
    df: pd.DataFrame,
    chunksize: int = 5000,
) -> None:
    """Insert cleaned IoT measurements into iot_measurements."""

    columns = [TIMESTAMP_COLUMN, *[column for column in SENSOR_COLUMNS if column in df.columns]]
    records = _prepare_timestamp(df[columns])
    records.to_sql(
        "iot_measurements",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=chunksize,
    )


def insert_predictions(
    engine: Engine,
    predictions: pd.DataFrame,
    chunksize: int = 5000,
) -> None:
    """Insert model predictions into the predictions table."""

    if predictions.empty:
        return
    columns = [
        "timestamp",
        "model_name",
        "target_sensor",
        "real_value",
        "predicted_value",
        "prediction_error",
    ]
    records = _prepare_timestamp(predictions[columns])
    records.to_sql(
        "predictions",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=chunksize,
    )


def insert_anomalies(
    engine: Engine,
    anomalies: pd.DataFrame,
    chunksize: int = 5000,
) -> None:
    """Insert anomaly detection results into the anomalies table."""

    if anomalies.empty:
        return
    columns = [
        "timestamp",
        "model_name",
        "anomaly_score",
        "threshold",
        "is_anomaly",
        "severity",
    ]
    records = _prepare_timestamp(anomalies[columns])
    records.to_sql(
        "anomalies",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=chunksize,
    )


def insert_model_metrics(
    engine: Engine,
    metrics: pd.DataFrame,
    chunksize: int = 5000,
) -> None:
    """Insert model metrics into the model_metrics table."""

    if metrics.empty:
        return
    columns = [
        "model_name",
        "task",
        "rmse",
        "mae",
        "r2",
        "precision",
        "recall",
        "f1_score",
    ]
    records = metrics[columns].copy()
    records["created_at"] = datetime.utcnow()
    records.to_sql(
        "model_metrics",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=chunksize,
    )

