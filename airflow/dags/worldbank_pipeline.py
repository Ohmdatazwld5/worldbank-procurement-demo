"""Daily idempotent ETL, drift monitoring, evaluation, and candidate training."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

from worldbank_demo_backend.evaluation import evaluate_cases
from worldbank_demo_backend.ingestion import read_structured_feed
from worldbank_demo_backend.monitoring import (
    generate_drift_report,
    generate_model_drift_report,
)
from worldbank_demo_backend.observability import emit_event
from worldbank_demo_backend.pipeline import _lake_path, run_pipeline
from worldbank_demo_backend.service import ProcurementService


def _feed_fingerprint(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_training() -> dict[str, Any]:
    import pandas as pd

    from deltalake import DeltaTable

    root = os.getenv("DELTA_LAKE_ROOT", "./data/lakehouse")
    delta_path = _lake_path(root, "silver", "procurement_records")
    if delta_path.startswith("s3a://"):
        delta_path = delta_path.replace("s3a://", "s3://", 1)
    options = __import__("json").loads(os.getenv("DELTA_STORAGE_OPTIONS_JSON", "{}"))
    table = DeltaTable(delta_path, storage_options=options or None)
    frame = table.to_pyarrow_table().to_pandas()
    frame = frame.sort_values("source_record_id", kind="stable").reset_index(drop=True)
    data_digest = hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    ).hexdigest()
    fingerprint = f"{data_digest}:{_feed_fingerprint(os.environ['PROCUREMENT_DATA_PATH'])}"
    variable_key = "worldbank_demo_last_candidate_fingerprint"
    if Variable.get(variable_key, default_var=None) == fingerprint:
        emit_event("candidate_training_skipped", reason="no_new_labeled_data")
        return {"status": "skipped", "reason": "no_new_labeled_data"}

    from worldbank_demo_backend.modeling import train_models

    result = train_models(
        delta_path,
        experiment_name=os.getenv(
            "MLFLOW_EXPERIMENT_NAME", "worldbank-procurement-demo"
        ),
        time_budget_seconds=int(os.getenv("FLAML_TIME_BUDGET_SECONDS", "60")),
        register_models=True,
    )
    Variable.set(variable_key, fingerprint)
    emit_event(
        "candidate_models_logged",
        model_names=sorted(result["models"]),
        delta_version=table.version(),
    )
    return {"status": "candidate_registered", "models": result["models"]}


def _run_drift_monitor() -> dict[str, Any]:
    reference_path = os.getenv("DRIFT_REFERENCE_DATA_PATH")
    current_path = os.getenv("PROCUREMENT_DATA_PATH")
    if not reference_path or not current_path:
        raise RuntimeError(
            "Set DRIFT_REFERENCE_DATA_PATH and PROCUREMENT_DATA_PATH for drift monitoring."
        )
    result = generate_drift_report(
        read_structured_feed(reference_path),
        read_structured_feed(current_path),
    )
    model_reference = os.getenv("MODEL_DRIFT_REFERENCE_DATA_PATH")
    model_current = os.getenv("MODEL_DRIFT_CURRENT_DATA_PATH")
    if bool(model_reference) != bool(model_current):
        raise RuntimeError(
            "Set both MODEL_DRIFT_REFERENCE_DATA_PATH and MODEL_DRIFT_CURRENT_DATA_PATH."
        )
    if model_reference and model_current:
        generate_model_drift_report(
            read_structured_feed(model_reference),
            read_structured_feed(model_current),
        )
    return {
        "status": "reported",
        "data_drift_artifact": "artifacts/monitoring/data_drift.html",
        "model_drift_reported": bool(model_reference),
    }


def _run_evaluation() -> dict[str, Any]:
    cases_path = os.getenv("EVAL_CASES_PATH")
    if not cases_path:
        raise RuntimeError("Set EVAL_CASES_PATH to human-reviewed evaluation cases.")
    import json

    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("EVAL_CASES_PATH must contain a JSON list.")
    service = ProcurementService.from_environment()
    try:
        result = evaluate_cases(
            cases,
            lambda project_id, question: service.query(
                project_id, question, user_id="airflow-evaluation"
            ),
        )
        return result
    finally:
        service.connection.close()


with DAG(
    dag_id="worldbank_procurement_demo",
    description="World Bank indicators, procurement data, documents, risk, and evaluation",
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    schedule="0 8 * * *",
    catchup=False,
    default_args={
        "owner": "data-platform",
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["world-bank", "lakehouse", "procurement", "multi-agent"],
) as dag:
    ingest_and_index = PythonOperator(
        task_id="ingest_world_bank_procurement_and_documents",
        python_callable=run_pipeline,
    )
    monitor_drift = PythonOperator(
        task_id="monitor_data_drift",
        python_callable=_run_drift_monitor,
    )
    evaluate_agents = PythonOperator(
        task_id="evaluate_retrieval_and_guardrails",
        python_callable=_run_evaluation,
    )
    train_candidates = PythonOperator(
        task_id="train_changed_data_candidate_models",
        python_callable=_run_training,
    )
    ingest_and_index >> [monitor_drift, evaluate_agents, train_candidates]
