"""Manually triggered local validation of Airflow scheduling and Delta/Hive."""

from __future__ import annotations

from datetime import datetime, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator

from worldbank_demo_backend.local_smoke import run_delta_hive_smoke


with DAG(
    dag_id="worldbank_local_stack_smoke",
    description="Verify a scheduled Airflow task can write and read Delta through Hive.",
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    tags=["local", "smoke-test", "delta", "hive"],
) as dag:
    validate_delta_hive = PythonOperator(
        task_id="validate_delta_hive_catalog",
        python_callable=run_delta_hive_smoke,
    )
