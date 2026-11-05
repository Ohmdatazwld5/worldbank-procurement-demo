"""Evidently drift reports and OpenTelemetry metrics/exporter setup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .modeling import APPROVED_FEATURES


def generate_drift_report(
    reference_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
    *,
    output_dir: str | Path = "artifacts/monitoring",
) -> dict[str, Any]:
    """Generate an Evidently data drift report and persist JSON/HTML artifacts."""
    import pandas as pd
    from evidently import Report
    from evidently.presets import DataDriftPreset

    if not reference_rows or not current_rows:
        raise ValueError("Both reference and current observations are required for drift monitoring.")
    reference = pd.DataFrame(reference_rows)
    current = pd.DataFrame(current_rows)
    columns = sorted(
        set(reference.columns).intersection(current.columns).intersection(APPROVED_FEATURES)
    )
    if not columns:
        raise ValueError("Reference and current feeds have no shared approved model features.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = Report([DataDriftPreset()])
    result = report.run(
        reference_data=reference[columns],
        current_data=current[columns],
    )
    payload = result.dict()
    (output / "data_drift.json").write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
    (output / "data_drift.html").write_text(result.get_html(), encoding="utf-8")
    return payload


def generate_model_drift_report(
    reference_predictions: list[dict[str, Any]],
    current_predictions: list[dict[str, Any]],
    *,
    output_dir: str | Path = "artifacts/monitoring",
) -> dict[str, Any]:
    """Compare persisted model outputs using Evidently's numeric/categorical drift."""
    import pandas as pd
    from evidently import Report
    from evidently.presets import DataDriftPreset

    if not reference_predictions or not current_predictions:
        raise ValueError("Both reference and current prediction samples are required.")
    reference = pd.DataFrame(reference_predictions)
    current = pd.DataFrame(current_predictions)
    shared_columns = sorted(set(reference.columns).intersection(current.columns))
    tracked = {
        "timeline_days",
        "amendment_probability",
        "rebid_probability",
        "risk_level",
        "model_version",
    }
    columns = [column for column in shared_columns if column in tracked]
    if not columns:
        raise ValueError("Prediction samples have no shared tracked output columns.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = Report([DataDriftPreset()]).run(
        reference_data=reference[columns],
        current_data=current[columns],
    )
    payload = result.dict()
    (output / "model_drift.json").write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
    (output / "model_drift.html").write_text(result.get_html(), encoding="utf-8")
    return payload


def _record_operation_histogram() -> Any:
    global _OPERATION_HISTOGRAM
    if _OPERATION_HISTOGRAM is None:
        return None
    return _OPERATION_HISTOGRAM


def configure_otel(service_name: str = "worldbank-procurement-demo") -> None:
    """Configure OTLP export only when an endpoint has been explicitly provided."""
    import os

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({"service.name": service_name})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(tracer_provider)
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=endpoint))],
    )
    metrics.set_meter_provider(meter_provider)
    global _OPERATION_HISTOGRAM
    _OPERATION_HISTOGRAM = metrics.get_meter(service_name).create_histogram(
        "worldbank.operation.duration", unit="ms"
    )


_OPERATION_HISTOGRAM: Any = None
