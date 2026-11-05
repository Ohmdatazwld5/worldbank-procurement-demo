"""Build a safe, factual capability and deployment-status summary for the demo UI."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping

from .core import resolve_entities
from .modeling import MIN_TRAINING_ROWS


def _file_name(value: str | None) -> str | None:
    if not value:
        return None
    path_type = PureWindowsPath if "\\" in value else Path
    return path_type(value).name


def _status(is_ready: bool, *, inactive_status: str = "not_configured") -> str:
    return "active" if is_ready else inactive_status


def build_demo_overview(
    service: Any,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Summarize loaded inputs and configured capabilities without exposing secrets."""
    env = os.environ if environment is None else environment
    records = list(service.records)
    chunks = list(service.chunks)
    documents: dict[str, dict[str, Any]] = {}
    source_files: dict[str, dict[str, int]] = {}

    for row in records:
        source_name = _file_name(row.get("source_uri")) or "unspecified source"
        source_files.setdefault(source_name, {"procurement_records": 0, "document_chunks": 0})
        source_files[source_name]["procurement_records"] += 1

    for chunk in chunks:
        documents.setdefault(
            chunk.document_id,
            {
                "document_id": chunk.document_id,
                "document_type": chunk.document_type,
                "source_file": _file_name(chunk.source_uri),
            },
        )
        source_name = _file_name(chunk.source_uri) or "unspecified source"
        source_files.setdefault(source_name, {"procurement_records": 0, "document_chunks": 0})
        source_files[source_name]["document_chunks"] += 1

    document_metadata = [
        {
            "document_id": chunk.document_id,
            "project_id": chunk.project_id,
            "sector_code": chunk.sector_code,
            "document_year": chunk.document_year,
        }
        for chunk in chunks
    ]
    distinct_document_metadata = {
        row["document_id"]: row for row in document_metadata
    }
    entity_matches = resolve_entities(records, distinct_document_metadata.values())
    exact_links = sum(match.method == "project_id" for match in entity_matches)
    inferred_links = sum(match.method == "sector_temporal" for match in entity_matches)

    fixture_data = any(
        name.casefold() == "procurement_demo.csv"
        or name.casefold().startswith("illustrative_")
        for name in source_files
    ) or any(document_id.startswith("SYN-") for document_id in documents)
    workflow = service.workflow
    loaded_models = sorted(getattr(workflow, "automl_models", {}))
    feedback_path = Path(env.get("FEEDBACK_JSONL_PATH", "data/feedback.jsonl"))
    feedback_count = 0
    if feedback_path.is_file():
        with feedback_path.open("r", encoding="utf-8") as handle:
            feedback_count = sum(1 for line in handle if line.strip())

    evaluation_path = env.get("EVAL_CASES_PATH")
    evaluation_configured = bool(evaluation_path and Path(evaluation_path).is_file())
    otel_configured = bool(env.get("OTEL_EXPORTER_OTLP_ENDPOINT"))
    delta_backed = bool(service.delta_root)
    catalog_mode = str(env.get("CATALOG_MODE", "hive")).casefold()
    evidently_installed = importlib.util.find_spec("evidently") is not None
    drift_report_dir = Path("artifacts/monitoring")
    drift_report_available = any(
        (drift_report_dir / filename).is_file()
        for filename in ("data_drift.json", "data_drift.html")
    )
    monitoring_partially_configured = (
        evaluation_configured or drift_report_available or otel_configured
    )

    return {
        "data": {
            "procurement_records": len(records),
            "projects": len({str(row.get("project_id")) for row in records if row.get("project_id")}),
            "documents": len(documents),
            "evidence_chunks": len(chunks),
            "source_files": [
                {"file": name, **counts}
                for name, counts in sorted(source_files.items())
            ],
            "classification": "fictional_demo_fixtures" if fixture_data else "source_provenance_unverified",
            "ai_procurement_workbook_connected": False,
            "disclaimer": (
                "Bundled procurement and document examples are fictional test fixtures. "
                "The AI Procurement Contracts workbook is separate and is not loaded by this Q&A service."
            ),
        },
        "knowledge_model": {
            "exact_project_links": exact_links,
            "sector_and_temporal_links": inferred_links,
            "graph_materialized_in_serving_service": False,
            "topology": [
                "Project -> Contract -> Supplier",
                "Project -> Document -> EvidenceChunk",
                "Project -> Sector / Country",
                "Project -> Document via exact project ID or sector and bounded year distance",
            ],
            "resolution_policy": "Exact project ID first; otherwise sector match within two years, with lower confidence.",
            "document_processing": {
                "formats": ["PDF", "DOCX", "TXT"],
                "narrative_chunk_words": 512,
                "narrative_overlap_words": 64,
                "tables": "Preserve row groups and repeat headers",
                "ocr": "Scanned PDFs require a separate OCR preprocessing step",
            },
        },
        "analytics": {
            "risk_method": (
                "trained_models_and_transparent_heuristic"
                if loaded_models
                else "transparent_heuristic_only"
            ),
            "loaded_model_types": loaded_models,
            "registered_model_predictions_available": bool(loaded_models),
            "model_training_minimum_rows": MIN_TRAINING_ROWS,
            "model_training_note": "Training requires reviewed, labeled procurement history; demo fixtures are not valid training data.",
        },
        "guardrails": {
            "api_access_control": "Bearer roles and project-level authorization are enforced by the API.",
            "object_store_acl": "Must be configured separately in the storage provider.",
            "citation_validation": True,
            "abstain_when_evidence_is_insufficient": True,
            "conflicting_icr_isr_policy": "Preserve both sources and require human review for same-claim, same-period conflicts.",
            "human_decision_required": True,
        },
        "operations": {
            "serving_store": "Delta Lake snapshot" if delta_backed else "DuckDB in-memory",
            "delta_tables_configured": delta_backed,
            "catalog": (
                "Nessie/Iceberg configured; Delta tables remain path-addressed"
                if delta_backed and catalog_mode == "nessie"
                else "Hive Metastore for Spark ETL"
                if delta_backed
                else "not active in the Render API service"
            ),
            "airflow_dag_in_repository": True,
            "airflow_scheduler_running_with_this_render_service": False,
            "evaluation_cases_configured": evaluation_configured,
            "evaluation_metrics_available": False,
            "evidently_installed": evidently_installed,
            "drift_report_available": drift_report_available,
            "otel_export_configured": otel_configured,
            "application_events_to_stdout": True,
            "feedback_entries_recorded": feedback_count,
            "feedback_requires_human_review": True,
            "feedback_storage": (
                "ephemeral" if str(feedback_path).replace("\\", "/").startswith("/tmp/") else "local_path"
            ),
        },
        "features": [
            {
                "id": "ingestion",
                "title": "Structured and document ingestion",
                "status": _status(bool(records or chunks), inactive_status="no_inputs_loaded"),
                "detail": f"{len(records)} procurement rows and {len(documents)} parsed documents are loaded for this service.",
            },
            {
                "id": "lakehouse",
                "title": "PySpark, Delta Lake OSS, and catalog ETL",
                "status": (
                    "delta_serving_snapshot"
                    if delta_backed
                    else "code_available_not_running"
                ),
                "detail": (
                    "The API reads configured Delta Lake OSS snapshots. This Render web service does not run PySpark or Airflow."
                    if delta_backed
                    else "The ETL pipeline and Delta/Hive or Nessie configuration are in the repository; this Render web service does not run Spark or Airflow."
                ),
            },
            {
                "id": "knowledge_graph",
                "title": "Entity resolution and knowledge graph",
                "status": _status(bool(entity_matches), inactive_status="available_no_links"),
                "detail": f"{exact_links} exact-project and {inferred_links} sector/time links are available from the loaded records.",
            },
            {
                "id": "retrieval",
                "title": "LangGraph evidence retrieval and guardrails",
                "status": "active",
                "detail": "The request graph retrieves evidence, adjudicates annotated ICR/ISR conflicts, validates citations, and abstains when unsupported.",
            },
            {
                "id": "predictive_models",
                "title": "FLAML models tracked with MLflow",
                "status": _status(bool(loaded_models), inactive_status="training_and_registry_not_configured"),
                "detail": f"{len(loaded_models)} model types are loaded. The current rule-based risk signal is not a trained probability.",
            },
            {
                "id": "evaluation_monitoring",
                "title": "Evaluation, drift, and telemetry",
                "status": (
                    "partially_configured"
                    if monitoring_partially_configured
                    else "not_configured_on_render"
                ),
                "detail": "Evaluation, Evidently drift reports, and OpenTelemetry export require their reviewed inputs and external services.",
            },
            {
                "id": "feedback",
                "title": "Human-reviewed feedback loop",
                "status": "active",
                "detail": f"{feedback_count} feedback entries recorded; feedback is not automatically used as a model label.",
            },
        ],
        "kpi_targets": {
            "planning_cycle_time_reduction": "15% target; not measured",
            "retrieval_precision_at_5": "0.80 target; not measured",
            "retrieval_recall_at_5": "0.75 target; not measured",
            "citation_coverage": "1.00 target; not measured",
            "abstention_accuracy": "0.95 target; not measured",
        },
    }
