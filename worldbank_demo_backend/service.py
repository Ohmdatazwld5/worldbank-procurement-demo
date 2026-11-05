"""DuckDB-backed query service, retriever, model loading, and feedback capture."""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any

from .agents import ProcurementGraph
from .core import EvidenceChunk
from .ingestion import parse_document, read_document_manifest, read_structured_feed, source_record_id
from .observability import emit_event, trace_operation


def _delta_rows(path: str) -> list[dict[str, Any]]:
    import json

    from deltalake import DeltaTable

    raw_options = os.getenv("DELTA_STORAGE_OPTIONS_JSON", "{}")
    try:
        options = json.loads(raw_options)
    except json.JSONDecodeError as exc:
        raise ValueError("DELTA_STORAGE_OPTIONS_JSON must be valid JSON.") from exc
    if not isinstance(options, dict):
        raise ValueError("DELTA_STORAGE_OPTIONS_JSON must be a JSON object.")
    table = DeltaTable(path, storage_options=options or None)
    return table.to_pyarrow_table().to_pylist()


def _delta_path(root: str, *parts: str) -> str:
    if root.startswith(("s3://", "s3a://", "abfs://", "abfss://", "gs://", "dbfs:/", "file:/")):
        value = root.rstrip("/") + "/" + "/".join(part.strip("/") for part in parts)
    else:
        value = str(Path(root).joinpath(*parts))
    return value.replace("s3a://", "s3://", 1) if value.startswith("s3a://") else value


def _read_delta_snapshot(
    procurement_path: str, evidence_path: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[int, int]]:
    for _ in range(3):
        before = (_delta_version(procurement_path), _delta_version(evidence_path))
        records = _delta_rows(procurement_path)
        chunks = _delta_rows(evidence_path)
        after = (_delta_version(procurement_path), _delta_version(evidence_path))
        if before == after:
            return records, chunks, after
    raise RuntimeError("Delta inputs changed repeatedly while a serving snapshot was loaded.")


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> list[float]:
        vector = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        return [float(value) for value in vector]

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors]


class ProcurementService:
    def __init__(
        self,
        records: list[dict[str, Any]],
        chunks: list[EvidenceChunk],
        *,
        embedding_model: str,
        automl_models: dict[str, Any] | None = None,
        model_uris: dict[str, str] | None = None,
        embedder: SentenceTransformerEmbedder | None = None,
        delta_root: str | None = None,
        delta_versions: tuple[int, int] | None = None,
    ) -> None:
        import duckdb
        import pandas as pd

        self.connection = duckdb.connect(database=":memory:")
        self.records = records
        self.chunks = chunks
        self.delta_root = delta_root
        self.delta_versions = delta_versions
        self._snapshot_lock = threading.RLock()
        if records:
            missing = {
                "project_id",
                "sector_code",
                "planned_days",
                "actual_days",
                "amendment_count",
                "outcome",
            }.difference(records[0])
            if missing:
                raise ValueError(f"Procurement rows are missing required fields: {sorted(missing)}")
        self.embedder = embedder or SentenceTransformerEmbedder(embedding_model)
        if records:
            frame = pd.DataFrame(records)
            self.connection.register("procurement_input", frame)
            self.connection.execute(
                "CREATE TABLE procurement_records AS SELECT * FROM procurement_input"
            )
        else:
            self.connection.execute(
                "CREATE TABLE procurement_records (project_id VARCHAR, sector_code VARCHAR, planned_days DOUBLE, actual_days DOUBLE, amendment_count INTEGER, outcome VARCHAR, source_record_id VARCHAR, source_uri VARCHAR)"
            )
        self.workflow = ProcurementGraph(
            records,
            chunks,
            self.embedder,
            automl_models=automl_models,
            model_uris=model_uris,
        )

    @classmethod
    def from_environment(cls) -> "ProcurementService":
        procurement_path = os.getenv("PROCUREMENT_DATA_PATH")
        delta_root = os.getenv("DELTA_LAKE_ROOT")
        documents_dir = os.getenv("DOCUMENTS_DIR")
        manifest_path = os.getenv("DOCUMENT_MANIFEST_PATH")
        records: list[dict[str, Any]] = []
        chunks: list[EvidenceChunk] = []
        delta_versions: tuple[int, int] | None = None
        if delta_root:
            procurement_table_path = _delta_path(delta_root, "silver", "procurement_records")
            evidence_table_path = _delta_path(delta_root, "silver", "evidence_chunks")
            records, raw_chunks, delta_versions = _read_delta_snapshot(
                procurement_table_path, evidence_table_path
            )
            chunks = [
                EvidenceChunk(
                    **{
                        key: row.get(key)
                        for key in EvidenceChunk.__dataclass_fields__
                        if key in row
                    }
                )
                for row in raw_chunks
            ]
        elif procurement_path:
            path = Path(procurement_path)
            records = read_structured_feed(path)
            for record in records:
                record.setdefault("source_uri", str(path.resolve()))
                record["source_record_id"] = source_record_id(record, str(path.resolve()))

        if not delta_root and documents_dir and manifest_path:
            root = Path(documents_dir)
            manifest = read_document_manifest(manifest_path)
            for filename, metadata in manifest.items():
                path = root / filename
                if not path.is_file():
                    raise FileNotFoundError(f"Manifest document does not exist: {path}")
                chunks.extend(parse_document(path, metadata))
        elif not delta_root and bool(documents_dir) != bool(manifest_path):
            raise ValueError("Set both DOCUMENTS_DIR and DOCUMENT_MANIFEST_PATH, or neither.")

        if not records and not chunks:
            raise RuntimeError(
                "No procurement records or evidence documents are available. "
                "Configure DELTA_LAKE_ROOT, PROCUREMENT_DATA_PATH, or DOCUMENTS_DIR with DOCUMENT_MANIFEST_PATH."
            )
        embedding_model = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        embedder = SentenceTransformerEmbedder(embedding_model)
        if chunks and any(not chunk.embedding for chunk in chunks):
            vectors = embedder.encode_batch([chunk.chunk_text for chunk in chunks])
            chunks = [
                EvidenceChunk(**{**chunk.__dict__, "embedding": tuple(vector)})
                for chunk, vector in zip(chunks, vectors)
            ]
        automl_models = {}
        loaded_model_uris: dict[str, str] = {}
        model_environment = {
            "timeline_days": "MLFLOW_MODEL_TIMELINE_URI",
            "amendment_probability": "MLFLOW_MODEL_AMENDMENT_URI",
            "rebid_probability": "MLFLOW_MODEL_REBID_URI",
        }
        configured_models = {
            key: os.getenv(environment_name)
            for key, environment_name in model_environment.items()
            if os.getenv(environment_name)
        }
        if configured_models:
            import mlflow

            automl_models = {
                key: mlflow.pyfunc.load_model(uri)
                for key, uri in configured_models.items()
            }
            loaded_model_uris = {
                key: str(uri) for key, uri in configured_models.items()
            }
        return cls(
            records,
            chunks,
            embedding_model=embedding_model,
            automl_models=automl_models,
            model_uris=loaded_model_uris,
            embedder=embedder,
            delta_root=delta_root,
            delta_versions=delta_versions,
        )

    def query(
        self,
        project_id: str,
        question: str,
        *,
        package_context: dict[str, Any] | None = None,
        user_id: str,
    ) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        with trace_operation("procurement_query", project_id=project_id, request_id=request_id):
            with self._snapshot_lock:
                self._refresh_delta_snapshot()
                # DuckDB parameter binding prevents SQL injection in analytics lookups.
                rows = self.connection.execute(
                    "SELECT * FROM procurement_records WHERE project_id = ? LIMIT 1",
                    [project_id],
                ).fetchdf().to_dict(orient="records")
                state = self.workflow.invoke(project_id, question, package_context)
                answer = state.get("answer", {})
                risk = answer.get("risk", {})
                predictions = risk.get("predictions", {})
                if predictions:
                    prediction_record = {
                        "request_id": request_id,
                        "project_id": project_id,
                        "model_version": "|".join(
                            self.workflow.model_uris.get(name, "unversioned")
                            for name in sorted(predictions)
                        ),
                        **predictions,
                    }
                    prediction_path = Path(
                        os.getenv("PREDICTION_LOG_PATH", "data/predictions.jsonl")
                    )
                    prediction_path.parent.mkdir(parents=True, exist_ok=True)
                    with prediction_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(prediction_record, ensure_ascii=False) + "\n")
                event = emit_event(
                    "agent_request",
                    request_id=request_id,
                    user_id=user_id,
                    project_id=project_id,
                    status=answer.get("status", "UNKNOWN"),
                    citation_count=len(answer.get("citations", [])),
                    result_row_found=bool(rows),
                    model_predictions=predictions,
                    historical_risk_level=risk.get("risk_level"),
                )
                return {"request_id": request_id, **answer, "event_id": event["event_id"]}

    def analytics(self) -> list[dict[str, Any]]:
        with self._snapshot_lock:
            self._refresh_delta_snapshot()
            return self.connection.execute(
                """
            SELECT sector_code, COUNT(*) AS procurement_count,
                   AVG(TRY_CAST(actual_days AS DOUBLE)) AS avg_actual_days,
                   AVG(TRY_CAST(amendment_count AS DOUBLE)) AS avg_amendments
            FROM procurement_records
            GROUP BY sector_code
            ORDER BY sector_code
            """
            ).fetchdf().to_dict(orient="records")

    def _refresh_delta_snapshot(self) -> None:
        if self.delta_root is None:
            return
        procurement_path = _delta_path(
            self.delta_root, "silver", "procurement_records"
        )
        evidence_path = _delta_path(
            self.delta_root, "silver", "evidence_chunks"
        )
        records, raw_chunks, current_versions = _read_delta_snapshot(
            procurement_path, evidence_path
        )
        if current_versions == self.delta_versions:
            return
        chunks = [
            EvidenceChunk(
                **{
                    key: row.get(key)
                    for key in EvidenceChunk.__dataclass_fields__
                    if key in row
                }
            )
            for row in raw_chunks
        ]
        if chunks and any(not chunk.embedding for chunk in chunks):
            vectors = self.embedder.encode_batch([chunk.chunk_text for chunk in chunks])
            chunks = [
                EvidenceChunk(**{**chunk.__dict__, "embedding": tuple(vector)})
                for chunk, vector in zip(chunks, vectors)
            ]
        self._replace_duckdb_records(records)
        self.records = records
        self.chunks = chunks
        self.workflow = ProcurementGraph(
            records,
            chunks,
            self.embedder,
            automl_models=self.workflow.automl_models,
            model_uris=self.workflow.model_uris,
        )
        self.delta_versions = current_versions

    def _replace_duckdb_records(self, records: list[dict[str, Any]]) -> None:
        import pandas as pd

        if records:
            missing = {
                "project_id",
                "sector_code",
                "planned_days",
                "actual_days",
                "amendment_count",
                "outcome",
            }.difference(records[0])
            if missing:
                raise ValueError(f"Refreshed procurement rows are missing fields: {sorted(missing)}")
            frame = pd.DataFrame(records)
            self.connection.register("procurement_refresh", frame)
            try:
                self.connection.execute(
                    "CREATE OR REPLACE TABLE procurement_records AS SELECT * FROM procurement_refresh"
                )
            finally:
                self.connection.unregister("procurement_refresh")
            return
        self.connection.execute(
            "CREATE OR REPLACE TABLE procurement_records (project_id VARCHAR, sector_code VARCHAR, planned_days DOUBLE, actual_days DOUBLE, amendment_count INTEGER, outcome VARCHAR, source_record_id VARCHAR, source_uri VARCHAR)"
        )

    @staticmethod
    def record_feedback(
        request_id: str,
        rating: int,
        *,
        comment: str | None = None,
        expected_citations: list[str] | None = None,
    ) -> dict[str, Any]:
        if not 1 <= rating <= 5:
            raise ValueError("rating must be between 1 and 5.")
        feedback = {
            "feedback_id": str(uuid.uuid4()),
            "request_id": request_id,
            "rating": rating,
            "comment": comment,
            "expected_citations": expected_citations or [],
            "requires_human_review": True,
        }
        path = Path(os.getenv("FEEDBACK_JSONL_PATH", "data/feedback.jsonl"))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(feedback, ensure_ascii=False) + "\n")
        emit_event("user_feedback", **feedback)
        return feedback


def _delta_version(path: str) -> int:
    import json

    from deltalake import DeltaTable

    options = json.loads(os.getenv("DELTA_STORAGE_OPTIONS_JSON", "{}"))
    if not isinstance(options, dict):
        raise ValueError("DELTA_STORAGE_OPTIONS_JSON must be a JSON object.")
    return int(DeltaTable(path, storage_options=options or None).version())
