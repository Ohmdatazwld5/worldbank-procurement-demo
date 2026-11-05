"""Airflow-invokable World Bank/procurement/document ETL into Delta Lake."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any

from .core import chunk_to_dict, heuristic_risk, resolve_entities, stable_id
from .governance import prepare_delta_spark
from .ingestion import (
    embed_chunks,
    fetch_world_bank_indicator,
    parse_document,
    read_document_manifest,
    read_structured_feed,
    source_record_id,
    upsert_delta_rows,
)
from .observability import emit_event, trace_operation


DEFAULT_INDICATORS = (
    "NY.GDP.MKTP.KD.ZG",
    "FP.CPI.TOTL.ZG",
    "SP.POP.TOTL",
    "GE.EST",
    "RQ.EST",
    "CC.EST",
)


def create_delta_spark() -> Any:
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName("worldbank-procurement-demo")
    builder = prepare_delta_spark(builder)
    if os.getenv("CATALOG_MODE", "hive").casefold() == "hive":
        warehouse_dir = os.getenv("SPARK_SQL_WAREHOUSE_DIR")
        if warehouse_dir:
            builder = builder.config("spark.sql.warehouse.dir", warehouse_dir)
        builder = builder.enableHiveSupport()
    return builder.getOrCreate()


def _lake_path(root: str, *parts: str) -> str:
    if root.startswith(("s3://", "s3a://", "abfs://", "abfss://", "gs://", "dbfs:/", "file:/")):
        return root.rstrip("/") + "/" + "/".join(part.strip("/") for part in parts)
    return str(Path(root).joinpath(*parts))


def ingest_world_bank(spark: Any, root: str, countries: list[str], start_year: int, end_year: int) -> list[dict[str, Any]]:
    rows = []
    for indicator in DEFAULT_INDICATORS:
        for row in fetch_world_bank_indicator(
            countries, indicator, start_year=start_year, end_year=end_year
        ):
            rows.append({"layer": "bronze", **row})
    if not rows:
        raise RuntimeError("World Bank ingestion returned no observations.")
    upsert_delta_rows(
        rows,
        delta_path=_lake_path(root, "bronze", "world_bank_indicators"),
        key_columns=("country_code", "indicator_code", "year"),
        spark=spark,
    )
    return rows


def enrich_with_governance(
    records: list[dict[str, Any]],
    indicator_rows: list[dict[str, Any]],
    *,
    maximum_year_gap: int = 2,
) -> list[dict[str, Any]]:
    """Attach nearest prior/adjacent World Bank WGI values by country and year."""
    governance_indicators = {
        "GE.EST": "governance_effectiveness",
        "RQ.EST": "regulatory_quality",
        "CC.EST": "control_of_corruption",
    }
    indexed: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in indicator_rows:
        indicator = row.get("indicator_code")
        if indicator in governance_indicators:
            indexed.setdefault((str(row.get("country_code")), str(indicator)), []).append(row)
    for values in indexed.values():
        values.sort(key=lambda item: int(item["year"]))

    enriched: list[dict[str, Any]] = []
    for record in records:
        result = dict(record)
        code = str(record.get("country_code") or "")
        try:
            project_year = int(record.get("project_year"))
        except (TypeError, ValueError):
            project_year = None
        for indicator, feature in governance_indicators.items():
            candidates = indexed.get((code, indicator), [])
            nearby = [
                row
                for row in candidates
                if project_year is not None
                and 0 <= project_year - int(row["year"]) <= maximum_year_gap
            ]
            selected = (
                max(nearby, key=lambda row: int(row["year"]))
                if nearby
                else None
            )
            result[feature] = float(selected["value"]) if selected else None
            result[f"{feature}_source_year"] = int(selected["year"]) if selected else None
        enriched.append(result)
    return enriched


def ingest_procurement(
    spark: Any, root: str, source_path: str, indicator_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    source = Path(source_path)
    feed_sources = [source]
    approved_sources = [
        Path(value.strip())
        for value in os.getenv("APPROVED_CONTRACT_UPDATE_FEEDS", "").split(",")
        if value.strip()
    ]
    feed_sources.extend(approved_sources)
    records: list[dict[str, Any]] = []
    for feed in feed_sources:
        feed_records = read_structured_feed(feed)
        is_approved_feed = feed in approved_sources
        if is_approved_feed:
            for record in feed_records:
                if (
                    str(record.get("review_status", "")).casefold() != "approved"
                    or not str(record.get("reviewed_by") or "").strip()
                    or not str(record.get("reviewed_at") or "").strip()
                    or not str(record.get("source_uri") or "").strip()
                ):
                    raise ValueError(
                        f"Reviewed contract update feed {feed} contains a row without "
                        "approved status, reviewer, review date, or source URI."
                    )
                try:
                    _parse_reviewed_at(record["reviewed_at"])
                except ValueError as exc:
                    raise ValueError(
                        f"Reviewed contract update feed {feed} has an invalid reviewed_at value."
                    ) from exc
        for record in feed_records:
            record["feed_uri"] = str(feed.resolve())
            record["_approved_update"] = is_approved_feed
        records.extend(feed_records)
    if not records:
        raise ValueError(f"Procurement feeds have no rows: {feed_sources}")
    required = {"project_id", "sector_code", "project_year"}
    for index, record in enumerate(records):
        missing = required.difference(record.keys())
        if missing:
            raise ValueError(
                f"Procurement feed row {index + 1} is missing join keys: {sorted(missing)}"
            )
        if not any(
            record.get(name)
            for name in (
                "contract_id",
                "tender_id",
                "notice_id",
                "procurement_id",
                "record_id",
                "id",
            )
        ):
            raise ValueError(
                f"Procurement feed row {index + 1} needs a stable contract/tender ID."
            )
    snapshots: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, str] = {}
    reviewed_dates: dict[str, datetime | None] = {}
    for row in records:
        source_id = source_record_id(row, str(row["feed_uri"]))
        content = {
            key: value
            for key, value in row.items()
            if key not in {"feed_uri", "_approved_update"}
        }
        fingerprint = stable_id(json.dumps(content, sort_keys=True, default=str))
        if source_id in snapshots:
            if fingerprints[source_id] == fingerprint:
                continue
            if not row["_approved_update"]:
                raise ValueError(
                    f"Duplicate contract snapshot {source_id} is not reviewer-approved."
                )
            current_review_date = _parse_reviewed_at(row["reviewed_at"])
            previous_review_date = reviewed_dates[source_id]
            if previous_review_date is not None and current_review_date <= previous_review_date:
                raise ValueError(
                    f"Duplicate contract snapshot {source_id} has no later reviewed_at."
                )
        else:
            current_review_date = (
                _parse_reviewed_at(row["reviewed_at"])
                if row["_approved_update"]
                else None
            )
        snapshots[source_id] = {
            key: value
            for key, value in row.items()
            if key != "_approved_update"
        }
        fingerprints[source_id] = fingerprint
        reviewed_dates[source_id] = current_review_date

    canonical_records = list(snapshots.values())
    enriched_records = enrich_with_governance(canonical_records, indicator_rows)
    clean_records = []
    for row in enriched_records:
        clean = dict(row)
        clean.update(
            {
                "source_uri": str(row.get("source_uri") or row["feed_uri"]),
                "source_record_id": source_record_id(row, str(row["feed_uri"])),
                "layer": "silver",
            }
        )
        clean_records.append(clean)
    upsert_delta_rows(
        clean_records,
        delta_path=_lake_path(root, "silver", "procurement_records"),
        key_columns=("source_record_id",),
        spark=spark,
    )
    return clean_records


def _parse_reviewed_at(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("reviewed_at must include a timezone.")
    return parsed


def write_gold_risk_signals(spark: Any, root: str, records: list[dict[str, Any]]) -> int:
    rows = []
    for record in records:
        risk = heuristic_risk(record)
        rows.append(
            {
                "project_id": record["project_id"],
                "source_record_id": record["source_record_id"],
                "source_uri": record["source_uri"],
                "risk_level": risk["risk_level"],
                "risk_score": risk["risk_score"],
                "risk_reasons": risk["reasons"],
                "scoring_method": risk["method"],
                "model_prediction": False,
                "layer": "gold",
            }
        )
    upsert_delta_rows(
        rows,
        delta_path=_lake_path(root, "gold", "procurement_risk_signals"),
        key_columns=("source_record_id",),
        spark=spark,
    )
    return len(rows)


def ingest_documents(
    spark: Any,
    root: str,
    documents_dir: str,
    manifest_path: str,
    embedding_model: str,
) -> list[dict[str, Any]]:
    directory = Path(documents_dir)
    manifest = read_document_manifest(manifest_path)
    chunks = []
    for filename, metadata in manifest.items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Manifest document does not exist: {path}")
        chunks.extend(parse_document(path, metadata))
    if not chunks:
        raise ValueError("Document ingestion produced no text or table chunks.")
    embedded = embed_chunks(chunks, embedding_model)
    rows = [chunk_to_dict(chunk) for chunk in embedded]
    for row in rows:
        row["layer"] = "silver"
    upsert_delta_rows(
        rows,
        delta_path=_lake_path(root, "silver", "evidence_chunks"),
        key_columns=("chunk_id",),
        spark=spark,
    )
    return rows


def build_entity_resolution(
    spark: Any,
    root: str,
    procurement_records: list[dict[str, Any]],
    document_chunks: list[dict[str, Any]],
    temporal_window_years: int = 2,
) -> list[dict[str, Any]]:
    documents = {
        row["document_id"]: row
        for row in document_chunks
    }
    matches = [
        {
            **match.__dict__,
            "layer": "silver",
            "document_source_uri": documents.get(match.document_id, {}).get("source_uri"),
        }
        for match in resolve_entities(
            procurement_records,
            list(documents.values()),
            temporal_window_years=temporal_window_years,
        )
    ]
    if not matches:
        emit_event("entity_resolution_empty", projects=len(procurement_records), documents=len(documents))
    upsert_delta_rows(
        matches,
        delta_path=_lake_path(root, "silver", "entity_resolution"),
        key_columns=("project_id", "document_id"),
        spark=spark,
    )
    return matches


def build_knowledge_graph(
    spark: Any,
    root: str,
    procurement_records: list[dict[str, Any]],
    document_chunks: list[dict[str, Any]],
    entity_matches: list[dict[str, Any]],
) -> dict[str, int]:
    """Persist a Delta adjacency model for project-contract-supplier-document links."""
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}

    def add_node(node_id: str, node_type: str, label: str, source_uri: str | None) -> None:
        nodes[node_id] = {
            "node_id": node_id,
            "node_type": node_type,
            "node_label": label,
            "source_uri": source_uri,
            "layer": "silver",
        }

    def add_edge(
        source_id: str,
        target_id: str,
        relationship: str,
        confidence: float,
        method: str,
        source_record_id: str | None = None,
    ) -> None:
        edge_id = stable_id(source_id, target_id, relationship, prefix="edge-")
        edges[edge_id] = {
            "edge_id": edge_id,
            "source_node": source_id,
            "target_node": target_id,
            "relationship": relationship,
            "confidence": confidence,
            "resolution_method": method,
            "source_record_id": source_record_id,
            "layer": "silver",
        }

    for record in procurement_records:
        project_id = str(record["project_id"])
        project_node = f"project:{project_id}"
        add_node(project_node, "Project", project_id, record.get("source_uri"))
        sector = str(record.get("sector_code") or "").strip()
        if sector:
            sector_node = f"sector:{sector.casefold()}"
            add_node(sector_node, "Sector", sector, None)
            add_edge(project_node, sector_node, "belongs_to_sector", 1.0, "source_record", record.get("source_record_id"))
        country = str(record.get("country_code") or "").strip()
        if country:
            country_node = f"country:{country.upper()}"
            add_node(country_node, "Country", country.upper(), None)
            add_edge(project_node, country_node, "located_in", 1.0, "source_record", record.get("source_record_id"))
        contract_id = str(
            record.get("contract_id")
            or record.get("tender_id")
            or record.get("source_record_id")
        )
        contract_node = f"contract:{contract_id}"
        add_node(contract_node, "Contract", contract_id, record.get("source_uri"))
        add_edge(contract_node, project_node, "for_project", 1.0, "project_id", record.get("source_record_id"))
        supplier_id = str(record.get("supplier_id") or "").strip()
        if supplier_id:
            supplier_node = f"supplier:{supplier_id}"
            add_node(supplier_node, "Supplier", supplier_id, record.get("source_uri"))
            add_edge(contract_node, supplier_node, "awarded_to", 1.0, "supplier_id", record.get("source_record_id"))

    document_ids = set()
    for chunk in document_chunks:
        document_id = str(chunk["document_id"])
        if document_id in document_ids:
            continue
        document_ids.add(document_id)
        document_node = f"document:{document_id}"
        add_node(document_node, "Document", document_id, chunk.get("source_uri"))
        sector = str(chunk.get("sector_code") or "").strip()
        if sector:
            sector_node = f"sector:{sector.casefold()}"
            add_node(sector_node, "Sector", sector, None)
            add_edge(document_node, sector_node, "documents_sector", 1.0, "document_metadata")

    for match in entity_matches:
        project_node = f"project:{match['project_id']}"
        document_node = f"document:{match['document_id']}"
        add_edge(
            project_node,
            document_node,
            "has_project_evidence",
            float(match["confidence"]),
            str(match["method"]),
        )

    upsert_delta_rows(
        list(nodes.values()),
        delta_path=_lake_path(root, "silver", "kg_nodes"),
        key_columns=("node_id",),
        spark=spark,
    )
    upsert_delta_rows(
        list(edges.values()),
        delta_path=_lake_path(root, "silver", "kg_edges"),
        key_columns=("edge_id",),
        spark=spark,
    )
    return {"nodes": len(nodes), "edges": len(edges)}


def run_pipeline() -> dict[str, Any]:
    root = os.getenv("DELTA_LAKE_ROOT", "./data/lakehouse")
    countries = [item.strip() for item in os.getenv("WORLD_BANK_COUNTRIES", "IND,USA,CHN").split(",") if item.strip()]
    start_year = int(os.getenv("WORLD_BANK_START_YEAR", "2015"))
    end_year = int(os.getenv("WORLD_BANK_END_YEAR", "2025"))
    procurement_path = os.getenv("PROCUREMENT_DATA_PATH")
    documents_dir = os.getenv("DOCUMENTS_DIR")
    manifest_path = os.getenv("DOCUMENT_MANIFEST_PATH")
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    if not procurement_path:
        raise RuntimeError("Set PROCUREMENT_DATA_PATH to a labeled project-level procurement CSV or JSON feed.")
    if not documents_dir or not manifest_path:
        raise RuntimeError("Set DOCUMENTS_DIR and DOCUMENT_MANIFEST_PATH for document ingestion.")

    spark = create_delta_spark()
    try:
        with trace_operation("worldbank_ingestion"):
            indicator_rows = ingest_world_bank(spark, root, countries, start_year, end_year)
        with trace_operation("procurement_ingestion"):
            records = ingest_procurement(spark, root, procurement_path, indicator_rows)
        with trace_operation("gold_risk_signals"):
            risk_signals = write_gold_risk_signals(spark, root, records)
        with trace_operation("document_ingestion"):
            chunks = ingest_documents(
                spark, root, documents_dir, manifest_path, embedding_model
            )
        with trace_operation("entity_resolution"):
            matches = build_entity_resolution(spark, root, records, chunks)
        with trace_operation("knowledge_graph"):
            graph_counts = build_knowledge_graph(
                spark, root, records, chunks, matches
            )
        result = {
            "world_bank_indicator_rows": len(indicator_rows),
            "procurement_rows": len(records),
            "gold_risk_signals": risk_signals,
            "document_chunks": len(chunks),
            "entity_matches": len(matches),
            **{f"knowledge_graph_{key}": value for key, value in graph_counts.items()},
            "delta_lake_root": root if "://" in root or root.startswith("dbfs:/") else str(Path(root).resolve()),
        }
        emit_event("pipeline_completed", **result)
        return result
    except Exception as exc:
        emit_event("pipeline_failed", error_type=type(exc).__name__, detail=str(exc))
        raise
    finally:
        spark.stop()


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all",), default="all")
    parser.parse_args()
    print(json.dumps(run_pipeline(), indent=2))


if __name__ == "__main__":
    main()
