"""Public-source, tabular, and long-document ingestion adapters."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from .core import EvidenceChunk, chunk_document, stable_id

LOGGER = logging.getLogger(__name__)
WORLD_BANK_API = "https://api.worldbank.org/v2"


def fetch_world_bank_indicator(
    country_codes: Iterable[str],
    indicator_code: str,
    *,
    start_year: int,
    end_year: int,
    timeout_seconds: int = 30,
) -> list[dict[str, Any]]:
    """Fetch paginated indicator observations; World Bank records remain source-linked."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    if start_year > end_year:
        raise ValueError("start_year must not exceed end_year.")
    session = requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=4,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset({"GET"}),
            )
        ),
    )
    records: list[dict[str, Any]] = []
    for country in country_codes:
        page = 1
        while True:
            response = session.get(
                f"{WORLD_BANK_API}/country/{country}/indicator/{indicator_code}",
                params={
                    "format": "json",
                    "date": f"{start_year}:{end_year}",
                    "per_page": 1000,
                    "page": page,
                },
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list) or len(payload) < 2:
                raise ValueError(f"Unexpected World Bank response for {country}/{indicator_code}.")
            metadata, observations = payload[0], payload[1] or []
            if not isinstance(metadata, dict) or not isinstance(observations, list):
                raise ValueError("World Bank API returned malformed pagination data.")
            for item in observations:
                if item.get("value") is None:
                    continue
                records.append(
                    {
                        "country_code": country,
                        "country": (item.get("country") or {}).get("value"),
                        "year": int(item["date"]),
                        "indicator_code": indicator_code,
                        "value": float(item["value"]),
                        "source_uri": response.url,
                    }
                )
            page_count = int(metadata.get("pages") or 1)
            if page >= page_count:
                break
            page += 1
            time.sleep(0.1)
    return records


def read_structured_feed(path: str | Path) -> list[dict[str, Any]]:
    """Load CSV or JSON/JSONL rows; no implicit field renaming is applied."""
    import pandas as pd

    source = Path(path)
    suffix = source.suffix.casefold()
    if suffix == ".csv":
        frame = pd.read_csv(source)
    elif suffix == ".jsonl":
        frame = pd.read_json(source, lines=True)
    elif suffix == ".json":
        frame = pd.read_json(source)
    else:
        raise ValueError(f"Unsupported structured feed: {source.suffix}")
    if frame.empty:
        LOGGER.warning("Structured feed contains zero rows: %s", source)
        return []
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


def _parse_pdf(path: Path) -> list[tuple[str, str, str]]:
    import pdfplumber

    pages: list[tuple[str, str, str]] = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((f"Page {page_number}", text, "narrative"))
            for table_number, table in enumerate(page.extract_tables() or [], start=1):
                rows = [
                    " | ".join("" if cell is None else str(cell).replace("\n", " ") for cell in row)
                    for row in table
                ]
                if rows:
                    pages.append(
                        (
                            f"Table {table_number} | page {page_number}",
                            "\n".join(rows),
                            "table",
                        )
                    )
    return pages


def _parse_docx(path: Path) -> list[tuple[str, str, str]]:
    from docx import Document

    document = Document(path)
    blocks: list[tuple[str, str, str]] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            blocks.append(("Document", text, "narrative"))
    for table_number, table in enumerate(document.tables, start=1):
        rows = [
            " | ".join(cell.text.replace("\n", " ").strip() for cell in row.cells)
            for row in table.rows
        ]
        if rows:
            blocks.append((f"Table {table_number}", "\n".join(rows), "table"))
    return blocks


def parse_document(
    path: str | Path,
    metadata: Mapping[str, Any],
    *,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[EvidenceChunk]:
    """Parse PDF/DOCX/TXT and require provenance metadata from a manifest.

    `metadata` must include document_id and document_type. Project, sector, date,
    and document type are never guessed from report text.
    """
    source = Path(path)
    document_id = str(metadata.get("document_id") or "").strip()
    document_type = str(metadata.get("document_type") or "").strip()
    if not document_id or not document_type:
        raise ValueError("Document manifest must provide document_id and document_type.")
    if source.suffix.casefold() == ".pdf":
        blocks = _parse_pdf(source)
    elif source.suffix.casefold() == ".docx":
        blocks = _parse_docx(source)
    elif source.suffix.casefold() == ".txt":
        blocks = [("Document", source.read_text(encoding="utf-8"), "narrative")]
    else:
        raise ValueError(f"Unsupported document format: {source.suffix}")

    chunks: list[EvidenceChunk] = []
    for section_title, body, block_type in blocks:
        if block_type == "table":
            chunks.extend(
                _chunk_table(
                    section_title,
                    body,
                    document_id=document_id,
                    document_type=document_type,
                    project_id=metadata.get("project_id"),
                    sector_code=metadata.get("sector_code"),
                    document_year=_to_year(metadata.get("document_year")),
                    source_uri=str(source.resolve()),
                    confidence_tier=int(metadata.get("confidence_tier", 3)),
                    max_tokens=max_tokens,
                    effective_date=metadata.get("effective_date"),
                )
            )
            continue
        text = f"{section_title}\n{body}"
        chunks.extend(
            chunk_document(
                text,
                document_id=document_id,
                document_type=document_type,
                project_id=metadata.get("project_id"),
                sector_code=metadata.get("sector_code"),
                document_year=_to_year(metadata.get("document_year")),
                source_uri=str(source.resolve()),
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                confidence_tier=int(metadata.get("confidence_tier", 3)),
                chunk_type="narrative",
            )
        )
    if metadata.get("effective_date"):
        chunks = [
            EvidenceChunk(
                **{**chunk.__dict__, "effective_date": str(metadata["effective_date"])}
            )
            for chunk in chunks
        ]
    annotations = metadata.get("claim_annotations", [])
    if not isinstance(annotations, list):
        raise ValueError("claim_annotations must be a list.")
    for annotation in annotations:
        if not isinstance(annotation, dict) or not annotation.get("section_title"):
            raise ValueError("Each claim annotation needs a section_title.")
        if not annotation.get("claim_key") or not annotation.get("claim_value"):
            raise ValueError("Each claim annotation needs claim_key and claim_value.")
        matched = False
        for index, chunk in enumerate(chunks):
            if chunk.section_title == annotation["section_title"]:
                chunks[index] = EvidenceChunk(
                    **{
                        **chunk.__dict__,
                        "effective_date": metadata.get("effective_date"),
                        "claim_key": str(annotation["claim_key"]),
                        "claim_value": str(annotation["claim_value"]),
                    }
                )
                matched = True
        if not matched:
            raise ValueError(
                f"Claim annotation section not found in parsed document: {annotation['section_title']!r}"
            )
    return chunks


def _chunk_table(
    section_title: str,
    table_text: str,
    *,
    document_id: str,
    document_type: str,
    project_id: str | None,
    sector_code: str | None,
    document_year: int | None,
    source_uri: str,
    confidence_tier: int,
    max_tokens: int,
    effective_date: str | None,
) -> list[EvidenceChunk]:
    """Keep row boundaries intact; repeat the header in each bounded row group."""
    from .core import stable_id

    rows = [line.strip() for line in table_text.splitlines() if line.strip()]
    if not rows:
        return []
    header = rows[0]
    groups: list[list[str]] = []
    current: list[str] = []
    current_words = len(header.split())
    for row in rows[1:]:
        row_words = len(row.split())
        if current and current_words + row_words > max_tokens:
            groups.append(current)
            current = []
            current_words = len(header.split())
        current.append(row)
        current_words += row_words
    if current:
        groups.append(current)
    if not groups:
        groups = [[]]

    result: list[EvidenceChunk] = []
    for group_number, group in enumerate(groups, start=1):
        body = "\n".join([header, *group])
        chunk_text = f"{section_title} | rows {group_number}\n{body}"
        result.append(
            EvidenceChunk(
                document_id=document_id,
                chunk_id=stable_id(document_id, section_title, group_number, body, prefix="chunk-"),
                chunk_text=chunk_text,
                section_title=section_title,
                document_type=document_type.upper(),
                project_id=project_id,
                sector_code=sector_code,
                document_year=document_year,
                source_uri=source_uri,
                confidence_tier=confidence_tier,
                chunk_type="table",
                effective_date=effective_date,
            )
        )
    return result


def read_document_manifest(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read JSON manifest keyed by filename; caller owns source-document review."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    documents = data.get("documents")
    if not isinstance(documents, list):
        raise ValueError("Manifest JSON must have a 'documents' list.")
    result: dict[str, dict[str, Any]] = {}
    for item in documents:
        if not isinstance(item, dict) or not item.get("file"):
            raise ValueError("Each document manifest item needs a 'file' value.")
        name = Path(str(item["file"])).name
        if name in result:
            raise ValueError(f"Duplicate document manifest entry: {name}")
        result[name] = dict(item)
    return result


def embed_chunks(chunks: list[EvidenceChunk], model_name: str) -> list[EvidenceChunk]:
    """Generate normalized sentence-transformer vectors for semantic retrieval."""
    from sentence_transformers import SentenceTransformer

    if not chunks:
        return []
    model = SentenceTransformer(model_name)
    vectors = model.encode(
        [chunk.chunk_text for chunk in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return [
        EvidenceChunk(
            **{
                **chunk.__dict__,
                "embedding": tuple(float(value) for value in vector),
            }
        )
        for chunk, vector in zip(chunks, vectors)
    ]


def write_delta_rows(
    rows: list[Mapping[str, Any]],
    *,
    delta_path: str,
    spark: Any = None,
    mode: str = "append",
) -> None:
    """Persist rows using Delta Lake OSS; caller configures Spark/Delta extensions."""
    if mode not in {"append", "overwrite"}:
        raise ValueError("mode must be append or overwrite.")
    if not rows:
        LOGGER.info("Skipping empty Delta write to %s.", delta_path)
        return
    if spark is None:
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("No active SparkSession; configure PySpark and Delta Lake first.")
    spark.createDataFrame([dict(row) for row in rows]).write.format("delta").mode(mode).save(delta_path)


def upsert_delta_rows(
    rows: list[Mapping[str, Any]],
    *,
    delta_path: str,
    key_columns: tuple[str, ...],
    spark: Any = None,
) -> None:
    """Idempotently merge source rows into a Delta path using stable business keys."""
    if not rows:
        LOGGER.info("Skipping empty Delta merge to %s.", delta_path)
        return
    if not key_columns:
        raise ValueError("At least one Delta merge key is required.")
    missing = [key for key in key_columns if any(row.get(key) is None for row in rows)]
    if missing:
        raise ValueError(f"Delta merge key values cannot be null: {sorted(set(missing))}")
    if spark is None:
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("No active SparkSession; configure PySpark and Delta Lake first.")

    source = spark.createDataFrame([dict(row) for row in rows])
    from delta.tables import DeltaTable

    if not DeltaTable.isDeltaTable(spark, delta_path):
        source.write.format("delta").mode("overwrite").save(delta_path)
        return
    condition = " AND ".join(
        f"target.`{key}` = source.`{key}`" for key in key_columns
    )
    (
        DeltaTable.forPath(spark, delta_path)
        .alias("target")
        .merge(source.alias("source"), condition)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )


def source_record_id(row: Mapping[str, Any], source_uri: str) -> str:
    stable_key = next(
        (
            row.get(name)
            for name in (
                "contract_id",
                "tender_id",
                "notice_id",
                "procurement_id",
                "amendment_id",
                "milestone_id",
                "record_id",
                "id",
            )
            if row.get(name) not in (None, "")
        ),
        None,
    )
    identity = (
        json.dumps(
            [row.get("source_system"), row.get("project_id"), str(stable_key)],
            sort_keys=True,
            default=str,
        )
        if stable_key is not None
        else json.dumps(dict(row), sort_keys=True, default=str)
    )
    return stable_id(
        row.get("source_system", source_uri if stable_key is None else ""),
        identity,
        prefix="record-",
    )


def _to_year(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value)[:4])
    except (TypeError, ValueError):
        raise ValueError(f"Invalid document_year in manifest: {value!r}") from None
