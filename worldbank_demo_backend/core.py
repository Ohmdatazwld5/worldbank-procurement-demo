"""Dependency-light domain logic for ingestion, retrieval, risk, and guardrails."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class EvidenceChunk:
    document_id: str
    chunk_id: str
    chunk_text: str
    section_title: str
    document_type: str
    project_id: str | None = None
    sector_code: str | None = None
    document_year: int | None = None
    source_uri: str | None = None
    effective_date: str | None = None
    confidence_tier: int = 3
    chunk_type: str = "narrative"
    effective_date: str | None = None
    claim_key: str | None = None
    claim_value: str | None = None
    embedding: Sequence[float] = field(default_factory=tuple)

    @property
    def citation_id(self) -> str:
        section = self.section_title.replace("]", ")").strip()
        return f"[{self.document_id} | {section} | chunk:{self.chunk_id}]"


@dataclass(frozen=True)
class EntityMatch:
    project_id: str
    document_id: str
    method: str
    confidence: float
    year_distance: int | None


def stable_id(*parts: object, prefix: str = "") -> str:
    """Return a content-derived identifier suitable for idempotent ingestion."""
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:24]}"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_document(
    text: str,
    *,
    document_id: str,
    document_type: str,
    project_id: str | None = None,
    sector_code: str | None = None,
    document_year: int | None = None,
    source_uri: str | None = None,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    confidence_tier: int = 3,
    chunk_type: str = "narrative",
) -> list[EvidenceChunk]:
    """Split sectioned text into overlapping word windows, retaining section labels.

    Tables should be supplied as textual blocks by the parser, not flattened into
    unrelated cells. Chunk boundaries are word-based and deterministic.
    """
    if max_tokens < 1 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("Require max_tokens > overlap_tokens >= 0.")

    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    sections = re.split(
        r"(?im)^(?=(?:section|annex|appendix)\s+[a-z0-9][^\n]*$)",
        normalized,
    )
    chunks: list[EvidenceChunk] = []
    for section_number, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        lines = section.splitlines()
        heading = lines[0].strip() if lines else f"Section {section_number + 1}"
        body = normalize_text("\n".join(lines[1:]) if len(lines) > 1 else section)
        words = body.split()
        if not words:
            words = section.split()
        step = max_tokens - overlap_tokens
        for start in range(0, len(words), step):
            window = words[start : start + max_tokens]
            if not window:
                continue
            chunk_text = f"{heading}\n{' '.join(window)}"
            chunk_id = stable_id(document_id, heading, start, chunk_text, prefix="chunk-")
            chunks.append(
                EvidenceChunk(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    section_title=heading,
                    document_type=document_type.upper(),
                    project_id=project_id,
                    sector_code=sector_code,
                    document_year=document_year,
                    source_uri=source_uri,
                    confidence_tier=confidence_tier,
                    chunk_type=chunk_type,
                )
            )
            if start + max_tokens >= len(words):
                break
    return chunks


def resolve_entities(
    projects: Iterable[Mapping[str, Any]],
    documents: Iterable[Mapping[str, Any]],
    *,
    temporal_window_years: int = 2,
) -> list[EntityMatch]:
    """Prefer exact project IDs; otherwise link on sector and bounded time distance."""
    if temporal_window_years < 0:
        raise ValueError("temporal_window_years cannot be negative.")
    docs = list(documents)
    matches: list[EntityMatch] = []
    for project in projects:
        project_id = str(project.get("project_id") or "").strip()
        if not project_id:
            continue
        project_sector = str(project.get("sector_code") or "").strip().casefold()
        project_year = _as_year(project.get("project_year"))
        exact = [
            doc
            for doc in docs
            if str(doc.get("project_id") or "").strip() == project_id
        ]
        if exact:
            for doc in exact:
                matches.append(
                    EntityMatch(project_id, str(doc["document_id"]), "project_id", 1.0, 0)
                )
            continue

        candidates: list[tuple[int, Mapping[str, Any]]] = []
        if not project_sector or project_year is None:
            continue
        for doc in docs:
            document_id = str(doc.get("document_id") or "").strip()
            doc_sector = str(doc.get("sector_code") or "").strip().casefold()
            doc_year = _as_year(doc.get("document_year"))
            if not document_id or doc_sector != project_sector or doc_year is None:
                continue
            distance = abs(project_year - doc_year)
            if distance <= temporal_window_years:
                candidates.append((distance, doc))
        for distance, doc in sorted(candidates, key=lambda item: (item[0], str(item[1]["document_id"]))):
            confidence = max(0.5, 0.8 - 0.1 * distance)
            matches.append(
                EntityMatch(
                    project_id,
                    str(doc["document_id"]),
                    "sector_temporal",
                    confidence,
                    distance,
                )
            )
    return matches


def _as_year(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(str(value)[:4])
    except (TypeError, ValueError):
        return None
    return parsed if 1900 <= parsed <= date.today().year + 10 else None


def heuristic_risk(record: Mapping[str, Any]) -> dict[str, Any]:
    """Transparent fallback score; it is not a trained or calibrated prediction."""
    reasons: list[str] = []
    planned = _number(record.get("planned_days"))
    actual = _number(record.get("actual_days"))
    amendments = _number(record.get("amendment_count")) or 0
    outcome = str(record.get("outcome") or "").casefold()
    delay_ratio = ((actual - planned) / planned) if planned and actual is not None else None
    if outcome in {"rebid", "re-bid", "retendered"}:
        reasons.append("recorded outcome indicates rebidding")
    if amendments >= 2:
        reasons.append("recorded amendment count is at least two")
    if delay_ratio is not None and delay_ratio >= 0.25:
        reasons.append("recorded completion time exceeded plan by at least 25%")
    score = min(
        1.0,
        (0.50 if outcome in {"rebid", "re-bid", "retendered"} else 0.0)
        + min(amendments, 4) * 0.10
        + (0.25 if delay_ratio is not None and delay_ratio >= 0.25 else 0.0),
    )
    level = "HIGH" if score >= 0.65 else "MEDIUM" if score >= 0.30 else "LOW"
    return {
        "risk_level": level,
        "risk_score": round(score, 4),
        "reasons": reasons or ["no configured historical risk trigger was observed"],
        "method": "transparent_heuristic_not_a_trained_prediction",
        "delay_ratio": delay_ratio,
    }


def resolve_conflicting_evidence(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare like-for-like claims without treating all ICRs as universally superior.

    ICRs are terminal retrospective sources; ISRs describe supervision at their
    effective date. If evidence makes incompatible claims about the same period,
    retain both and escalate instead of silently selecting a winner.
    """
    normalized = [dict(item) for item in items]
    icrs = [item for item in normalized if str(item.get("document_type", "")).upper() == "ICR"]
    isrs = [item for item in normalized if str(item.get("document_type", "")).upper() == "ISR"]
    if not icrs or not isrs:
        return {"status": "NO_CROSS_TYPE_CONFLICT", "selected": normalized, "conflicts": []}

    conflicts: list[dict[str, Any]] = []
    for icr in icrs:
        for isr in isrs:
            same_project = bool(icr.get("project_id")) and (
                icr.get("project_id") == isr.get("project_id")
            )
            same_period = (
                icr.get("claim_key")
                and icr.get("claim_key") == isr.get("claim_key")
            )
            icr_claim = str(icr.get("claim_value") or "").strip().casefold()
            isr_claim = str(isr.get("claim_value") or "").strip().casefold()
            if same_project and same_period and icr_claim and isr_claim and icr_claim != isr_claim:
                conflicts.append(
                    {
                        "claim_key": icr.get("claim_key"),
                        "icr_citation": icr.get("citation_id"),
                        "isr_citation": isr.get("citation_id"),
                        "icr_value": icr.get("claim_value"),
                        "isr_value": isr.get("claim_value"),
                        "resolution": "preserve_both_and_escalate_for_human_review",
                    }
                )

    if conflicts:
        return {"status": "CONFLICTING_EVIDENCE", "selected": normalized, "conflicts": conflicts}
    return {
        "status": "DIFFERENT_TIME_OR_CLAIM_SCOPE",
        "selected": sorted(
            normalized,
            key=lambda item: (
                str(item.get("effective_date") or ""),
                str(item.get("document_type") or ""),
            ),
        ),
        "conflicts": [],
    }


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def retrieve_by_embedding(
    query_vector: Sequence[float],
    chunks: Iterable[EvidenceChunk],
    *,
    limit: int = 5,
    minimum_score: float = 0.15,
    project_id: str | None = None,
    document_ids: set[str] | None = None,
) -> list[tuple[EvidenceChunk, float]]:
    if limit < 1:
        raise ValueError("limit must be at least one.")
    scored: list[tuple[EvidenceChunk, float]] = []
    for chunk in chunks:
        if document_ids is not None and chunk.document_id not in document_ids:
            continue
        if project_id and chunk.project_id and chunk.project_id != project_id:
            continue
        score = cosine_similarity(query_vector, chunk.embedding)
        if score >= minimum_score:
            scored.append((chunk, score))
    return sorted(scored, key=lambda item: (-item[1], item[0].document_id))[:limit]


def validate_citations(
    claims: Sequence[Mapping[str, Any]],
    allowed_citations: set[str],
) -> list[str]:
    """Return validation errors; every factual claim must carry a retrieved citation."""
    errors: list[str] = []
    for index, claim in enumerate(claims):
        citations = claim.get("citations") or []
        if not citations:
            errors.append(f"claim[{index}] has no citation")
        elif any(citation not in allowed_citations for citation in citations):
            errors.append(f"claim[{index}] cites a source not present in retrieval results")
    return errors


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def chunk_to_dict(chunk: EvidenceChunk) -> dict[str, Any]:
    result = asdict(chunk)
    result["embedding"] = list(chunk.embedding)
    result["citation_id"] = chunk.citation_id
    return result
