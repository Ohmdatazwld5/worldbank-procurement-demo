from __future__ import annotations

import unittest
from pathlib import Path

from worldbank_demo_backend.core import (
    EvidenceChunk,
    chunk_document,
    heuristic_risk,
    resolve_conflicting_evidence,
    resolve_entities,
    retrieve_by_embedding,
    stable_id,
    validate_citations,
)
from worldbank_demo_backend.ingestion import (
    _chunk_table,
    parse_document,
    read_document_manifest,
    source_record_id,
)
from worldbank_demo_backend.pipeline import enrich_with_governance


class CoreBehaviorTests(unittest.TestCase):
    def test_content_ids_are_deterministic(self) -> None:
        self.assertEqual(stable_id("x", 1), stable_id("x", 1))
        self.assertNotEqual(stable_id("x", 1), stable_id("x", 2))
        self.assertNotEqual(
            source_record_id({"contract_id": "C-1", "project_id": "P-1"}, "feed.csv"),
            source_record_id({"contract_id": "C-1", "project_id": "P-2"}, "feed.csv"),
        )
        self.assertEqual(
            source_record_id({"contract_id": "C-1", "project_id": "P-1"}, "feed-a.csv"),
            source_record_id({"contract_id": "C-1", "project_id": "P-1"}, "feed-b.csv"),
        )

    def test_chunking_respects_word_window_and_overlap(self) -> None:
        chunks = chunk_document(
            "Section 1: Work\n" + " ".join(f"w{index}" for index in range(12)),
            document_id="D-1",
            document_type="ICR",
            max_tokens=5,
            overlap_tokens=2,
        )
        self.assertGreater(len(chunks), 1)
        first_words = chunks[0].chunk_text.splitlines()[1].split()
        second_words = chunks[1].chunk_text.splitlines()[1].split()
        self.assertEqual(first_words[-2:], second_words[:2])
        self.assertTrue(all(chunk.citation_id.startswith("[D-1 |") for chunk in chunks))

    def test_entity_resolution_prefers_project_id_then_sector_year(self) -> None:
        projects = [
            {"project_id": "P1", "sector_code": "Water", "project_year": 2024},
            {"project_id": "P2", "sector_code": "Water", "project_year": 2024},
        ]
        documents = [
            {"document_id": "D1", "project_id": "P1", "sector_code": "Water", "document_year": 2018},
            {"document_id": "D2", "project_id": "OTHER", "sector_code": "water", "document_year": 2023},
            {"document_id": "D3", "project_id": "OTHER", "sector_code": "Energy", "document_year": 2024},
        ]
        matches = resolve_entities(projects, documents, temporal_window_years=2)
        self.assertEqual(
            [(item.project_id, item.document_id, item.method) for item in matches],
            [("P1", "D1", "project_id"), ("P2", "D2", "sector_temporal")],
        )
        self.assertEqual(matches[0].confidence, 1.0)

    def test_risk_baseline_is_marked_as_heuristic(self) -> None:
        risk = heuristic_risk(
            {"planned_days": 100, "actual_days": 140, "amendment_count": 2, "outcome": "Rebid"}
        )
        self.assertEqual(risk["risk_level"], "HIGH")
        self.assertEqual(risk["method"], "transparent_heuristic_not_a_trained_prediction")
        self.assertGreaterEqual(risk["risk_score"], 0.65)

    def test_vector_retrieval_is_ranked_and_thresholded(self) -> None:
        first = EvidenceChunk(
            document_id="D1", chunk_id="C1", chunk_text="one", section_title="s",
            document_type="ICR", embedding=(1.0, 0.0),
        )
        second = EvidenceChunk(
            document_id="D2", chunk_id="C2", chunk_text="two", section_title="s",
            document_type="ISR", embedding=(0.0, 1.0),
        )
        results = retrieve_by_embedding([1.0, 0.0], [second, first], minimum_score=0.1)
        self.assertEqual([item[0].document_id for item in results], ["D1"])
        self.assertAlmostEqual(results[0][1], 1.0)

    def test_citation_guard_rejects_missing_and_unretrieved_sources(self) -> None:
        claims = [
            {"text": "unsupported", "citations": []},
            {"text": "fabricated source", "citations": ["[NOT-RETRIEVED]"]},
        ]
        errors = validate_citations(claims, {"[VALID]"})
        self.assertEqual(len(errors), 2)

    def test_conflicting_same_period_claim_is_escalated(self) -> None:
        outcome = resolve_conflicting_evidence(
            [
                {"document_type": "ICR", "project_id": "P1", "claim_key": "rating_2023",
                 "claim_value": "satisfactory", "citation_id": "[ICR]"},
                {"document_type": "ISR", "project_id": "P1", "claim_key": "rating_2023",
                 "claim_value": "moderately satisfactory", "citation_id": "[ISR]"},
            ]
        )
        self.assertEqual(outcome["status"], "CONFLICTING_EVIDENCE")
        self.assertEqual(
            outcome["conflicts"][0]["resolution"],
            "preserve_both_and_escalate_for_human_review",
        )

    def test_world_bank_governance_is_joined_by_country_and_nearest_year(self) -> None:
        projects = [
            {"project_id": "P1", "country_code": "IND", "project_year": 2024},
            {"project_id": "P2", "country_code": "USA", "project_year": 2024},
        ]
        indicators = [
            {"country_code": "IND", "indicator_code": "GE.EST", "year": 2023, "value": 0.5},
            {"country_code": "IND", "indicator_code": "GE.EST", "year": 2025, "value": 0.7},
        ]
        result = enrich_with_governance(projects, indicators)
        self.assertEqual(result[0]["governance_effectiveness"], 0.5)
        self.assertEqual(result[0]["governance_effectiveness_source_year"], 2023)
        self.assertIsNone(result[1]["governance_effectiveness"])

    def test_manifest_document_parser_preserves_claim_annotations(self) -> None:
        root = Path(__file__).parent.parent / "examples"
        manifest = read_document_manifest(root / "document_manifest.json")
        chunks = parse_document(
            root / "documents" / "illustrative_icr.txt",
            manifest["illustrative_icr.txt"],
            max_tokens=40,
            overlap_tokens=5,
        )
        rating = next(chunk for chunk in chunks if chunk.claim_key)
        self.assertEqual(rating.claim_key, "implementation_rating_2023")
        self.assertEqual(rating.claim_value, "satisfactory")
        self.assertEqual(rating.effective_date, "2025-03-01")
        self.assertTrue(rating.citation_id.endswith(f"chunk:{rating.chunk_id}]"))

    def test_table_chunks_keep_rows_and_repeat_headers(self) -> None:
        chunks = _chunk_table(
            "Table 1",
            "Supplier | Amount\nA | 100\nB | 200\nC | 300",
            document_id="D1",
            document_type="ICR",
            project_id="P1",
            sector_code="Water",
            document_year=2024,
            source_uri="fixture.pdf",
            confidence_tier=1,
            max_tokens=3,
            effective_date=None,
        )
        self.assertTrue(all(chunk.chunk_type == "table" for chunk in chunks))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all("Supplier | Amount" in chunk.chunk_text for chunk in chunks))
        self.assertTrue(any("A | 100" in chunk.chunk_text for chunk in chunks))
        self.assertTrue(any("B | 200" in chunk.chunk_text for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
