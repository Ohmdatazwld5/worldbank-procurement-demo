from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from worldbank_demo_backend.core import EvidenceChunk
from worldbank_demo_backend.demo_overview import build_demo_overview

from worldbank_demo_backend.governance import Principal


class DemoOverviewTests(unittest.TestCase):
    def test_overview_reports_loaded_fixtures_and_unconfigured_services(self) -> None:
        records = [
            {
                "project_id": "P005",
                "sector_code": "Water",
                "project_year": 2024,
                "source_uri": "examples/procurement_demo.csv",
            }
        ]
        chunks = [
            EvidenceChunk(
                document_id="SYN-ICR-001",
                chunk_id="chunk-1",
                chunk_text="fictional evidence",
                section_title="Section 1",
                document_type="ICR",
                project_id="P005",
                sector_code="Water",
                document_year=2024,
                source_uri="examples/illustrative_icr.txt",
            )
        ]
        service = SimpleNamespace(
            records=records,
            chunks=chunks,
            workflow=SimpleNamespace(automl_models={}, model_uris={}),
            delta_root=None,
        )

        with patch.dict(os.environ, {}, clear=True):
            overview = build_demo_overview(service)

        self.assertEqual(overview["data"]["classification"], "fictional_demo_fixtures")
        self.assertEqual(overview["data"]["procurement_records"], 1)
        self.assertEqual(overview["data"]["documents"], 1)
        self.assertFalse(overview["data"]["ai_procurement_workbook_connected"])
        self.assertEqual(overview["knowledge_model"]["exact_project_links"], 1)
        self.assertEqual(overview["analytics"]["risk_method"], "transparent_heuristic_only")
        self.assertFalse(overview["analytics"]["registered_model_predictions_available"])
        self.assertFalse(overview["operations"]["airflow_scheduler_running_with_this_render_service"])
        self.assertFalse(overview["operations"]["otel_export_configured"])
        self.assertTrue(overview["guardrails"]["citation_validation"])

    def test_overview_counts_feedback_without_exposing_feedback_contents(self) -> None:
        service = SimpleNamespace(
            records=[],
            chunks=[],
            workflow=SimpleNamespace(automl_models={}, model_uris={}),
            delta_root=None,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            feedback_path = Path(temporary_directory) / "feedback.jsonl"
            feedback_path.write_text('{"comment":"private reviewer note"}\n\n{}\n', encoding="utf-8")
            overview = build_demo_overview(
                service,
                {"FEEDBACK_JSONL_PATH": str(feedback_path)},
            )

        self.assertEqual(overview["operations"]["feedback_entries_recorded"], 2)
        self.assertNotIn("private reviewer note", str(overview))

    def test_api_overview_route_requires_analytics_authorization(self) -> None:
        from worldbank_demo_backend.api import demo_overview

        service = SimpleNamespace(
            records=[],
            chunks=[],
            workflow=SimpleNamespace(automl_models={}, model_uris={}),
            delta_root=None,
        )
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(service=service)))
        principal = Principal("viewer", frozenset({"viewer"}), frozenset())
        with (
            patch("worldbank_demo_backend.api._principal", return_value=principal),
            patch.dict(os.environ, {}, clear=True),
        ):
            with self.assertRaises(HTTPException) as raised:
                demo_overview(request, credentials=None)

        self.assertEqual(raised.exception.status_code, 403)


if __name__ == "__main__":
    unittest.main()
