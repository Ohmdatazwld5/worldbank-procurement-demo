from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from worldbank_demo_backend.ai_procurement import (
    aggregate_annual_counts,
    aggregate_source_year_counts,
    backtest_source_baselines,
    load_workbook_rows,
    parse_notice_date,
    prepare_contract_rows,
    source_coverage_summary,
    train_ai_procurement_forecasts,
    _evaluate_flaml_source,
)


class AiProcurementTests(unittest.TestCase):
    def test_notice_date_uses_confirmed_us_format(self) -> None:
        self.assertEqual(parse_notice_date("11/13/2025 0:00"), date(2025, 11, 13))
        self.assertEqual(parse_notice_date("11/10/2025 0:00"), date(2025, 11, 10))
        self.assertIsNone(parse_notice_date(""))

    def test_unrecognized_date_is_reported_not_guessed(self) -> None:
        rows, report = prepare_contract_rows(
            [
                {
                    "notice_id": "N-1",
                    "notice_date": "not-a-date",
                    "source": "ted",
                    "income_group": "High income",
                    "tool_category": "Other",
                }
            ]
        )
        self.assertEqual(rows, [])
        self.assertEqual(report["invalid_notice_date_rows_excluded"], 1)

    def test_missing_dates_are_excluded_and_unknown_categories_retained(self) -> None:
        rows, report = prepare_contract_rows(
            [
                {
                    "notice_id": "N-1",
                    "notice_date": "2024-01-02",
                    "source": "world_bank",
                    "income_group": "Low income",
                    "tool_category": None,
                },
                {
                    "notice_id": "N-2",
                    "notice_date": None,
                    "source": "ted",
                    "income_group": "High income",
                    "tool_category": "Other",
                },
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tool_category"], "Unknown")
        self.assertEqual(report["missing_notice_date_rows_excluded"], 1)
        self.assertEqual(
            aggregate_annual_counts(rows),
            [
                {
                    "year": 2024,
                    "source": "world_bank",
                    "income_group": "Low income",
                    "tool_category": "Unknown",
                    "contract_count": 1,
                }
            ],
        )

    def test_duplicate_notice_id_is_rejected(self) -> None:
        row = {
            "notice_id": "N-1",
            "notice_date": "2024-01-02",
            "source": "ted",
            "income_group": "High income",
            "tool_category": "Other",
        }
        with self.assertRaisesRegex(ValueError, "Duplicate notice_id"):
            prepare_contract_rows([row, row])

    def test_workbook_loader_requires_expected_columns(self) -> None:
        from openpyxl import Workbook

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "missing-columns.xlsx"
            workbook = Workbook()
            worksheet = workbook.active
            worksheet.append(["notice_id", "notice_date"])
            worksheet.append(["N-1", "2024-01-02"])
            workbook.save(path)
            workbook.close()
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_workbook_rows(path)

    def test_source_year_coverage_keeps_gaps_unknown(self) -> None:
        summary = source_coverage_summary(
            [
                {"year": 2020, "source": "a", "contract_count": 3},
                {"year": 2022, "source": "a", "contract_count": 4},
            ]
        )
        self.assertEqual(summary[0]["missing_years_within_range"], [2021])
        self.assertIn("not zero", summary[0]["coverage_note"])

    def test_baselines_use_rolling_chronological_holdout(self) -> None:
        source_years = [
            {"year": year, "source": "a", "contract_count": year - 2009}
            for year in range(2010, 2020)
        ]
        result = backtest_source_baselines(source_years, holdout_years=2)
        evaluated = result["results"][0]
        self.assertEqual(evaluated["status"], "evaluated")
        self.assertEqual(evaluated["training_years"], list(range(2010, 2018)))
        self.assertEqual(evaluated["holdout_years"], [2018, 2019])
        self.assertFalse(result["missing_source_years_treated_as_zero"])

    def test_short_or_gapped_source_series_is_not_backtested(self) -> None:
        sparse = [
            {"year": 2024, "source": "eu_ted", "contract_count": 1},
            {"year": 2025, "source": "eu_ted", "contract_count": 2},
        ]
        result = backtest_source_baselines(sparse)
        self.assertEqual(result["results"][0]["status"], "not_evaluated")
        self.assertEqual(result["results"][0]["consecutive_years"], 2)

    def test_source_year_aggregation(self) -> None:
        records = [
            {"year": 2024, "source": "ted"},
            {"year": 2024, "source": "ted"},
            {"year": 2025, "source": "open_contract"},
        ]
        self.assertEqual(
            aggregate_source_year_counts(records),
            [
                {"year": 2024, "source": "ted", "contract_count": 2},
                {"year": 2025, "source": "open_contract", "contract_count": 1},
            ],
        )

    def test_baselines_include_fixed_origin_scores_for_flaml_comparison(self) -> None:
        source_years = [
            {"year": year, "source": "a", "contract_count": year - 2009}
            for year in range(2010, 2020)
        ]
        result = backtest_source_baselines(source_years, holdout_years=2)
        evaluated = result["results"][0]
        self.assertIn("fixed_origin_metrics", evaluated)
        self.assertEqual(
            evaluated["fixed_origin_observations"][0]["last_year_baseline"],
            8,
        )
        self.assertEqual(
            evaluated["fixed_origin_observations"][1]["last_year_baseline"],
            8,
        )

    def test_flaml_evaluation_uses_source_specific_contiguous_holdout(self) -> None:
        class FakeAutoML:
            fit_calls = []

            def __init__(self) -> None:
                self.fit_arguments = None
                self.best_estimator = "fake"

            def fit(self, **kwargs) -> None:
                self.fit_arguments = kwargs
                self.fit_calls.append(kwargs)

            def predict(self, timestamps):
                import numpy as np

                return np.repeat(8.0, len(timestamps))

            def pickle(self, path: str) -> None:
                Path(path).write_bytes(b"fake-model")

        years = list(range(2010, 2020))
        yearly_counts = {year: year - 2009 for year in years}
        result = _evaluate_flaml_source(
            "open_contract",
            yearly_counts,
            years[-2:],
            FakeAutoML,
            time_budget_seconds=1,
        )
        self.assertEqual(result["rolling_origin_count"], 2)
        self.assertEqual(result["rolling_origin_metrics"]["flaml"]["mae"], 1.5)
        self.assertEqual(result["next_unobserved_year_estimate"]["year"], 2020)
        self.assertEqual(
            result["next_unobserved_year_estimate"]["forecast_origin_year"], 2019
        )
        self.assertEqual(
            result["next_unobserved_year_estimate"]["predicted_count"], 8.0
        )
        self.assertFalse(
            result["next_unobserved_year_estimate"]["actual_available_in_dataset"]
        )
        self.assertEqual(
            result["next_unobserved_year_estimate"]["status"],
            "historical_year_not_validated_against_actuals",
        )
        self.assertEqual(result["provenance_status"], "unverified")
        self.assertEqual(result["decision_use"], "exploratory_only_not_for_decisions")

    def test_flaml_mlflow_run_excludes_ted_and_never_registers(self) -> None:
        class FakeAutoML:
            fit_calls = []

            def __init__(self) -> None:
                self.best_estimator = "fake"

            def fit(self, **kwargs) -> None:
                self.fit_arguments = kwargs
                self.fit_calls.append(kwargs)

            def predict(self, timestamps):
                import numpy as np

                return np.repeat(10.0, len(timestamps))

            def pickle(self, path: str) -> None:
                Path(path).write_bytes(b"fake-model")

        class FakeMlflow(ModuleType):
            def __init__(self) -> None:
                super().__init__("mlflow")
                self.tags = {}
                self.metrics = {}
                self.params = {}
                self.artifacts = []
                self.register_model = lambda *args, **kwargs: self.fail_registration()

            def set_experiment(self, name):
                self.experiment_name = name

            def start_run(self, run_name):
                self.run_name = run_name
                return self

            def __enter__(self):
                self.info = SimpleNamespace(run_id="test-run")
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def set_tags(self, tags):
                self.tags.update(tags)

            def log_params(self, params):
                self.params.update(params)

            def log_param(self, key, value):
                self.params[key] = value

            def log_metrics(self, metrics):
                self.metrics.update(metrics)

            def log_artifact(self, path, artifact_path=None):
                self.artifacts.append((path, artifact_path))

            def fail_registration(self):
                raise AssertionError("Exploratory models must not be registered.")

        fake_mlflow = FakeMlflow()
        fake_flaml = ModuleType("flaml")
        fake_flaml.AutoML = FakeAutoML
        series = [
            {"year": year, "source": source, "contract_count": year - 1999}
            for source in ("open_contract", "world_bank")
            for year in range(2010, 2020)
        ]
        baselines = backtest_source_baselines(series, holdout_years=2)
        with patch.dict(
            "sys.modules",
            {"flaml": fake_flaml, "mlflow": fake_mlflow},
        ):
            result = train_ai_procurement_forecasts(
                series
                + [
                    {"year": 2025, "source": "eu_ted", "contract_count": 10}
                ],
                baselines,
                time_budget_seconds=1,
            )
        self.assertEqual(set(result["sources"]), {"open_contract", "world_bank"})
        self.assertEqual(result["model_registered"], False)
        self.assertEqual(result["mlflow_run_id"], "test-run")
        self.assertEqual(fake_mlflow.tags["provenance_status"], "unverified")
        self.assertEqual(
            fake_mlflow.tags["model_intended_use"],
            "exploratory_only_not_for_decisions",
        )
        self.assertEqual(fake_mlflow.tags["excluded_sources"], "eu_ted")
        self.assertTrue(FakeAutoML.fit_calls)
        self.assertTrue(
            all(call["mlflow_logging"] is False for call in FakeAutoML.fit_calls)
        )


if __name__ == "__main__":
    unittest.main()
