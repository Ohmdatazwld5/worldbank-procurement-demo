"""Validate AI procurement contracts and export annual contract counts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


REQUIRED_COLUMNS = {
    "notice_id",
    "notice_date",
    "source",
    "income_group",
    "tool_category",
}
COUNT_COLUMNS = ("year", "source", "income_group", "tool_category", "contract_count")
SOURCE_YEAR_COLUMNS = ("year", "source", "contract_count")


def parse_notice_date(value: Any) -> date | None:
    """Parse the workbook's confirmed US month/day/year dates without guessing."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError(f"Unsupported notice_date value: {value!r}")

    text = value.strip()
    for date_format in (
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(text, date_format).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognized notice_date format: {value!r}")


def load_workbook_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read the first worksheet as dictionaries, rejecting malformed headers."""
    from openpyxl import load_workbook

    source = Path(path)
    workbook = load_workbook(source, read_only=True, data_only=True)
    try:
        worksheet = workbook.active
        rows = worksheet.iter_rows(values_only=True)
        try:
            header_values = next(rows)
        except StopIteration:
            raise ValueError(f"Workbook contains no header row: {source}") from None
        headers = [
            str(value).strip() if value is not None else ""
            for value in header_values
        ]
        if not all(headers) or len(headers) != len(set(headers)):
            raise ValueError("Workbook headers must be non-empty and unique.")
        missing = REQUIRED_COLUMNS.difference(headers)
        if missing:
            raise ValueError(f"Workbook is missing required columns: {sorted(missing)}")
        return [
            dict(zip(headers, values))
            for values in rows
            if any(value is not None for value in values)
        ]
    finally:
        workbook.close()


def _label(value: Any) -> str:
    if value is None:
        return "Unknown"
    text = str(value).strip()
    return text if text else "Unknown"


def prepare_contract_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate IDs/dates and return dated rows plus a quality summary."""
    prepared: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    missing_counts = Counter()
    invalid_dates: list[dict[str, str]] = []
    model_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    total_rows = 0
    missing_notice_dates = 0

    for row_number, row in enumerate(rows, start=2):
        total_rows += 1
        notice_id = _label(row.get("notice_id"))
        if notice_id == "Unknown":
            raise ValueError(f"Missing notice_id at workbook row {row_number}.")
        if notice_id in seen_ids:
            raise ValueError(f"Duplicate notice_id {notice_id!r} at workbook row {row_number}.")
        seen_ids.add(notice_id)

        source = _label(row.get("source"))
        income_group = _label(row.get("income_group"))
        tool_category = _label(row.get("tool_category"))
        model = _label(row.get("model"))
        source_counts[source] += 1
        model_counts[model] += 1
        for column, value in (
            ("source", source),
            ("income_group", income_group),
            ("tool_category", tool_category),
        ):
            if value == "Unknown":
                missing_counts[column] += 1

        raw_date = row.get("notice_date")
        try:
            notice_date = parse_notice_date(raw_date)
        except ValueError:
            invalid_dates.append(
                {"notice_id": notice_id, "notice_date": str(raw_date)}
            )
            continue
        if notice_date is None:
            missing_notice_dates += 1
            continue
        prepared.append(
            {
                "notice_id": notice_id,
                "year": notice_date.year,
                "source": source,
                "income_group": income_group,
                "tool_category": tool_category,
            }
        )

    summary = {
        "total_rows": total_rows,
        "unique_notice_ids": len(seen_ids),
        "dated_rows_included": len(prepared),
        "missing_notice_date_rows_excluded": missing_notice_dates,
        "invalid_notice_date_rows_excluded": len(invalid_dates),
        "invalid_notice_date_examples": invalid_dates[:20],
        "unknown_dimension_counts": dict(sorted(missing_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "model_field_counts": dict(sorted(model_counts.items())),
        "date_format": "US month/day/year; ISO year-month-day also accepted",
        "notice_year_range": (
            [min(row["year"] for row in prepared), max(row["year"] for row in prepared)]
            if prepared
            else None
        ),
    }
    return prepared, summary


def aggregate_annual_counts(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Count unique dated notices by year, source, income group, and category."""
    counts: Counter[tuple[int, str, str, str]] = Counter()
    for row in rows:
        key = (
            int(row["year"]),
            str(row["source"]),
            str(row["income_group"]),
            str(row["tool_category"]),
        )
        counts[key] += 1
    return [
        {
            "year": year,
            "source": source,
            "income_group": income_group,
            "tool_category": tool_category,
            "contract_count": count,
        }
        for (year, source, income_group, tool_category), count in sorted(counts.items())
    ]


def export_annual_counts(
    counts: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COUNT_COLUMNS)
        writer.writeheader()
        writer.writerows(counts)


def aggregate_source_year_counts(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate observed dated notices by source and year."""
    counts: Counter[tuple[int, str]] = Counter()
    for row in rows:
        counts[(int(row["year"]), str(row["source"]))] += 1
    return [
        {"year": year, "source": source, "contract_count": count}
        for (year, source), count in sorted(counts.items())
    ]


def export_source_year_counts(
    counts: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SOURCE_YEAR_COLUMNS)
        writer.writeheader()
        writer.writerows(counts)


def source_coverage_summary(
    source_year_counts: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Describe observed years and gaps without interpreting gaps as zero counts."""
    years_by_source: dict[str, set[int]] = {}
    for row in source_year_counts:
        years_by_source.setdefault(str(row["source"]), set()).add(int(row["year"]))

    summary = []
    for source, years in sorted(years_by_source.items()):
        first_year, last_year = min(years), max(years)
        missing_years = [
            year for year in range(first_year, last_year + 1) if year not in years
        ]
        summary.append(
            {
                "source": source,
                "first_observed_year": first_year,
                "last_observed_year": last_year,
                "observed_years": len(years),
                "missing_years_within_range": missing_years,
                "coverage_note": (
                    "Missing source-years are unknown coverage, not zero contracts."
                    if missing_years
                    else "Every year in the observed range has at least one dated record; completeness is not verified."
                ),
            }
        )
    return summary


def _mae(actual: list[int], predicted: list[float]) -> float:
    return sum(abs(observed - estimate) for observed, estimate in zip(actual, predicted)) / len(actual)


def _wape(actual: list[int], predicted: list[float]) -> float | None:
    denominator = sum(actual)
    if denominator == 0:
        return None
    return sum(abs(observed - estimate) for observed, estimate in zip(actual, predicted)) / denominator


def _fixed_origin_baselines(
    yearly_counts: Mapping[int, int],
    holdout_years: list[int],
) -> dict[str, Any]:
    first_holdout_year = holdout_years[0]
    last_observed = float(yearly_counts[first_holdout_year - 1])
    trailing_mean = sum(
        yearly_counts[first_holdout_year - offset] for offset in (1, 2, 3)
    ) / 3
    actual = [yearly_counts[year] for year in holdout_years]
    last_year_predictions = [last_observed] * len(holdout_years)
    trailing_mean_predictions = [trailing_mean] * len(holdout_years)
    observations = [
        {
            "year": year,
            "actual": yearly_counts[year],
            "last_year_baseline": last_observed,
            "trailing_3_year_mean_baseline": round(trailing_mean, 3),
        }
        for year in holdout_years
    ]
    return {
        "observations": observations,
        "metrics": {
            "last_year_baseline": {
                "mae": round(_mae(actual, last_year_predictions), 3),
                "wape": round(_wape(actual, last_year_predictions), 4),
            },
            "trailing_3_year_mean_baseline": {
                "mae": round(_mae(actual, trailing_mean_predictions), 3),
                "wape": round(_wape(actual, trailing_mean_predictions), 4),
            },
        },
    }


def backtest_source_baselines(
    source_year_counts: Iterable[Mapping[str, Any]],
    *,
    holdout_years: int = 5,
    minimum_history_years: int = 8,
) -> dict[str, Any]:
    """Backtest naïve count baselines on each source's latest contiguous years."""
    if holdout_years < 1:
        raise ValueError("holdout_years must be at least one.")
    if minimum_history_years <= holdout_years:
        raise ValueError("minimum_history_years must exceed holdout_years.")

    counts_by_source: dict[str, dict[int, int]] = {}
    for row in source_year_counts:
        source = str(row["source"])
        year = int(row["year"])
        count = int(row["contract_count"])
        if count <= 0:
            raise ValueError("Source-year counts must be positive observed counts.")
        if year in counts_by_source.setdefault(source, {}):
            raise ValueError(f"Duplicate source-year count for {source!r}, {year}.")
        counts_by_source[source][year] = count

    results = []
    for source, yearly_counts in sorted(counts_by_source.items()):
        latest_year = max(yearly_counts)
        contiguous_years = [latest_year]
        while contiguous_years[-1] - 1 in yearly_counts:
            contiguous_years.append(contiguous_years[-1] - 1)
        contiguous_years.reverse()
        if len(contiguous_years) <= minimum_history_years:
            results.append(
                {
                    "source": source,
                    "status": "not_evaluated",
                    "reason": (
                        f"Only {len(contiguous_years)} consecutive observed years "
                        f"through {latest_year}; more than {minimum_history_years} "
                        "are required to retain the minimum training history and at "
                        "least one holdout. Missing years are not treated as zero."
                    ),
                    "consecutive_years": len(contiguous_years),
                }
            )
            continue

        test_size = min(
            holdout_years,
            len(contiguous_years) - minimum_history_years,
        )
        test_years = contiguous_years[-test_size:]
        actual = [yearly_counts[year] for year in test_years]
        fixed_origin = _fixed_origin_baselines(yearly_counts, test_years)
        last_year_predictions = [
            float(yearly_counts[year - 1]) for year in test_years
        ]
        trailing_mean_predictions = [
            sum(yearly_counts[year - offset] for offset in (1, 2, 3)) / 3
            for year in test_years
        ]
        results.append(
            {
                "source": source,
                "status": "evaluated",
                "method": "rolling one-year-ahead evaluation on latest contiguous observed years",
                "consecutive_year_range": [
                    contiguous_years[0],
                    contiguous_years[-1],
                ],
                "training_years": contiguous_years[:-test_size],
                "holdout_years": test_years,
                "observations": [
                    {
                        "year": year,
                        "actual": yearly_counts[year],
                        "last_year_baseline": yearly_counts[year - 1],
                        "trailing_3_year_mean_baseline": round(
                            sum(yearly_counts[year - offset] for offset in (1, 2, 3))
                            / 3,
                            3,
                        ),
                    }
                    for year in test_years
                ],
                "fixed_origin_observations": fixed_origin["observations"],
                "metrics": {
                    "last_year_baseline": {
                        "mae": round(_mae(actual, last_year_predictions), 3),
                        "wape": round(_wape(actual, last_year_predictions), 4),
                    },
                    "trailing_3_year_mean_baseline": {
                        "mae": round(_mae(actual, trailing_mean_predictions), 3),
                        "wape": round(_wape(actual, trailing_mean_predictions), 4),
                    },
                },
                "fixed_origin_metrics": fixed_origin["metrics"],
                "coverage_warning": (
                    "Counts measure records present in this workbook, not necessarily "
                    "all contracts issued by the source. Performance is descriptive, "
                    "not evidence of future forecast accuracy."
                ),
            }
        )
    return {
        "target": "annual observed AI contract records per source",
        "holdout_years_requested": holdout_years,
        "backtest_method": "expanding_window_rolling_origin_one_year_ahead",
        "missing_source_years_treated_as_zero": False,
        "results": results,
    }


def _fit_time_series_model(
    automl_class: Any,
    years: list[int],
    counts: list[int],
    *,
    period: int,
    time_budget_seconds: int,
) -> Any:
    import numpy as np

    if len(years) != len(counts) or not years:
        raise ValueError("A non-empty, aligned yearly series is required.")
    if period < 1:
        raise ValueError("Forecast period must be at least one.")
    timestamps = np.asarray(
        [np.datetime64(f"{year}-01-01") for year in years],
        dtype="datetime64[D]",
    )
    automl = automl_class()
    automl.fit(
        X_train=timestamps,
        y_train=np.asarray(counts, dtype=float),
        task="ts_forecast",
        estimator_list=["lgbm", "rf", "extra_tree"],
        period=period,
        time_budget=max(1, int(time_budget_seconds)),
        metric="mae",
        eval_method="holdout",
        mlflow_logging=False,
        verbose=0,
    )
    return automl


def _evaluate_flaml_source(
    source: str,
    yearly_counts: Mapping[int, int],
    evaluation_years: list[int],
    automl_class: Any,
    *,
    time_budget_seconds: int,
) -> dict[str, Any]:
    import numpy as np

    observed_years = sorted(int(year) for year in yearly_counts)
    if not observed_years or observed_years != list(
        range(observed_years[0], observed_years[-1] + 1)
    ):
        raise ValueError(f"{source} history must be a non-empty contiguous yearly series.")
    if not evaluation_years:
        raise ValueError(f"{source} has no eligible rolling-origin years.")
    holdout_years = sorted(set(int(year) for year in evaluation_years))
    if len(holdout_years) != len(evaluation_years):
        raise ValueError(f"{source} rolling-origin years must be unique.")
    if any(year not in yearly_counts for year in holdout_years):
        raise ValueError(f"{source} is missing a year in its evaluated series.")
    if holdout_years != [year for year in observed_years if year in holdout_years]:
        raise ValueError(f"{source} rolling-origin years are out of chronological order.")

    observations = []
    actual_values: list[int] = []
    flaml_predictions: list[float] = []
    last_year_predictions: list[float] = []
    trailing_mean_predictions: list[float] = []
    estimators: list[str] = []
    for holdout_year in holdout_years:
        training_years = [year for year in observed_years if year < holdout_year]
        if len(training_years) < 8:
            raise ValueError(
                f"{source} has only {len(training_years)} training years before "
                f"rolling origin {holdout_year}; at least eight are required."
            )
        if training_years[-1] + 1 != holdout_year:
            raise ValueError(f"{source} history is not contiguous before {holdout_year}.")

        model = _fit_time_series_model(
            automl_class,
            training_years,
            [int(yearly_counts[year]) for year in training_years],
            period=1,
            time_budget_seconds=time_budget_seconds,
        )
        timestamp = np.asarray(
            [np.datetime64(f"{holdout_year}-01-01")],
            dtype="datetime64[D]",
        )
        prediction = np.asarray(model.predict(timestamp), dtype=float).reshape(-1)
        if len(prediction) != 1 or not np.isfinite(prediction[0]):
            raise RuntimeError(
                f"FLAML returned an invalid prediction for {source} in {holdout_year}."
            )
        prediction_value = max(0.0, float(prediction[0]))
        actual_value = int(yearly_counts[holdout_year])
        last_year_value = float(yearly_counts[holdout_year - 1])
        trailing_mean_value = sum(
            yearly_counts[holdout_year - offset] for offset in (1, 2, 3)
        ) / 3
        observations.append(
            {
                "year": holdout_year,
                "training_through_year": training_years[-1],
                "actual": actual_value,
                "flaml_prediction": round(prediction_value, 3),
                "last_year_baseline": round(last_year_value, 3),
                "trailing_3_year_mean_baseline": round(trailing_mean_value, 3),
                "best_estimator": str(getattr(model, "best_estimator", "unknown")),
            }
        )
        actual_values.append(actual_value)
        flaml_predictions.append(prediction_value)
        last_year_predictions.append(last_year_value)
        trailing_mean_predictions.append(trailing_mean_value)
        estimators.append(str(getattr(model, "best_estimator", "unknown")))

    final_years = observed_years
    final_counts = [int(yearly_counts[year]) for year in final_years]
    final_model = _fit_time_series_model(
        automl_class,
        final_years,
        final_counts,
        period=1,
        time_budget_seconds=time_budget_seconds,
    )
    forecast_year = final_years[-1] + 1
    forecast_timestamp = np.asarray(
        [np.datetime64(f"{forecast_year}-01-01")],
        dtype="datetime64[D]",
    )
    next_prediction = np.asarray(
        final_model.predict(forecast_timestamp),
        dtype=float,
    ).reshape(-1)
    if len(next_prediction) != 1 or not np.isfinite(next_prediction[0]):
        raise RuntimeError(f"FLAML returned an invalid forecast for {source}.")
    forecast_count = max(0.0, float(next_prediction[0]))
    current_year = date.today().year
    if forecast_year < current_year:
        estimate_status = "historical_year_not_validated_against_actuals"
    elif forecast_year == current_year:
        estimate_status = "current_calendar_year_estimate_not_full_year_validated"
    else:
        estimate_status = "future_estimate_unvalidated"
    return {
        "source": source,
        "status": "exploratory",
        "backtest_method": "expanding_window_rolling_origin_one_year_ahead",
        "rolling_origin_count": len(holdout_years),
        "holdout_years": holdout_years,
        "rolling_origin_observations": observations,
        "rolling_origin_metrics": {
            "flaml": {
                "mae": round(_mae(actual_values, flaml_predictions), 3),
                "wape": round(_wape(actual_values, flaml_predictions), 4),
            },
            "last_year_baseline": {
                "mae": round(_mae(actual_values, last_year_predictions), 3),
                "wape": round(_wape(actual_values, last_year_predictions), 4),
            },
            "trailing_3_year_mean_baseline": {
                "mae": round(_mae(actual_values, trailing_mean_predictions), 3),
                "wape": round(_wape(actual_values, trailing_mean_predictions), 4),
            },
        },
        "latest_observed_year": final_years[-1],
        "training_years": final_years,
        "next_unobserved_year_estimate": {
            "year": forecast_year,
            "forecast_origin_year": final_years[-1],
            "predicted_count": round(forecast_count, 3),
            "status": estimate_status,
            "actual_available_in_dataset": forecast_year in yearly_counts,
            "description": "Exploratory one-step estimate from the latest year in this dataset, not a validated full-year forecast.",
            "last_year_baseline": int(yearly_counts[final_years[-1]]),
            "trailing_3_year_mean_baseline": round(
                sum(yearly_counts[final_years[-offset]] for offset in (1, 2, 3))
                / 3,
                3,
            ),
        },
        "best_estimator": str(getattr(final_model, "best_estimator", "unknown")),
        "rolling_origin_estimators": estimators,
        "training_rows": len(final_years),
        "provenance_status": "unverified",
        "decision_use": "exploratory_only_not_for_decisions",
        "_final_model": final_model,
    }


def train_ai_procurement_forecasts(
    source_year_counts: Iterable[Mapping[str, Any]],
    baseline_report: Mapping[str, Any],
    *,
    experiment_name: str = "ai-procurement-exploratory-forecast",
    time_budget_seconds: int = 10,
) -> dict[str, Any]:
    """Train exploratory FLAML forecasts for the two sources with usable history.

    TED is intentionally excluded because only one year is observed. Results are
    logged to MLflow with provenance and decision-use guardrail tags and are never
    registered.
    """
    if time_budget_seconds < 1:
        raise ValueError("time_budget_seconds must be at least one.")
    from flaml import AutoML
    import mlflow

    source_counts: dict[str, dict[int, int]] = {}
    for row in source_year_counts:
        source = str(row["source"])
        if source not in {"open_contract", "world_bank"}:
            continue
        year = int(row["year"])
        count = int(row["contract_count"])
        if count <= 0:
            raise ValueError("Source-year counts must be positive observed counts.")
        if year in source_counts.setdefault(source, {}):
            raise ValueError(f"Duplicate source-year count for {source!r}, {year}.")
        source_counts[source][year] = count

    baseline_results = {
        item["source"]: item
        for item in baseline_report.get("results", [])
        if item.get("status") == "evaluated"
        and item.get("source") in {"open_contract", "world_bank"}
    }
    eligible_sources = sorted(set(source_counts).intersection(baseline_results))
    if not eligible_sources:
        raise ValueError(
            "No eligible Open Contract or World Bank source has a completed baseline backtest."
        )

    mlflow.set_experiment(experiment_name)
    results: dict[str, Any] = {
        "status": "exploratory",
        "experiment_name": experiment_name,
        "provenance_status": "unverified",
        "decision_use": "exploratory_only_not_for_decisions",
        "model_registered": False,
        "excluded_sources": {"eu_ted": "Only one observed year; insufficient history."},
        "sources": {},
    }
    with mlflow.start_run(run_name="flaml-source-separated-ai-contract-counts") as run:
        mlflow.set_tags(
            {
                "dataset": "worldbank_ai_procurement_contracts",
                "provenance_status": "unverified",
                "model_intended_use": "exploratory_only_not_for_decisions",
                "model_registration": "prohibited_pending_validation",
                "included_sources": "open_contract,world_bank",
                "excluded_sources": "eu_ted",
                "forecast_target": "annual_observed_ai_contract_record_count",
            }
        )
        mlflow.log_params(
            {
                "task": "ts_forecast",
                "time_budget_seconds_per_fit": time_budget_seconds,
                "holdout_protocol": "expanding_window_rolling_origin_one_year_ahead",
                "rolling_origins": 5,
                "minimum_training_years_per_origin": 8,
                "automl_framework": "FLAML",
                "forecast_horizon_years": 1,
            }
        )
        with tempfile.TemporaryDirectory(prefix="ai_procurement_forecast_") as temporary:
            temporary_path = Path(temporary)
            for source in eligible_sources:
                result = _evaluate_flaml_source(
                    source,
                    source_counts[source],
                    baseline_results[source]["holdout_years"],
                    AutoML,
                    time_budget_seconds=time_budget_seconds,
                )
                final_model = result.pop("_final_model")
                results["sources"][source] = result
                mlflow.log_metrics(
                    {
                        f"{source}_flaml_rolling_mae": result["rolling_origin_metrics"]["flaml"]["mae"],
                        f"{source}_flaml_rolling_wape": result["rolling_origin_metrics"]["flaml"]["wape"],
                        f"{source}_last_year_baseline_rolling_mae": result["rolling_origin_metrics"]["last_year_baseline"]["mae"],
                        f"{source}_last_year_baseline_rolling_wape": result["rolling_origin_metrics"]["last_year_baseline"]["wape"],
                        f"{source}_trailing_mean_baseline_rolling_mae": result["rolling_origin_metrics"]["trailing_3_year_mean_baseline"]["mae"],
                        f"{source}_trailing_mean_baseline_rolling_wape": result["rolling_origin_metrics"]["trailing_3_year_mean_baseline"]["wape"],
                        f"{source}_next_unobserved_year_estimate": result["next_unobserved_year_estimate"]["predicted_count"],
                    }
                )
                mlflow.log_param(
                    f"{source}_training_rows",
                    result["training_rows"],
                )
                mlflow.log_param(
                    f"{source}_best_estimator",
                    result["best_estimator"],
                )
                artifact_path = temporary_path / f"{source}_flaml_automl.pkl"
                final_model.pickle(str(artifact_path))
                mlflow.log_artifact(str(artifact_path), artifact_path="models")

            forecast_report_path = temporary_path / "ai_procurement_flaml_forecasts.json"
            forecast_report_path.write_text(
                json.dumps(results, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(forecast_report_path), artifact_path="reports")
        results["mlflow_run_id"] = run.info.run_id
    return results


def run(
    workbook_path: str | Path,
    counts_path: str | Path,
    report_path: str | Path,
    source_year_path: str | Path,
    backtest_path: str | Path,
    forecast_path: str | Path | None = None,
    *,
    train_flaml: bool = False,
    flaml_time_budget_seconds: int = 10,
) -> dict[str, Any]:
    rows = load_workbook_rows(workbook_path)
    prepared, report = prepare_contract_rows(rows)
    counts = aggregate_annual_counts(prepared)
    source_year_counts = aggregate_source_year_counts(prepared)
    coverage = source_coverage_summary(source_year_counts)
    backtest = backtest_source_baselines(source_year_counts)
    export_annual_counts(counts, counts_path)
    export_source_year_counts(source_year_counts, source_year_path)
    report["annual_count_groups"] = len(counts)
    report["annual_count_total"] = sum(row["contract_count"] for row in counts)
    report["source_year_summary_file"] = str(source_year_path)
    report["source_coverage"] = coverage
    report["baseline_backtest"] = backtest
    report["baseline_backtest_file"] = str(backtest_path)
    report["counts_file"] = str(counts_path)
    report["source_file"] = str(workbook_path)

    if train_flaml:
        if forecast_path is None:
            raise ValueError("forecast_path is required when FLAML training is enabled.")
        forecast_report = train_ai_procurement_forecasts(
            source_year_counts,
            backtest,
            experiment_name=os.getenv(
                "MLFLOW_AI_PROCUREMENT_EXPERIMENT",
                "ai-procurement-exploratory-forecast",
            ),
            time_budget_seconds=flaml_time_budget_seconds,
        )
        report["exploratory_flaml_forecasts"] = forecast_report
        report["exploratory_flaml_forecast_file"] = str(forecast_path)
        forecast_output = Path(forecast_path)
        forecast_output.parent.mkdir(parents=True, exist_ok=True)
        forecast_output.write_text(
            json.dumps(forecast_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    backtest_output = Path(backtest_path)
    backtest_output.parent.mkdir(parents=True, exist_ok=True)
    backtest_output.write_text(
        json.dumps(backtest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workbook",
        default="examples/AI_Procurement_Contracts.xlsx",
        help="Path to the source Excel workbook.",
    )
    parser.add_argument(
        "--counts",
        default="data/processed/ai_procurement_annual_counts.csv",
        help="Output CSV path for annual grouped counts.",
    )
    parser.add_argument(
        "--report",
        default="data/processed/ai_procurement_validation.json",
        help="Output JSON path for validation statistics.",
    )
    parser.add_argument(
        "--source-year-summary",
        default="data/processed/ai_procurement_source_year_summary.csv",
        help="Output CSV path for observed source-year totals.",
    )
    parser.add_argument(
        "--backtest",
        default="data/processed/ai_procurement_baseline_backtest.json",
        help="Output JSON path for chronological baseline evaluation.",
    )
    parser.add_argument(
        "--train-flaml",
        action="store_true",
        help="Train source-specific exploratory FLAML forecasts and log them to MLflow.",
    )
    parser.add_argument(
        "--flaml-time-budget-seconds",
        type=int,
        default=int(os.getenv("FLAML_AI_PROCUREMENT_TIME_BUDGET_SECONDS", "10")),
        help="FLAML search budget per source model fit.",
    )
    parser.add_argument(
        "--forecasts",
        default="data/processed/ai_procurement_flaml_forecasts.json",
        help="Output JSON path for exploratory FLAML forecasts.",
    )
    args = parser.parse_args()
    report = run(
        args.workbook,
        args.counts,
        args.report,
        args.source_year_summary,
        args.backtest,
        args.forecasts,
        train_flaml=args.train_flaml,
        flaml_time_budget_seconds=args.flaml_time_budget_seconds,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
