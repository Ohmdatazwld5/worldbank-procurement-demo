"""Retrieval/evidence evaluation and closed-loop feedback promotion."""

from __future__ import annotations

from typing import Any


def score_retrieval(
    retrieved_document_ids: list[str],
    expected_document_ids: list[str],
) -> dict[str, float]:
    retrieved, expected = set(retrieved_document_ids), set(expected_document_ids)
    true_positives = len(retrieved.intersection(expected))
    precision = true_positives / len(retrieved) if retrieved else 0.0
    recall = true_positives / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_cases(
    cases: list[dict[str, Any]],
    query_function: Any,
    *,
    experiment_name: str = "worldbank-procurement-evaluation",
) -> dict[str, Any]:
    """Run fixed human-reviewed cases; expected evidence IDs are never self-labeled."""
    if not cases:
        raise ValueError("At least one human-reviewed evaluation case is required.")
    outcomes = []
    for case in cases:
        result = query_function(case["project_id"], case["question"])
        retrieval = score_retrieval(
            [item.get("document_id", "") for item in result.get("evidence", [])],
            case.get("expected_document_ids", []),
        )
        cited = set(result.get("citations", []))
        expected_citations = set(case.get("expected_citations", []))
        outcomes.append(
            {
                "status_correct": result.get("status") == case.get("expected_status"),
                "citation_recall": (
                    len(cited.intersection(expected_citations)) / len(expected_citations)
                    if expected_citations
                    else None
                ),
                **retrieval,
            }
        )
    summary = {}
    for metric in ("precision", "recall", "f1", "citation_recall", "status_correct"):
        observed = [
            float(item[metric]) for item in outcomes if item[metric] is not None
        ]
        summary[metric] = sum(observed) / len(observed) if observed else None
    try:
        import mlflow

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="retrieval-and-guardrail-eval"):
            mlflow.log_metrics(
                {key: value for key, value in summary.items() if value is not None}
            )
            mlflow.set_tag("evaluation_source", "human_reviewed_cases")
    except ImportError:
        raise RuntimeError("MLflow is required to persist evaluation results.") from None
    return {"summary": summary, "cases": outcomes}
