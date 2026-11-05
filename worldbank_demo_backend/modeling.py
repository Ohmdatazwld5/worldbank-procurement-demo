"""FLAML procurement models with MLflow experiment and registry integration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


MIN_TRAINING_ROWS = 100
TARGETS = {
    "timeline_days": "actual_days",
    "amendment_probability": "has_amendment",
    "rebid_probability": "is_rebid",
}
REQUIRED_FEATURES = {
    "planned_days",
    "project_year",
    "country_code",
    "sector_code",
}
APPROVED_FEATURES = REQUIRED_FEATURES | {
    "procurement_method",
    "governance_effectiveness",
    "regulatory_quality",
    "control_of_corruption",
    "governance_effectiveness_source_year",
    "regulatory_quality_source_year",
    "control_of_corruption_source_year",
    "gdp_growth_pct",
    "inflation_pct",
    "population",
    "project_budget",
}


def train_models(
    procurement_csv: str | Path,
    *,
    experiment_name: str = "worldbank-procurement-demo",
    time_budget_seconds: int = 60,
    register_models: bool = False,
) -> dict[str, Any]:
    """Train three supervised models from labeled procurement history.

    The feed must contain actual_days, amendment_count, outcome, and pre-award
    predictors such as sector_code, planned_days, procurement_method,
    governance_score, country_code, and project_year. Outcome fields are
    explicitly excluded from the predictor matrix to prevent label leakage.
    """
    import mlflow
    import pandas as pd
    from flaml import AutoML
    from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
    from sklearn.model_selection import train_test_split

    source_path = str(procurement_csv)
    if "://" in source_path or Path(source_path).is_dir():
        from deltalake import DeltaTable

        storage_options = os.getenv("DELTA_STORAGE_OPTIONS_JSON", "{}")
        try:
            options = __import__("json").loads(storage_options)
        except ValueError as exc:
            raise ValueError("DELTA_STORAGE_OPTIONS_JSON must be valid JSON.") from exc
        if source_path.startswith("s3a://"):
            source_path = source_path.replace("s3a://", "s3://", 1)
        frame = DeltaTable(
            source_path, storage_options=options or None
        ).to_pyarrow_table().to_pandas()
    else:
        frame = pd.read_csv(source_path)
    required = {
        "actual_days",
        "amendment_count",
        "outcome",
        *REQUIRED_FEATURES,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Procurement data is missing required labels: {sorted(missing)}")
    if len(frame) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Need at least {MIN_TRAINING_ROWS} labeled historical records; "
            f"received {len(frame)}. Do not treat demo rows as a trained model."
        )
    planned_days = pd.to_numeric(frame["planned_days"], errors="coerce")
    actual_days = pd.to_numeric(frame["actual_days"], errors="coerce")
    amendment_count = pd.to_numeric(frame["amendment_count"], errors="coerce")
    if planned_days.isna().any() or (planned_days <= 0).any():
        raise ValueError("planned_days must contain positive numeric values for every training row.")
    if actual_days.dropna().empty or (actual_days.dropna() <= 0).any():
        raise ValueError("Completed-label actual_days values must be positive.")
    if amendment_count.isna().any() or (amendment_count < 0).any():
        raise ValueError("amendment_count must be a non-negative label for every training row.")
    if frame["outcome"].isna().any() or frame["outcome"].astype(str).str.strip().eq("").any():
        raise ValueError("outcome must be labeled for every training row.")
    unresolved_outcomes = {
        "pending",
        "ongoing",
        "on-going",
        "in progress",
        "in_progress",
        "not started",
        "unknown",
        "not available",
    }
    if frame["outcome"].astype(str).str.strip().str.casefold().isin(unresolved_outcomes).any():
        raise ValueError(
            "Training includes unresolved tenders; train only on finalized outcome labels."
        )

    feature_columns = [
        column
        for column in sorted(APPROVED_FEATURES)
        if column in frame.columns
    ]
    if not feature_columns:
        raise ValueError("No pre-award predictor columns remain after leakage checks.")
    features = frame[feature_columns].copy()
    for column in features.select_dtypes(include=["object"]).columns:
        features[column] = features[column].fillna("UNKNOWN").astype(str)
    for column in features.select_dtypes(exclude=["object"]).columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    if "project_year" in frame and frame["project_year"].notna().sum() >= 10:
        years = pd.to_numeric(frame["project_year"], errors="coerce")
        threshold = years.quantile(0.8)
        train_mask = years < threshold
        test_mask = years >= threshold
        if train_mask.sum() < 2 or test_mask.sum() < 2:
            train_mask, test_mask = _random_split(frame, pd, train_test_split)
            holdout_strategy = "random"
        else:
            holdout_strategy = "chronological"
    else:
        train_mask, test_mask = _random_split(frame, pd, train_test_split)
        holdout_strategy = "random"
    train_x, test_x = features.loc[train_mask], features.loc[test_mask]
    if train_x.empty or test_x.empty:
        raise ValueError("Unable to create non-empty train and validation partitions.")

    mlflow.set_experiment(experiment_name)
    results: dict[str, Any] = {"feature_columns": feature_columns, "models": {}}
    for model_key, target_column in TARGETS.items():
        if target_column == "has_amendment":
            target = (pd.to_numeric(frame["amendment_count"], errors="coerce").fillna(0) > 0).astype(int)
            task = "classification"
        elif target_column == "is_rebid":
            target = frame["outcome"].astype(str).str.strip().str.casefold().isin(
                {"rebid", "re-bid", "retendered"}
            ).astype(int)
            task = "classification"
        else:
            target = pd.to_numeric(frame[target_column], errors="coerce")
            valid = target.notna()
            train_x_for_target = train_x.loc[valid.loc[train_x.index]]
            test_x_for_target = test_x.loc[valid.loc[test_x.index]]
            if len(train_x_for_target) < 2 or test_x_for_target.empty:
                raise ValueError(f"Not enough non-null {target_column} labels to train.")
            model, metrics = _fit_model(
                AutoML,
                mlflow,
                model_key,
                train_x_for_target,
                test_x_for_target,
                target.loc[train_x_for_target.index],
                target.loc[test_x_for_target.index],
                task,
                time_budget_seconds,
                mean_absolute_error,
                roc_auc_score,
                accuracy_score,
                holdout_strategy,
            )
            results["models"][model_key] = _model_result(model, metrics, model_key, mlflow, register_models)
            continue

        y_train, y_test = target.loc[train_x.index], target.loc[test_x.index]
        if y_train.nunique() < 2:
            raise ValueError(f"{model_key} needs at least two classes in the training partition.")
        model, metrics = _fit_model(
            AutoML,
            mlflow,
            model_key,
            train_x,
            test_x,
            y_train,
            y_test,
            task,
            time_budget_seconds,
            mean_absolute_error,
            roc_auc_score,
            accuracy_score,
            holdout_strategy,
        )
        results["models"][model_key] = _model_result(model, metrics, model_key, mlflow, register_models)
    return results


def _random_split(frame: Any, pd: Any, train_test_split: Any) -> tuple[Any, Any]:
    indexes = frame.index.to_numpy()
    train, test = train_test_split(indexes, test_size=0.2, random_state=42)
    return pd.Series(frame.index.isin(train), index=frame.index), pd.Series(
        frame.index.isin(test), index=frame.index
    )


def _fit_model(
    automl_class: Any,
    mlflow: Any,
    model_key: str,
    train_x: Any,
    test_x: Any,
    train_y: Any,
    test_y: Any,
    task: str,
    time_budget_seconds: int,
    mean_absolute_error: Any,
    roc_auc_score: Any,
    accuracy_score: Any,
    holdout_strategy: str,
) -> tuple[Any, dict[str, Any]]:
    import pandas as pd

    from sklearn.metrics import brier_score_loss, precision_score, recall_score

    train_x = train_x.copy()
    test_x = test_x.copy()
    numeric_columns = list(train_x.select_dtypes(exclude=["object"]).columns)
    numeric_fill_values: dict[str, float] = {}
    for column in numeric_columns:
        median = train_x[column].median()
        fill_value = float(median) if pd.notna(median) else 0.0
        numeric_fill_values[column] = fill_value
        train_x[column] = train_x[column].fillna(fill_value)
        test_x[column] = test_x[column].fillna(fill_value)

    automl = automl_class()
    with mlflow.start_run(run_name=model_key) as run:
        automl.fit(
            X_train=train_x,
            y_train=train_y,
            task=task,
            time_budget=max(1, int(time_budget_seconds)),
            eval_method="cv",
            metric="mae" if task == "regression" else "accuracy",
            seed=42,
            verbose=0,
        )
        predicted = automl.predict(test_x)
        if task == "regression":
            metrics = {"mae_days": float(mean_absolute_error(test_y, predicted))}
        else:
            predicted_labels = [int(value) for value in predicted]
            metrics = {
                "accuracy": float(accuracy_score(test_y, predicted_labels)),
                "precision": float(precision_score(test_y, predicted_labels, zero_division=0)),
                "recall": float(recall_score(test_y, predicted_labels, zero_division=0)),
            }
            if hasattr(automl, "predict_proba") and test_y.nunique() == 2:
                probabilities = automl.predict_proba(test_x)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(test_y, probabilities))
                metrics["brier_score"] = float(
                    brier_score_loss(test_y, probabilities)
                )
        mlflow.log_params(
            {
                "task": task,
                "target": model_key,
                "feature_columns": ",".join(train_x.columns),
                "training_rows": len(train_x),
                "validation_rows": len(test_x),
                "time_budget_seconds": int(time_budget_seconds),
                "holdout_strategy": holdout_strategy,
                "automl_estimator": str(getattr(automl, "best_estimator", "unknown")),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tag("label_leakage_guard", "target_and_post_award_fields_excluded")
        mlflow.set_tag("model_intended_use", "decision_support_not_automated_award")
        model_uri = f"runs:/{run.info.run_id}/{model_key}"
        pyfunc_model = _create_pyfunc_model(
            mlflow,
            automl,
            task,
            list(train_x.columns),
            list(train_x.select_dtypes(include=["object"]).columns),
            numeric_fill_values,
        )
        mlflow.pyfunc.log_model(
            artifact_path=model_key,
            python_model=pyfunc_model,
            input_example=train_x.head(2),
        )
        return automl, {**metrics, "_run_id": run.info.run_id, "_model_uri": model_uri}


def _create_pyfunc_model(
    mlflow: Any,
    automl: Any,
    task: str,
    feature_columns: list[str],
    categorical_columns: list[str],
    numeric_fill_values: dict[str, float],
) -> Any:
    class FLAMLModel(mlflow.pyfunc.PythonModel):
        def predict(self, context: Any, model_input: Any) -> Any:
            import pandas as pd

            frame = pd.DataFrame(model_input).reindex(columns=feature_columns)
            for column in categorical_columns:
                frame[column] = frame[column].fillna("UNKNOWN").astype(str)
            for column in feature_columns:
                if column not in categorical_columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(
                        numeric_fill_values.get(column, 0.0)
                    )
            if task == "regression":
                return automl.predict(frame)
            return automl.predict_proba(frame)[:, 1]

    return FLAMLModel()


def _model_result(
    model: Any,
    metrics: dict[str, Any],
    model_key: str,
    mlflow: Any,
    register: bool,
) -> dict[str, Any]:
    result = {"metrics": {key: value for key, value in metrics.items() if not key.startswith("_")}}
    result["model_uri"] = metrics["_model_uri"]
    if register:
        registry_prefix = os.getenv("MODEL_REGISTRY_PREFIX", "worldbank_demo")
        registered = mlflow.register_model(
            result["model_uri"], f"{registry_prefix}.{model_key}"
        )
        result["registered_model"] = {"name": registered.name, "version": registered.version}
    return result


def main() -> None:
    import argparse
    import json

    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "procurement_data",
        nargs="?",
        default=os.getenv("PROCUREMENT_DATA_PATH"),
        help="Labeled procurement CSV or Delta Lake table path.",
    )
    parser.add_argument("--register", action="store_true")
    parser.add_argument(
        "--time-budget-seconds",
        type=int,
        default=int(os.getenv("FLAML_TIME_BUDGET_SECONDS", "60")),
    )
    args = parser.parse_args()
    if not args.procurement_data:
        parser.error("Provide a procurement data path or set PROCUREMENT_DATA_PATH.")
    print(
        json.dumps(
            train_models(
                args.procurement_data,
                time_budget_seconds=args.time_budget_seconds,
                register_models=args.register,
            ),
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
