"""Train, compare, evaluate and (when MLflow is installed) register classifiers."""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from .config import (
    ARTIFACTS_DIR, FEATURE_IMPORTANCE_PATH, FEATURE_STORE_PATH, METRICS_PATH,
    MODEL_METADATA_PATH, BEST_MODEL_PATH, FEATURE_COLUMNS, TARGET_MAP, RANDOM_STATE,
)
from .feature_store import get_features


def metrics_for(model, X: pd.DataFrame, y: pd.Series, threshold: float) -> dict:
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    try:
        auc = float(roc_auc_score(y, probabilities))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": round(float(accuracy_score(y, predictions)), 4),
        "precision": round(float(precision_score(y, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y, predictions, zero_division=0)), 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
        "support": {"acceptable": int((y == 0).sum()), "moisture_risk": int((y == 1).sum())},
    }


def make_pipeline(estimator) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[("numeric", SimpleImputer(strategy="median"), FEATURE_COLUMNS)],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocessor), ("classifier", estimator)])


def register_with_mlflow(model: Pipeline, metrics: dict, threshold: float, chosen_name: str, model_version: str) -> dict:
    """Log model and metrics; register as a versioned MLflow model when available."""
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        warnings.warn("MLflow is not installed in this environment; local model artifacts were saved, but registry logging was skipped.")
        return {"status": "not_executed", "reason": "MLflow package unavailable in build environment"}

    tracking_uri = (ARTIFACTS_DIR / "mlflow.db").as_uri()
    # A SQLite-backed tracking store keeps experiment metadata local and reproducible.
    tracking_uri = "sqlite:///" + str(ARTIFACTS_DIR / "mlflow.db").replace("\\", "/")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("manuscript-moisture-risk")
    with mlflow.start_run(run_name=f"{chosen_name}-{model_version}") as run:
        mlflow.log_param("dataset_version", "Version 1 - Baseline (Version 2 used for external stress evaluation)")
        mlflow.log_param("selected_model", chosen_name)
        mlflow.log_param("decision_threshold", threshold)
        mlflow.log_param("feature_count", len(FEATURE_COLUMNS))
        mlflow.log_param("model_version", model_version)
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            mlflow.log_metric(key, metrics[key])
        mlflow.log_dict(metrics, "evaluation_metrics.json")
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="ManuscriptMoistureRiskModel",
                skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
            )
        except TypeError:
            # Compatibility path for future MLflow versions using `name`.
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                registered_model_name="ManuscriptMoistureRiskModel",
                skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
            )
        return {
            "status": "registered",
            "tracking_uri": tracking_uri,
            "experiment_name": "manuscript-moisture-risk",
            "run_id": run.info.run_id,
            "registered_model_name": "ManuscriptMoistureRiskModel",
        }


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    data = get_features(FEATURE_STORE_PATH)
    data["target"] = data["moisture_risk_status"].map(TARGET_MAP).astype(int)
    v1 = data[data["dataset_version"] == "Version 1 - Baseline"].copy()
    v2 = data[data["dataset_version"] == "Version 2 - Monsoon"].copy()

    X = v1[FEATURE_COLUMNS]
    y = v1["target"]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=4, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=9, min_samples_leaf=3,
            class_weight="balanced_subsample", max_features="sqrt",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }
    threshold = 0.35  # recall-oriented operating point for an early-warning use case
    results = {}
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = make_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        valid_metrics = metrics_for(pipeline, X_valid, y_valid, threshold)
        results[name] = {"validation_v1": valid_metrics}
        fitted[name] = pipeline

    # Prefer recall (missed damage is costly), then F1 and ROC AUC as tie-breakers.
    best_name = sorted(
        results,
        key=lambda name: (
            results[name]["validation_v1"]["recall"],
            results[name]["validation_v1"]["f1"],
            results[name]["validation_v1"]["roc_auc"],
        ),
        reverse=True,
    )[0]
    best_model = fitted[best_name]
    y_v2 = v2["moisture_risk_status"].map(TARGET_MAP).astype(int)
    v2_metrics = metrics_for(best_model, v2[FEATURE_COLUMNS], y_v2, threshold)
    results[best_name]["external_stress_test_v2"] = v2_metrics

    joblib.dump(best_model, BEST_MODEL_PATH)
    feature_importance = None
    classifier = best_model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        feature_importance = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": classifier.feature_importances_,
        }).sort_values("importance", ascending=False)
        feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    mlflow_status = register_with_mlflow(best_model, results[best_name]["validation_v1"], threshold, best_name, "1")
    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "training_version": "Version 1 - Baseline",
            "external_evaluation_version": "Version 2 - Monsoon",
            "training_rows": int(len(v1)),
            "external_evaluation_rows": int(len(v2)),
        },
        "threshold": threshold,
        "selection_rule": "Highest Version 1 validation recall, then F1, then ROC-AUC",
        "candidate_metrics": results,
        "selected_model": best_name,
        "mlflow": mlflow_status,
    }
    METRICS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    metadata = {
        "model_name": "ManuscriptMoistureRiskModel",
        "model_family": best_name,
        "model_version": "1",
        "threshold": threshold,
        "feature_columns": FEATURE_COLUMNS,
        "target_map": TARGET_MAP,
        "artifact_path": str(BEST_MODEL_PATH),
        "mlflow_status": mlflow_status,
        "trained_at_utc": output["created_at_utc"],
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Selected model: {best_name}")
    print(f"Version 1 validation: {results[best_name]['validation_v1']}")
    print(f"Version 2 monsoon stress test: {v2_metrics}")
    print(f"MLflow: {mlflow_status['status']}")


if __name__ == "__main__":
    main()
