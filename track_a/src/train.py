"""Train >=3 different models, log to MLflow, compare, register best (Staging -> Production)."""
import sys
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import compute_metrics, save_confusion_matrix, save_roc_curve
from src.preprocess import FEATURES, RANDOM_STATE, ROOT, TARGET, build_preprocessor, load_data

TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
EXPERIMENT = "telco-churn"
MODEL_NAME = "telco-churn-model"

# Genuinely different families / hyperparameters.
CONFIGS = [
    ("logreg_C0.1_l2", True, LogisticRegression(C=0.1, penalty="l2", max_iter=1000)),
    ("logreg_balanced_C1", True, LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
    ("random_forest_d8_n300", False, RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
    ("gradient_boosting_d3_n200", False, GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE)),
]


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, stratify=df[TARGET], random_state=RANDOM_STATE)

    for name, scale, clf in CONFIGS:
        with mlflow.start_run(run_name=name):
            pipe = Pipeline([("prep", build_preprocessor(scale)), ("clf", clf)])
            pipe.fit(X_train, y_train)
            pred, proba = pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1]
            mlflow.log_param("model_type", type(clf).__name__)
            mlflow.log_params({k: v for k, v in clf.get_params().items() if v is not None})
            metrics = compute_metrics(y_test, pred, proba)
            mlflow.log_metrics(metrics)
            with tempfile.TemporaryDirectory() as d:
                save_confusion_matrix(y_test, pred, Path(d) / "confusion_matrix.png")
                save_roc_curve(y_test, proba, Path(d) / "roc_curve.png")
                mlflow.log_artifacts(d, artifact_path="plots")
            mlflow.sklearn.log_model(pipe, artifact_path="model",
                                     signature=infer_signature(X_test, pred),
                                     input_example=X_test.head(3))
            print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    register_best()


def register_best(metric: str = "f1"):
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    runs = mlflow.search_runs([exp.experiment_id], order_by=[f"metrics.{metric} DESC"])
    cols = ["tags.mlflow.runName", "metrics.accuracy", "metrics.precision",
            "metrics.recall", "metrics.f1", "metrics.roc_auc"]
    table = runs[cols].rename(columns=lambda c: c.split(".")[-1])
    out = ROOT / "reports" / "run_comparison.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(table.round(4).to_markdown(index=False) if _has_tabulate() else table.round(4).to_string(index=False))
    print("\nRun comparison (sorted by F1):\n" + table.round(4).to_string(index=False))

    best = runs.iloc[0]
    print(f"\nBest run: {best['tags.mlflow.runName']} ({metric}={best['metrics.' + metric]:.4f})")
    mv = mlflow.register_model(f"runs:/{best.run_id}/model", MODEL_NAME)
    client.set_model_version_tag(MODEL_NAME, mv.version, "selected_by", f"highest {metric}")
    for stage in ("Staging", "Production"):
        client.transition_model_version_stage(MODEL_NAME, mv.version, stage)
        print(f"{MODEL_NAME} v{mv.version} -> {stage}")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
