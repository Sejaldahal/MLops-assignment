"""Evidently data/target drift monitoring; the HTML report is logged to MLflow.

70% reference vs 30% current; synthetic drift is injected into current:
  * MonthlyCharges: +$15 shift plus N(0, 8) noise per row  (numeric feature drift)
  * Contract: Month-to-month oversampled                    (categorical skew)
  * Churn: 20% of 'No' labels flipped to 'Yes'              (label / concept drift)
"""
import sys
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidently import DataDefinition, Dataset, Report
from evidently.core.metric_types import SingleValueCalculation, SingleValueMetric
from evidently.metrics import DriftedColumnsCount, ValueDrift
from evidently.presets import DataDriftPreset

from src.preprocess import CATEGORICAL, RANDOM_STATE, ROOT, TARGET, load_data
from src.train import EXPERIMENT, TRACKING_URI

REPORT_PATH = ROOT / "reports" / "evidently_report.html"
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


class MeanShift(SingleValueMetric):
    """Custom metric: mean(column) in current data minus mean(column) in reference."""
    column: str


class MeanShiftCalculation(SingleValueCalculation[MeanShift]):
    def calculate(self, context, current_data, reference_data):
        cur = current_data.column(self.metric.column).data.mean()
        ref = reference_data.column(self.metric.column).data.mean()
        return self.result(cur - ref), self.result(0.0)

    def display_name(self) -> str:
        return f"Mean shift of '{self.metric.column}' (current - reference)"


class SegmentChurnShift(SingleValueMetric):
    """Custom metric: change in churn rate inside one Contract segment."""
    segment: str = "Month-to-month"


class SegmentChurnShiftCalculation(SingleValueCalculation[SegmentChurnShift]):
    def calculate(self, context, current_data, reference_data):
        def rate(ds):
            d = ds.as_dataframe()
            return d.loc[d["Contract"] == self.metric.segment, TARGET].mean()
        return self.result(rate(current_data) - rate(reference_data)), self.result(0.0)

    def display_name(self) -> str:
        return f"Churn-rate shift in Contract == '{self.metric.segment}'"


def make_split(seed: int = RANDOM_STATE):
    df = load_data()
    ref, cur = train_test_split(df, train_size=0.7, random_state=seed)
    return ref.reset_index(drop=True), cur.reset_index(drop=True)


def inject_drift(cur: pd.DataFrame, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cur = cur.copy()
    cur["MonthlyCharges"] = (cur["MonthlyCharges"] + 15 + rng.normal(0, 8, len(cur))).clip(lower=18)
    # oversample Month-to-month rows so they dominate the current set
    m2m = cur[cur["Contract"] == "Month-to-month"]
    cur = pd.concat([cur, m2m.sample(n=int(len(cur) * 0.8), replace=True, random_state=seed)],
                    ignore_index=True)
    # concept drift: 20% of non-churners become churners (churn rate rises)
    flip = (cur[TARGET] == 0) & (rng.random(len(cur)) < 0.20)
    cur.loc[flip, TARGET] = 1
    return cur


def main():
    ref, cur = make_split()
    cur = inject_drift(cur)
    cols = NUM_COLS + CATEGORICAL + [TARGET]
    definition = DataDefinition(numerical_columns=NUM_COLS, categorical_columns=CATEGORICAL + [TARGET])
    ref_ds = Dataset.from_pandas(ref[cols], data_definition=definition)
    cur_ds = Dataset.from_pandas(cur[cols], data_definition=definition)

    report = Report([
        DataDriftPreset(columns=NUM_COLS + CATEGORICAL),
        ValueDrift(column=TARGET),  # target drift
        DriftedColumnsCount(),
        MeanShift(column="MonthlyCharges"),
        SegmentChurnShift(segment="Month-to-month"),
    ])
    snap = report.run(cur_ds, ref_ds)
    REPORT_PATH.parent.mkdir(exist_ok=True)
    snap.save_html(str(REPORT_PATH))

    # Machine-readable summary
    values = {}
    print("Per-column drift (distance > threshold => drifted):")
    for item in snap.dict()["metrics"]:
        name, v = item["metric_name"], item["value"]
        if isinstance(v, dict):
            values.update({f"drifted_columns_{k}": float(x) for k, x in v.items()})
        elif name.startswith("ValueDrift"):
            col, thr = item["config"]["column"], item["config"]["threshold"]
            values[f"drift_score_{col}"] = float(v)
            print(f"  {col:18s} {float(v):.3f} {'DRIFT' if v > thr else 'ok'}")
        else:
            values[name.split("(")[0] + "_" + name.split("=")[-1].rstrip(")")] = float(v)
            print(f"  {name}: {float(v):.4f}")
    print({k: round(v, 3) for k, v in values.items() if k.startswith("drifted_")})

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="drift_monitoring"):
        mlflow.log_params({"reference_frac": 0.7, "current_frac": 0.3,
                           "injected": "MonthlyCharges shift+noise; Contract skew; 20% of No->Yes label flips"})
        mlflow.log_metric("mean_shift_monthly_charges", cur["MonthlyCharges"].mean() - ref["MonthlyCharges"].mean())
        mlflow.log_metric("churn_rate_reference", ref[TARGET].mean())
        mlflow.log_metric("churn_rate_current", cur[TARGET].mean())
        mlflow.log_metrics(values)
        mlflow.log_artifact(str(REPORT_PATH), artifact_path="evidently")
    print(f"\nReport saved: {REPORT_PATH} (and logged to MLflow)")


if __name__ == "__main__":
    main()
