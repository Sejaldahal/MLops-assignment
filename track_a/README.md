# Track A — Telco Churn MLOps (uv · MLflow · Evidently · FastAPI)

Workflow: **data → training → tracking → registry → serving → monitoring → retraining (if needed)**

```
data/telco_churn.csv        IBM Telco Customer Churn (7,043 rows, target Churn Yes/No)
src/preprocess.py           loading/cleaning, feature lists, sklearn ColumnTransformer
src/evaluate.py             metrics + confusion matrix / ROC plots
src/train.py                4 model configs -> MLflow runs -> compare -> register -> Staging -> Production
src/serve.py                FastAPI app loading models:/telco-churn-model/Production
src/drift_monitor.py        Evidently drift report (70/30 split, injected drift) logged to MLflow
reports/                    evidently_report.html, run_comparison.md
artifacts/                  confusion matrix + ROC PNG for every run
```

## a. Environment & reproducibility (uv)

The project needs mlflow, evidently, scikit-learn, FastAPI and SQLAlchemy to be mutually compatible.
This was a real problem: an unpinned install resolved SQLAlchemy 2.1, which breaks MLflow 2.22's SQLite
backend (`ImportError: FallbackAsyncAdaptedQueuePool`). `pyproject.toml` pins `sqlalchemy<2.1` and `mlflow<3`,
and `uv.lock` (committed) freezes the entire tree. Python is 3.11–3.13.

```bash
git clone <repo> && cd track_a
uv sync            # one command: creates .venv from uv.lock
```

## Commands
Everything at once: `./run_all.sh` (add `serve` to also start the API).

```bash
uv run python src/train.py                       # train 4 models, log, register, promote
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # compare runs at http://127.0.0.1:5000
uv run uvicorn src.serve:app --port 8000         # serve Production model
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d @sample_request.json
# or open http://localhost:8000/docs. All 19 feature columns are required (missing ones return 422).
uv run python src/drift_monitor.py               # Evidently report -> reports/ + MLflow artifact
```

`GET /health` reports the served model. `/predict` returns churn flag and probability per record.

## b. Experiment tracking (MLflow)

Stratified 80/20 train/test split (seed 42). Backend: `sqlite:///mlflow.db`, experiment `telco-churn`.
Each run logs all hyperparameters, accuracy/precision/recall/F1/ROC-AUC, the full sklearn pipeline
(preprocessing + model, with signature), and `plots/confusion_matrix.png` + `plots/roc_curve.png`.

Models varied on family and hyperparameters (not seeds):

| Run                       | Model               | Key hyperparameters                           |
| ------------------------- | ------------------- | --------------------------------------------- |
| logreg_C0.1_l2            | Logistic regression | C=0.1, L2, no class weights                   |
| logreg_balanced_C1        | Logistic regression | C=1.0, class_weight=balanced                  |
| random_forest_d8_n300     | Random forest       | 300 trees, max_depth=8, class_weight=balanced |
| gradient_boosting_d3_n200 | Gradient boosting   | 200 trees, depth 3, lr=0.05                   |

Results on the held-out test set (also in `reports/run_comparison.md`), sorted by F1:

| Run                             | Accuracy | Precision | Recall           | F1               | ROC-AUC |
| ------------------------------- | -------- | --------- | ---------------- | ---------------- | ------- |
| **random_forest_d8_n300** | 0.7473   | 0.5158    | **0.7834** | **0.6221** | 0.8400  |
| logreg_balanced_C1              | 0.7381   | 0.5043    | 0.7834           | 0.6136           | 0.8413  |
| logreg_C0.1_l2                  | 0.7999   | 0.6456    | 0.5455           | 0.5913           | 0.8409  |
| gradient_boosting_d3_n200       | 0.8027   | 0.6644    | 0.5187           | 0.5826           | 0.8432  |

**Decision: register `random_forest_d8_n300`.** Only ~27% of customers churn, so accuracy is misleading:
gradient boosting has the best accuracy (0.803) and ROC-AUC (0.843) but recalls only 51.9% of churners,
so it misses nearly half the customers we want to retain. The random forest has the highest F1 (0.622)
and recall (0.783) at the cost of accuracy (0.747) and precision (0.516). ROC-AUC is essentially tied
across all four (0.840–0.843), so it doesn't discriminate; F1 was the selection metric
(`register_best(metric="f1")`). The trade-off is more false alarms for retention offers in exchange for catching
more churners. The F1 margin over balanced logistic regression is small (0.622 vs 0.614), so the
simpler linear model is a reasonable alternative if interpretability matters; with a single split we
haven't measured whether that gap is significant.

**Registry:** `telco-churn-model` v1 was registered, then transitioned **None → Staging → Production**
(see `register_best` in `src/train.py`). Note MLflow deprecates stages in favour of aliases; stages still
work in 2.22 and are used here because the assignment asks for them.

## c. Monitoring & drift (Evidently AI)

* **Reference** = random 70% of the data (stands in for training-time data).
* **Current** = remaining 30% (stands in for production), with drift injected on purpose:
  `MonthlyCharges` shifted +$15 plus N(0, 8) noise; `Contract == Month-to-month` oversampled by 80% of
  the set size; 20% of `No` labels flipped to `Yes` (label/concept drift).
* Report: `DataDriftPreset` on all features, `ValueDrift` on `Churn` (target drift), `DriftedColumnsCount`,
  plus **two custom metrics** built on Evidently's `SingleValueMetric` API: `MeanShift(MonthlyCharges)`
  and `SegmentChurnShift(Month-to-month)`.
* Output: `reports/evidently_report.html`, logged to MLflow (run `drift_monitoring`, artifact path `evidently/`)
  together with the drift scores and custom metrics.

Results (Wasserstein-normed for numeric, Jensen-Shannon for categorical, threshold 0.1):

| Column             | Score      | Drifted?          |
| ------------------ | ---------- | ----------------- |
| MonthlyCharges     | 0.515      | yes (injected)    |
| Contract           | 0.157      | yes (injected)    |
| Churn (target)     | 0.163      | yes (injected)    |
| tenure             | 0.264      | yes (side effect) |
| TotalCharges       | 0.187      | yes (side effect) |
| all other features | 0.01–0.07 | no                |

Custom metrics: mean MonthlyCharges shift = +15.5; Month-to-month churn-rate shift = +0.138.

**Interpretation.** Every perturbation we engineered was detected, and no untouched independent feature was
falsely flagged. `tenure` and `TotalCharges` also drifted, but we did not edit them: month-to-month customers
have short tenure and low totals, so oversampling them moves those columns too. That is the kind of
knock-on effect that appears in production. In production, this pattern would mean the model sees
pricier, shorter-tenure, more churn-prone customers than it was trained on, so its calibration
(and precision/recall) can't be trusted, and an elevated churn rate would show up as target drift only
once labels arrive.

**Action policy.** If `drifted_columns_share` exceeds ~0.3, or target drift fires, or the custom
MonthlyCharges shift exceeds a business tolerance: alert, evaluate the Production model on recently labelled
data, and if F1 drops materially, rerun `train.py` on fresh data and promote the new version through
Staging → Production. Thresholds here are illustrative, not tuned.

## d. Orchestration

Airflow bonus not implemented.

## Caveats

* Single train/test split, no cross-validation or hyperparameter search; differences of ~0.01 are within noise.
* Drift is synthetic, so it demonstrates detection, not real-world drift behaviour.
* `mlflow.db`/`mlruns/` are git-ignored; regenerate them with the commands above (results may vary slightly by library version).
  Screenshots of the MLflow UI were not taken; use `reports/run_comparison.md` and the UI command above.
