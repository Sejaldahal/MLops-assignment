"""FastAPI service loading the Production model from the MLflow registry.

Run: uv run uvicorn src.serve:app --port 8000
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess import FEATURES, ROOT

MODEL_NAME = "telco-churn-model"
state = {}


@asynccontextmanager
async def lifespan(app):
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    state["model"] = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
    yield


app = FastAPI(title="Telco Churn API", lifespan=lifespan)


class Customers(BaseModel):
    records: list[dict]


@app.get("/")
def root():
    return {"service": "Telco Churn API", "docs": "/docs", "predict": "POST /predict",
            "required_features": FEATURES}


@app.get("/health")
def health():
    return {"status": "ok", "model": f"{MODEL_NAME}/Production"}


@app.post("/predict")
def predict(body: Customers):
    df = pd.DataFrame(body.records)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")
    df = df[FEATURES]
    proba = state["model"].predict_proba(df)[:, 1]
    return {"predictions": [{"churn": bool(p >= 0.5), "churn_probability": float(p)} for p in proba]}
