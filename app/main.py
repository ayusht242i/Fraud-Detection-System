import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

MODELS = Path(__file__).resolve().parent.parent / "ml" / "models"

rf = joblib.load(MODELS / "random_forest.joblib")
iso = joblib.load(MODELS / "isolation_forest.joblib")
scaler = joblib.load(MODELS / "scaler.joblib")
cfg = json.loads((MODELS / "composite_config.json").read_text())

THRESHOLD = cfg["threshold"]
IF_MIN, IF_MAX = cfg["if_min"], cfg["if_max"]
FEATURES = list(rf.feature_names_in_)

fields = {f"V{i}": (float, ...) for i in range(1, 29)}
fields["Amount"] = (float, Field(..., ge=0, description="Raw transaction amount"))
fields["Time"] = (float, Field(..., ge=0, description="Seconds since the first transaction in the dataset"))
Transaction = create_model("Transaction", **fields)


class ScoreResponse(BaseModel):
    fraud_probability: float = Field(description="Random Forest probability that the transaction is fraud")
    is_flagged: bool = Field(description="True when fraud_probability reaches the threshold")
    threshold: float
    anomaly_score: float = Field(description="Isolation Forest score scaled to 0-1. Informational only, not used in the flag decision")


app = FastAPI(title="Fraud Detection API",
              description="Scores a transaction with a Random Forest model. Interactive docs at /docs.")


@app.get("/")
def health():
    return {"status": "ok", "threshold": THRESHOLD, "n_features": len(FEATURES)}


@app.get("/samples")
def samples():
    path = MODELS / "sample_rows.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run ml/make_samples.py first")
    return json.loads(path.read_text())


@app.post("/score", response_model=ScoreResponse)
def score(tx: Transaction):
    row = tx.model_dump()
    row["hour"] = (row["Time"] // 3600) % 24
    df = pd.DataFrame([row])
    df[["Amount", "hour"]] = scaler.transform(df[["Amount", "hour"]])
    X = df[FEATURES]
    prob = float(rf.predict_proba(X)[0, 1])
    raw = float(-iso.score_samples(X)[0])
    anomaly = min(max((raw - IF_MIN) / (IF_MAX - IF_MIN), 0.0), 1.0)
    return ScoreResponse(fraud_probability=round(prob, 4),
                         is_flagged=prob >= THRESHOLD,
                         threshold=round(THRESHOLD, 4),
                         anomaly_score=round(anomaly, 4))
