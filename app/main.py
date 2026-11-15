import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from app.db import session

MODELS = Path(__file__).resolve().parent.parent / "ml" / "models"

rf = joblib.load(MODELS / "random_forest.joblib")
iso = joblib.load(MODELS / "isolation_forest.joblib")
scaler = joblib.load(MODELS / "scaler.joblib")
cfg = json.loads((MODELS / "composite_config.json").read_text())

THRESHOLD = cfg["threshold"]
IF_MIN, IF_MAX = cfg["if_min"], cfg["if_max"]
FEATURES = list(rf.feature_names_in_)
V_COLS = [f"V{i}" for i in range(1, 29)]

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
              description="Scores transactions with a Random Forest model and records flags in Postgres. Interactive docs at /docs.")


def predict_df(df):
    df = df.copy()
    df["hour"] = (df["Time"] // 3600) % 24
    df[["Amount", "hour"]] = scaler.transform(df[["Amount", "hour"]])
    X = df[FEATURES]
    prob = rf.predict_proba(X)[:, 1]
    raw = -iso.score_samples(X)
    anomaly = ((raw - IF_MIN) / (IF_MAX - IF_MIN)).clip(0, 1)
    return prob, anomaly


def predict(row):
    prob, anomaly = predict_df(pd.DataFrame([row]))
    return float(prob[0]), float(anomaly[0])


SELECT_FEATURES = ("SELECT f.txn_id, " + ", ".join(f"f.v{i}" for i in range(1, 29)) +
                   ", t.amount, f.time_sec FROM transaction_features f "
                   "JOIN Transactions t ON t.txn_id = f.txn_id ")


def fetch_features(cur, where="", params=()):
    cur.execute(SELECT_FEATURES + where, params)
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["txn_id"] + V_COLS + ["Amount", "Time"])
    df["Amount"] = df["Amount"].astype(float)
    return df


def save_score(cur, txn_id, prob, anomaly):
    flagged = bool(prob >= THRESHOLD)
    cur.execute(
        "INSERT INTO ml_scores (txn_id, fraud_probability, anomaly_score, is_flagged) "
        "VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (txn_id) DO UPDATE SET fraud_probability = EXCLUDED.fraud_probability, "
        "anomaly_score = EXCLUDED.anomaly_score, is_flagged = EXCLUDED.is_flagged, scored_at = NOW()",
        (txn_id, round(prob, 4), round(anomaly, 4), flagged))
    if flagged:
        cur.execute(
            "INSERT INTO Audit_Logs (txn_id, flag_reason, risk_score, flagged_at) "
            "SELECT %s::int, 'ML model flagged', %s::numeric, NOW() "
            "WHERE NOT EXISTS (SELECT 1 FROM Audit_Logs WHERE txn_id = %s AND flag_reason = 'ML model flagged')",
            (txn_id, round(prob, 3), txn_id))
    return flagged


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
    prob, anomaly = predict(tx.model_dump())
    return ScoreResponse(fraud_probability=round(prob, 4),
                         is_flagged=prob >= THRESHOLD,
                         threshold=round(THRESHOLD, 4),
                         anomaly_score=round(anomaly, 4))


@app.post("/score/{txn_id}")
def score_stored_transaction(txn_id: int):
    with session() as cur:
        df = fetch_features(cur, "WHERE f.txn_id = %s", (txn_id,))
        if df.empty:
            raise HTTPException(status_code=404, detail="No stored features for this txn_id. Run ml/replay_to_db.py")
        prob, anomaly = predict_df(df)
        flagged = save_score(cur, txn_id, float(prob[0]), float(anomaly[0]))
    return {"txn_id": txn_id, "fraud_probability": round(float(prob[0]), 4),
            "is_flagged": flagged, "threshold": round(THRESHOLD, 4),
            "anomaly_score": round(float(anomaly[0]), 4), "written_to_audit_logs": flagged}


@app.post("/score-batch")
def score_batch():
    with session() as cur:
        df = fetch_features(cur, "LEFT JOIN ml_scores m ON m.txn_id = f.txn_id "
                                 "WHERE m.txn_id IS NULL ORDER BY f.txn_id")
        if df.empty:
            return {"newly_scored": 0, "newly_flagged": 0}
        prob, anomaly = predict_df(df)
        flagged = 0
        for txn_id, p, a in zip(df["txn_id"], prob, anomaly):
            flagged += int(save_score(cur, int(txn_id), float(p), float(a)))
    return {"newly_scored": len(df), "newly_flagged": flagged}


@app.get("/flagged")
def flagged(limit: int = 25):
    with session() as cur:
        cur.execute(
            "SELECT a.log_id, a.txn_id, a.flag_reason, a.risk_score, t.amount, t.txn_time, f.true_class "
            "FROM Audit_Logs a JOIN Transactions t ON t.txn_id = a.txn_id "
            "LEFT JOIN transaction_features f ON f.txn_id = a.txn_id "
            "ORDER BY a.risk_score DESC, a.log_id LIMIT %s", (limit,))
        rows = cur.fetchall()
    return [{"log_id": r[0], "txn_id": r[1], "flag_reason": r[2], "risk_score": float(r[3]),
             "amount": float(r[4]), "txn_time": r[5].isoformat(), "true_class": r[6]} for r in rows]


@app.get("/metrics")
def metrics():
    with session() as cur:
        cur.execute(
            "SELECT m.is_flagged, f.true_class, COUNT(*), "
            "MAX(ABS(m.fraud_probability - f.offline_prob)) "
            "FROM ml_scores m JOIN transaction_features f ON f.txn_id = m.txn_id "
            "GROUP BY m.is_flagged, f.true_class")
        rows = cur.fetchall()
    tp = fp = fn = tn = 0
    diff = 0.0
    for is_flagged, true, n, d in rows:
        diff = max(diff, float(d or 0))
        if is_flagged and true == 1:
            tp += n
        elif is_flagged:
            fp += n
        elif true == 1:
            fn += n
        else:
            tn += n
    return {"scored": tp + fp + fn + tn, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "max_abs_diff_vs_offline_probability": round(diff, 4),
            "note": "Demo mix of 30 fraud and 270 legit rows from the test set. Precision here is not comparable to the real 0.17% fraud rate."}


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(CORSMiddleware,
                   allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/transactions")
def transactions(limit: int = 1000):
    with session() as cur:
        cur.execute(
            "SELECT t.txn_id, t.amount, t.txn_time, m.fraud_probability, m.anomaly_score, f.true_class "
            "FROM ml_scores m JOIN Transactions t ON t.txn_id = m.txn_id "
            "LEFT JOIN transaction_features f ON f.txn_id = m.txn_id "
            "ORDER BY m.fraud_probability DESC, t.txn_id LIMIT %s", (limit,))
        rows = cur.fetchall()
    return {"threshold": THRESHOLD,
            "transactions": [{"txn_id": r[0], "amount": float(r[1]), "txn_time": r[2].isoformat(),
                              "probability": float(r[3]),
                              "anomaly": float(r[4]) if r[4] is not None else None,
                              "true_class": r[5]} for r in rows]}


from fastapi.responses import FileResponse


@app.get("/demo", include_in_schema=False)
def demo():
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")
