import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.db import connect

N_FRAUD, N_LEGIT = 30, 270
rng = np.random.default_rng(42)

X = pd.read_csv("ml/data/X_test.csv")
y = pd.read_csv("ml/data/y_test.csv")["Class"].values
scaler = joblib.load("ml/models/scaler.joblib")
rf = joblib.load("ml/models/random_forest.joblib")

fraud_idx = rng.choice(np.where(y == 1)[0], N_FRAUD, replace=False)
legit_idx = rng.choice(np.where(y == 0)[0], N_LEGIT, replace=False)
idx = rng.permutation(np.concatenate([fraud_idx, legit_idx]))

sample = X.iloc[idx].reset_index(drop=True)
labels = y[idx]
offline_prob = rf.predict_proba(sample[list(rf.feature_names_in_)])[:, 1]

# Undo the scaling to get the raw Amount and hour back
raw = scaler.inverse_transform(sample[["Amount", "hour"]])
amounts = np.round(raw[:, 0], 2)
hours = np.round(raw[:, 1]).astype(int)

V = [f"v{i}" for i in range(1, 29)]
conn = connect()
cur = conn.cursor()

cols_ddl = ", ".join(f"{c} DOUBLE PRECISION" for c in V)
cur.execute(
    "CREATE TABLE IF NOT EXISTS transaction_features ("
    "txn_id INT PRIMARY KEY REFERENCES Transactions(txn_id), "
    + cols_ddl +
    ", time_sec DOUBLE PRECISION, true_class SMALLINT, offline_prob DOUBLE PRECISION)")
cur.execute(
    "CREATE TABLE IF NOT EXISTS ml_scores ("
    "txn_id INT PRIMARY KEY REFERENCES Transactions(txn_id), "
    "fraud_probability NUMERIC(5,4) NOT NULL, anomaly_score NUMERIC(5,4), "
    "is_flagged BOOLEAN NOT NULL, scored_at TIMESTAMP DEFAULT NOW())")

cur.execute("SELECT COUNT(*) FROM transaction_features")
if cur.fetchone()[0] > 0:
    conn.commit()
    print("transaction_features already has rows, so the replay was done before. Nothing inserted.")
    sys.exit()

cur.execute("SELECT account_id FROM Accounts")
accounts = [r[0] for r in cur.fetchall()]
cur.execute("SELECT merchant_id FROM Merchants")
merchants = [r[0] for r in cur.fetchall()]

base = datetime(2026, 3, 1)
for i in range(len(sample)):
    ts = base + timedelta(hours=int(hours[i]), seconds=i)
    cur.execute(
        "INSERT INTO Transactions (account_id, merchant_id, amount, txn_time, location, status) "
        "VALUES (%s, %s, %s, %s, 'Pune', 'confirmed') RETURNING txn_id",
        (int(rng.choice(accounts)), int(rng.choice(merchants)), float(amounts[i]), ts))
    txn_id = cur.fetchone()[0]
    vals = [float(sample.loc[i, c.upper()]) for c in V]
    cur.execute(
        f"INSERT INTO transaction_features (txn_id, {', '.join(V)}, time_sec, true_class, offline_prob) "
        f"VALUES ({', '.join(['%s'] * 32)})",
        [txn_id] + vals + [float(hours[i] * 3600), int(labels[i]), float(offline_prob[i])])

conn.commit()
print("Inserted", len(sample), "transactions,", int(labels.sum()), "of them fraud")
cur.execute("SELECT flag_reason, COUNT(*) FROM Audit_Logs GROUP BY flag_reason ORDER BY 1")
print("Audit_Logs by reason so far:", cur.fetchall())
conn.close()
