import json
import joblib
import numpy as np
import pandas as pd

X = pd.read_csv("ml/data/X_test.csv")
y = pd.read_csv("ml/data/y_test.csv")["Class"].values
scaler = joblib.load("ml/models/scaler.joblib")
rf = joblib.load("ml/models/random_forest.joblib")
thr = json.load(open("ml/models/composite_config.json"))["threshold"]

prob = rf.predict_proba(X)[:, 1]

def payload(i):
    row = X.iloc[i]
    back = scaler.inverse_transform(
        pd.DataFrame([[row["Amount"], row["hour"]]], columns=["Amount", "hour"]))[0]
    p = {f"V{k}": float(row[f"V{k}"]) for k in range(1, 29)}
    p["Amount"] = round(float(back[0]), 2)
    p["Time"] = float(round(back[1]) * 3600)
    return p

caught = int(np.where((y == 1) & (prob >= thr))[0][0])
missed = int(np.where((y == 1) & (prob < thr))[0][0])
legit = int(np.where((y == 0) & (prob < thr))[0][0])

out = {}
for name, i in [("fraud_caught_by_model", caught),
                ("fraud_missed_by_model", missed),
                ("legit_not_flagged", legit)]:
    out[name] = {"true_label": int(y[i]),
                 "model_probability": round(float(prob[i]), 4),
                 "payload": payload(i)}
    print(name, "| true label:", int(y[i]), "| model probability:", round(float(prob[i]), 4))

with open("ml/models/sample_rows.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved ml/models/sample_rows.json")
