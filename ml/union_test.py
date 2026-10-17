import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import StratifiedKFold

X_train = pd.read_csv("ml/data/X_train.csv")
X_test = pd.read_csv("ml/data/X_test.csv")
y_train = pd.read_csv("ml/data/y_train.csv")["Class"].values
y_test = pd.read_csv("ml/data/y_test.csv")["Class"].values

cfg = json.load(open("ml/models/composite_config.json"))
thr = cfg["threshold"]

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
oof_rf = np.zeros(len(X_train))
oof_if = np.zeros(len(X_train))
for tr, va in cv.split(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=150, max_depth=16, n_jobs=-1, random_state=42)
    rf.fit(X_train.iloc[tr], y_train[tr])
    oof_rf[va] = rf.predict_proba(X_train.iloc[va])[:, 1]
    iso = IsolationForest(n_estimators=100, max_samples=4096, n_jobs=-1, random_state=42)
    iso.fit(X_train.iloc[tr])
    oof_if[va] = -iso.score_samples(X_train.iloc[va])
    print("fold done", flush=True)

def evaluate(y, rf_s, if_s, cut):
    rf_flag = rf_s >= thr
    if_flag = (if_s >= cut) if cut is not None else np.zeros(len(y), dtype=bool)
    flag = rf_flag | if_flag
    pos = (y == 1)
    tp = int((flag & pos).sum())
    fp = int((flag & ~pos).sum())
    fn = int((~flag & pos).sum())
    return {"TP": tp, "FP": fp, "FN": fn,
            "extra_TP_from_IF": int((if_flag & ~rf_flag & pos).sum()),
            "extra_FP_from_IF": int((if_flag & ~rf_flag & ~pos).sum()),
            "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 4),
            "f1": round(2 * tp / (2 * tp + fp + fn), 4)}

rows = []
for rate in [0, 0.0005, 0.001, 0.002, 0.005]:
    cut = float(np.quantile(oof_if, 1 - rate)) if rate > 0 else None
    r = evaluate(y_train, oof_rf, oof_if, cut)
    r["if_flag_rate"] = rate
    rows.append(r)
print()
print("TRAIN (out-of-fold): flag if RF >= cutoff OR Isolation Forest in top X%")
print(pd.DataFrame(rows).to_string(index=False))
best = max(rows, key=lambda r: r["f1"])
print("\nBest on train CV by F1: IF flag rate =", best["if_flag_rate"])

rf = joblib.load("ml/models/random_forest.joblib")
iso = joblib.load("ml/models/isolation_forest.joblib")
te_rf = rf.predict_proba(X_test)[:, 1]
te_if = -iso.score_samples(X_test)
tr_if = -iso.score_samples(X_train)

print("\nTEST SET (used once):")
out = []
for label, rate in [("RF only", 0), ("RF + IF union", best["if_flag_rate"])]:
    cut = float(np.quantile(tr_if, 1 - rate)) if rate > 0 else None
    r = evaluate(y_test, te_rf, te_if, cut)
    r["setup"] = label
    out.append(r)
print(pd.DataFrame(out).to_string(index=False))

cfg["union_if_rate"] = best["if_flag_rate"]
cfg["union_if_cut"] = float(np.quantile(tr_if, 1 - best["if_flag_rate"])) if best["if_flag_rate"] > 0 else None
with open("ml/models/composite_config.json", "w") as f:
    json.dump(cfg, f, indent=2)
print("\nUpdated ml/models/composite_config.json")
