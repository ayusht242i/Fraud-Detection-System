import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, precision_score,
                             recall_score, f1_score, confusion_matrix)

X_train = pd.read_csv("ml/data/X_train.csv")
X_test = pd.read_csv("ml/data/X_test.csv")
y_train = pd.read_csv("ml/data/y_train.csv")["Class"].values
y_test = pd.read_csv("ml/data/y_test.csv")["Class"].values

def make_rf():
    return RandomForestClassifier(n_estimators=150, max_depth=16, n_jobs=-1, random_state=42)

def make_if():
    return IsolationForest(n_estimators=100, max_samples=4096, n_jobs=-1, random_state=42)

# Out-of-fold scores on TRAIN only
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
oof_rf = np.zeros(len(X_train))
oof_if = np.zeros(len(X_train))
for tr, va in cv.split(X_train, y_train):
    rf = make_rf().fit(X_train.iloc[tr], y_train[tr])
    oof_rf[va] = rf.predict_proba(X_train.iloc[va])[:, 1]
    iso = make_if().fit(X_train.iloc[tr])
    oof_if[va] = -iso.score_samples(X_train.iloc[va])
    print("fold done", flush=True)

# Final models on all of train
rf = make_rf().fit(X_train, y_train)
iso = make_if().fit(X_train)
te_rf = rf.predict_proba(X_test)[:, 1]
te_if = -iso.score_samples(X_test)

# Normalise the Isolation Forest score to [0,1] using train out-of-fold range
lo, hi = float(oof_if.min()), float(oof_if.max())
def norm(s):
    return np.clip((s - lo) / (hi - lo), 0, 1)
oof_if_n, te_if_n = norm(oof_if), norm(te_if)

# Choose the Isolation Forest weight on train CV
rows = []
for w_if in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    comp = w_if * oof_if_n + (1 - w_if) * oof_rf
    rows.append({"w_if": w_if, "w_rf": round(1 - w_if, 1),
                 "oof_pr_auc": round(average_precision_score(y_train, comp), 4),
                 "oof_roc_auc": round(roc_auc_score(y_train, comp), 4)})
print()
print(pd.DataFrame(rows).to_string(index=False))
best_w = max(rows, key=lambda r: r["oof_pr_auc"])["w_if"]
print("\nBest IF weight on train CV:", best_w)

def best_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    i = int(f1.argmax())
    return float(t[i]), float(f1[i])

def report(name, y, s, thr):
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(f"{name:<22} thr={thr:.3f} pr_auc={average_precision_score(y, s):.4f} "
          f"precision={precision_score(y, pred, zero_division=0):.4f} "
          f"recall={recall_score(y, pred):.4f} f1={f1_score(y, pred):.4f} "
          f"TP={tp} FP={fp} FN={fn}")

print("\nTEST SET (used once), cutoff chosen on train CV:")
final_thr = None
for name, wi in [("RF only", 0.0), ("Composite (chosen)", best_w)]:
    oof_c = wi * oof_if_n + (1 - wi) * oof_rf
    thr, _ = best_threshold(y_train, oof_c)
    te_c = wi * te_if_n + (1 - wi) * te_rf
    report(name, y_test, te_c, thr)
    if name.startswith("Composite"):
        final_thr = thr

joblib.dump(rf, "ml/models/random_forest.joblib", compress=3)
joblib.dump(iso, "ml/models/isolation_forest.joblib")
with open("ml/models/composite_config.json", "w") as f:
    json.dump({"w_if": best_w, "w_rf": round(1 - best_w, 4),
               "if_min": lo, "if_max": hi, "threshold": final_thr}, f, indent=2)
print("\nSaved models and ml/models/composite_config.json")
