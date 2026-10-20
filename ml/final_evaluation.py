import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve)

Path("ml/results").mkdir(parents=True, exist_ok=True)
Path("ml/plots").mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv("ml/data/X_train.csv")
X_test = pd.read_csv("ml/data/X_test.csv")
y_train = pd.read_csv("ml/data/y_train.csv")["Class"].values
y = pd.read_csv("ml/data/y_test.csv")["Class"].values

THR = json.load(open("ml/models/composite_config.json"))["threshold"]
rf = joblib.load("ml/models/random_forest.joblib")
iso = joblib.load("ml/models/isolation_forest.joblib")

lr_plain = LogisticRegression(max_iter=1000).fit(X_train, y_train)
lr_bal = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train, y_train)

iso_test = -iso.score_samples(X_test)
iso_cut = float(np.quantile(-iso.score_samples(X_train), 1 - 0.0017))  # flag the top 0.17% most anomalous

scores = {
    "Logistic Regression (plain)": (lr_plain.predict_proba(X_test)[:, 1], 0.5),
    "Logistic Regression (balanced)": (lr_bal.predict_proba(X_test)[:, 1], 0.5),
    "Isolation Forest (unsupervised)": (iso_test, iso_cut),
    "Random Forest (final model)": (rf.predict_proba(X_test)[:, 1], THR),
}


def counts(y, s, thr):
    pred = s >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    return tp, fp, fn, tn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

rows = []
for name, (s, thr) in scores.items():
    tp, fp, fn, tn = counts(y, s, thr)
    p, r, f = prf(tp, fp, fn)
    rows.append({"model": name, "cutoff": round(float(thr), 4),
                 "pr_auc": round(average_precision_score(y, s), 4),
                 "roc_auc": round(roc_auc_score(y, s), 4),
                 "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                 "TP": tp, "FP": fp, "FN": fn, "TN": tn})
results = pd.DataFrame(rows)
results.to_csv("ml/results/final_results.csv", index=False)
print("FINAL RESULTS (held-out test set, %d fraud cases)" % int(y.sum()))
print(results.to_string(index=False))

rf_s = scores["Random Forest (final model)"][0]
sens = []
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    tp, fp, fn, tn = counts(y, rf_s, t)
    p, r, f = prf(tp, fp, fn)
    sens.append({"cutoff": t, "flagged": tp + fp, "TP": tp, "FP": fp, "FN": fn,
                 "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)})
sens = pd.DataFrame(sens)
sens.to_csv("ml/results/threshold_sensitivity.csv", index=False)
print("\nRANDOM FOREST: CUTOFF SENSITIVITY (illustration only, the chosen cutoff was fixed earlier)")
print(sens.to_string(index=False))

# Bootstrap error bars: resample the test set 500 times
rng = np.random.default_rng(42)
n = len(y)
pred_rf = rf_s >= THR
boot = {"pr_auc": [], "precision": [], "recall": []}
for _ in range(500):
    i = rng.integers(0, n, n)
    yb, sb, pb = y[i], rf_s[i], pred_rf[i]
    boot["pr_auc"].append(average_precision_score(yb, sb))
    tp = int((pb & (yb == 1)).sum())
    fp = int((pb & (yb == 0)).sum())
    fn = int((~pb & (yb == 1)).sum())
    boot["precision"].append(tp / (tp + fp) if tp + fp else 0.0)
    boot["recall"].append(tp / (tp + fn) if tp + fn else 0.0)
print("\nRANDOM FOREST: 95% bootstrap intervals (500 resamples of the test set)")
for k, v in boot.items():
    lo, hi = np.percentile(v, [2.5, 97.5])
    print(f"  {k:<10} {np.mean(v):.4f}   [{lo:.4f}, {hi:.4f}]")

# Figure 1: precision-recall curves
plt.figure(figsize=(7, 5))
for name, (s, thr) in scores.items():
    p, r, _ = precision_recall_curve(y, s)
    plt.plot(r, p, label=f"{name} (PR-AUC {average_precision_score(y, s):.2f})")
plt.axhline(y.mean(), color="grey", linestyle=":", label=f"No-skill baseline ({y.mean():.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-recall curves on the held-out test set")
plt.legend(fontsize=8, loc="upper right")
plt.grid(alpha=0.3)
plt.savefig("ml/plots/pr_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 2: precision and recall by cutoff
ts = np.linspace(0.05, 0.95, 37)
ps, rs = [], []
for t in ts:
    tp, fp, fn, tn = counts(y, rf_s, t)
    p, r, f = prf(tp, fp, fn)
    ps.append(p)
    rs.append(r)
plt.figure(figsize=(7, 4.5))
plt.plot(ts, ps, label="Precision")
plt.plot(ts, rs, label="Recall")
plt.axvline(THR, color="grey", linestyle="--", label=f"Chosen cutoff ({THR:.3f})")
plt.xlabel("Probability cutoff")
plt.title("Random Forest: precision and recall by cutoff (test set)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("ml/plots/threshold_tradeoff.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 3: top features
imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values().tail(15)
plt.figure(figsize=(6, 5))
imp.plot(kind="barh", color="#1F4E79")
plt.xlabel("Importance")
plt.title("Random Forest: top 15 features")
plt.savefig("ml/plots/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved ml/results/*.csv and ml/plots/pr_curves.png, threshold_tradeoff.png, feature_importance.png")
