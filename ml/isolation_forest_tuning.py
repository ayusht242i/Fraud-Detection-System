import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

X_train = pd.read_csv("ml/data/X_train.csv")
X_test = pd.read_csv("ml/data/X_test.csv")
y = pd.read_csv("ml/data/y_test.csv")["Class"].values
k = int(y.sum())

def run(n_est, max_s, seed):
    iso = IsolationForest(n_estimators=n_est, max_samples=max_s,
                          random_state=seed, n_jobs=-1)
    iso.fit(X_train)
    score = -iso.score_samples(X_test)   # higher = more anomalous
    top = score.argsort()[::-1][:k]
    return (iso, score, average_precision_score(y, score),
            roc_auc_score(y, score), int(y[top].sum()))

rows = []
for n_est in [100, 300]:
    for max_s in [128, 256, 1024, 4096]:
        res = [run(n_est, max_s, s) for s in (0, 1, 2)]
        pr = [r[2] for r in res]
        ro = [r[3] for r in res]
        tk = [r[4] for r in res]
        rows.append({
            "n_estimators": n_est, "max_samples": max_s,
            "pr_auc": round(float(np.mean(pr)), 4),
            "pr_std": round(float(np.std(pr)), 4),
            "roc_auc": round(float(np.mean(ro)), 4),
            "fraud_in_top95": round(float(np.mean(tk)), 1),
        })

print(pd.DataFrame(rows).to_string(index=False))

best = max(rows, key=lambda r: r["pr_auc"])
print("\nBest by PR-AUC:", best["n_estimators"], "trees,", best["max_samples"], "samples per tree")

iso, score, pr, ro, tk = run(best["n_estimators"], best["max_samples"], 42)
print("\nCutoff table (flag the top X% most anomalous = what contamination controls):")
out = []
for rate in [0.0005, 0.001, 0.0017, 0.005, 0.01, 0.02]:
    cut = np.quantile(score, 1 - rate)
    pred = (score >= cut).astype(int)
    flagged = int(pred.sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    prec = tp / flagged if flagged else 0.0
    rec = tp / k
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    out.append({"flag_rate": rate, "flagged": flagged, "caught": tp,
                "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)})
print(pd.DataFrame(out).to_string(index=False))

joblib.dump(iso, "ml/models/isolation_forest.joblib")
print("\nSaved ml/models/isolation_forest.joblib")
