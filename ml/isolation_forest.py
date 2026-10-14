import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score,
                             confusion_matrix)

X_train = pd.read_csv("ml/data/X_train.csv")
X_test = pd.read_csv("ml/data/X_test.csv")
y = pd.read_csv("ml/data/y_test.csv")["Class"].values

# Unsupervised: no labels are used for fitting.
# contamination = the share of data we expect to be anomalies.
iso = IsolationForest(n_estimators=100, contamination=0.0017,
                      random_state=42, n_jobs=-1)
iso.fit(X_train)

score = -iso.score_samples(X_test)              # higher = more anomalous
pred = (iso.predict(X_test) == -1).astype(int)  # -1 means anomaly

tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
print("Flagged:", int(pred.sum()), "of", len(pred))
print("TP", tp, "| FP", fp, "| FN", fn, "| TN", tn)
print("precision", round(precision_score(y, pred, zero_division=0), 4))
print("recall   ", round(recall_score(y, pred), 4))
print("f1       ", round(f1_score(y, pred), 4))
print("pr_auc   ", round(average_precision_score(y, score), 4))
print("roc_auc  ", round(roc_auc_score(y, score), 4))

print("\nMean anomaly score  legit:", round(score[y == 0].mean(), 4),
      "| fraud:", round(score[y == 1].mean(), 4))
k = int(y.sum())
top = score.argsort()[::-1][:k]
print("Fraud among the", k, "highest-scoring transactions:", int(y[top].sum()))

joblib.dump(iso, "ml/models/isolation_forest.joblib")
print("\nSaved ml/models/isolation_forest.joblib")
