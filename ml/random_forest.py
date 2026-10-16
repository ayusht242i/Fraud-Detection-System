import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix)

X_train = pd.read_csv("ml/data/X_train.csv")
X_test = pd.read_csv("ml/data/X_test.csv")
y_train = pd.read_csv("ml/data/y_train.csv")["Class"]
y_test = pd.read_csv("ml/data/y_test.csv")["Class"]

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

rows = []
for cw in [None, "balanced_subsample"]:
    for depth in [10, 16]:
        rf = RandomForestClassifier(n_estimators=150, max_depth=depth,
                                    class_weight=cw, n_jobs=-1, random_state=42)
        s = cross_val_score(rf, X_train, y_train, cv=cv, scoring="average_precision")
        rows.append({"class_weight": str(cw), "max_depth": depth,
                     "cv_pr_auc": round(s.mean(), 4), "cv_std": round(s.std(), 4)})
        print(rows[-1], flush=True)

print()
print(pd.DataFrame(rows).to_string(index=False))
best = max(rows, key=lambda r: r["cv_pr_auc"])
print("\nBest on cross-validation:", best)

cw = None if best["class_weight"] == "None" else best["class_weight"]
rf = RandomForestClassifier(n_estimators=150, max_depth=best["max_depth"],
                            class_weight=cw, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

prob = rf.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print("\nTEST SET (used once):")
print("pr_auc   ", round(average_precision_score(y_test, prob), 4))
print("roc_auc  ", round(roc_auc_score(y_test, prob), 4))
print("precision", round(precision_score(y_test, pred, zero_division=0), 4))
print("recall   ", round(recall_score(y_test, pred), 4))
print("f1       ", round(f1_score(y_test, pred), 4))
print("TP", tp, "| FP", fp, "| FN", fn, "| TN", tn)

imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nTop 10 features:")
print(imp.head(10).round(4).to_string())

joblib.dump(rf, "ml/models/random_forest.joblib", compress=3)
print("\nSaved ml/models/random_forest.joblib")
