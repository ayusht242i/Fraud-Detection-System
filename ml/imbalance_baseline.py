import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score, roc_auc_score,
                             confusion_matrix)

X_train = pd.read_csv("ml/data/X_train.csv")
X_test = pd.read_csv("ml/data/X_test.csv")
y_train = pd.read_csv("ml/data/y_train.csv")["Class"]
y_test = pd.read_csv("ml/data/y_test.csv")["Class"]

models = {
    "1. Always predict legit (dummy)": DummyClassifier(strategy="most_frequent"),
    "2. Logistic Regression (plain)": LogisticRegression(max_iter=1000),
    "3. Logistic Regression (class_weight=balanced)": LogisticRegression(max_iter=1000, class_weight="balanced"),
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    rows.append({
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "pr_auc": round(average_precision_score(y_test, prob), 4),
        "roc_auc": round(roc_auc_score(y_test, prob), 4),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    })

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print(pd.DataFrame(rows).to_string(index=False))
