import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("ml/data/creditcard.csv")
print("Rows before dedupe:", len(df), "| fraud:", int(df["Class"].sum()))

df = df.drop_duplicates().reset_index(drop=True)
print("Rows after dedupe: ", len(df), "| fraud:", int(df["Class"].sum()))

# Time is seconds since the first transaction (about 2 days), so this is an approximate hour of day
df["hour"] = (df["Time"] // 3600) % 24
df = df.drop(columns=["Time"])

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Fit the scaler on TRAIN only, then apply it to test (prevents data leakage)
cols = ["Amount", "hour"]
scaler = StandardScaler()
X_train = X_train.copy()
X_test = X_test.copy()
X_train[cols] = scaler.fit_transform(X_train[cols])
X_test[cols] = scaler.transform(X_test[cols])

print("\nTrain:", X_train.shape, "| fraud %:", round(y_train.mean() * 100, 4))
print("Test: ", X_test.shape, "| fraud %:", round(y_test.mean() * 100, 4))
print("Train fraud count:", int(y_train.sum()), "| Test fraud count:", int(y_test.sum()))
print("\nScaled Amount (train) mean/std:", round(X_train["Amount"].mean(), 4), round(X_train["Amount"].std(), 4))

os.makedirs("ml/models", exist_ok=True)
X_train.to_csv("ml/data/X_train.csv", index=False)
X_test.to_csv("ml/data/X_test.csv", index=False)
y_train.to_csv("ml/data/y_train.csv", index=False)
y_test.to_csv("ml/data/y_test.csv", index=False)
joblib.dump(scaler, "ml/models/scaler.joblib")
print("\nSaved splits to ml/data/ and scaler to ml/models/")
