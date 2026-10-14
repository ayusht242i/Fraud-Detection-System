import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("ml/data/creditcard.csv")
os.makedirs("ml/plots", exist_ok=True)

print("Shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

print("\nClass counts:")
print(df["Class"].value_counts())
print("\nClass percentages:")
print((df["Class"].value_counts(normalize=True) * 100).round(4))

print("\nAmount, all transactions:")
print(df["Amount"].describe().round(2))
print("\nAmount by class (0 = legit, 1 = fraud):")
print(df.groupby("Class")["Amount"].describe().round(2))

ax = df["Class"].value_counts().plot(kind="bar", color=["#1F4E79", "#B5651D"])
ax.set_yscale("log")
ax.set_xticklabels(["Legit (0)", "Fraud (1)"], rotation=0)
ax.set_title("Class distribution (log scale)")
plt.savefig("ml/plots/class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

plt.hist(df[df.Class == 0]["Amount"].clip(upper=500), bins=50, alpha=0.6, label="Legit", density=True)
plt.hist(df[df.Class == 1]["Amount"].clip(upper=500), bins=50, alpha=0.6, label="Fraud", density=True)
plt.xlabel("Amount (clipped at 500)")
plt.legend()
plt.title("Transaction amount by class")
plt.savefig("ml/plots/amount_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

df["hour"] = (df["Time"] // 3600) % 24
plt.hist(df[df.Class == 0]["hour"], bins=24, alpha=0.6, label="Legit", density=True)
plt.hist(df[df.Class == 1]["hour"], bins=24, alpha=0.6, label="Fraud", density=True)
plt.xlabel("Hour of day")
plt.legend()
plt.title("Transactions by hour of day")
plt.savefig("ml/plots/hour_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nPlots saved to ml/plots/")
