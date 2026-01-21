# ==============================
# Fraud Detection using Isolation Forest
# Local Execution Version
# ==============================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ------------------------------
# 1. Load Dataset
# ------------------------------
DATA_PATH = "financial_anomaly_data.csv"

df = pd.read_csv(DATA_PATH)

print("\n✅ Dataset Loaded Successfully")
print(df.head())
print("\n📊 Dataset Info")
print(df.info())


# ------------------------------
# 2. Basic Validation
# ------------------------------
required_columns = ["Amount", "Timestamp", "TransactionType"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"❌ Missing required column: {col}")

df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")


# ------------------------------
# 3. Feature Selection
# ------------------------------
features = ["Amount"]
X = df[features]


# ------------------------------
# 4. Handle Missing Values
# ------------------------------
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)


# ------------------------------
# 5. Feature Scaling
# ------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)


# ------------------------------
# 6. Train Isolation Forest
# ------------------------------
model = IsolationForest(
    n_estimators=200,
    contamination=0.01,
    random_state=42,
    n_jobs=-1
)

model.fit(X_scaled)
print("\n🤖 Isolation Forest model trained")


# ------------------------------
# 7. Predict Anomalies
# ------------------------------
df["Fraudulent"] = model.predict(X_scaled)
df["AnomalyScore"] = model.decision_function(X_scaled)

print("\n🚨 Potential Fraud Count:", (df["Fraudulent"] == -1).sum())


# ------------------------------
# 8. Human-readable labels (CRITICAL FIX)
# ------------------------------
df["FraudulentLabel"] = df["Fraudulent"].map({
    -1: "Fraud",
     1: "Normal"
})


# ------------------------------
# 9. Feature Engineering
# ------------------------------
df["TransactionHour"] = df["Timestamp"].dt.hour
df["TransactionDay"] = df["Timestamp"].dt.day_name()


# ------------------------------
# 10. Monthly Analysis (June 2023)
# ------------------------------
june_data = df[
    (df["Timestamp"] >= "2023-06-01") &
    (df["Timestamp"] < "2023-07-01")
]

monthly_spend = june_data.groupby("TransactionType")["Amount"].sum()

print("\n💰 June Spend by Transaction Type")
print(monthly_spend)


# ------------------------------
# 11. Visualizations
# ------------------------------
sns.set(style="whitegrid")

# Anomaly scores over time
plt.figure(figsize=(12, 6))
sns.scatterplot(
    x="Timestamp",
    y="AnomalyScore",
    hue="FraudulentLabel",
    data=df,
    palette={"Fraud": "red", "Normal": "green"},
    alpha=0.6
)
plt.title("Anomaly Scores Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Distribution of anomaly scores
plt.figure(figsize=(10, 5))
sns.histplot(df["AnomalyScore"], bins=50, kde=True)
plt.title("Distribution of Anomaly Scores")
plt.tight_layout()
plt.show()


# Box plot
plt.figure(figsize=(8, 5))
sns.boxplot(
    x="FraudulentLabel",
    y="AnomalyScore",
    data=df,
    palette={"Fraud": "red", "Normal": "green"}
)
plt.title("Anomaly Scores by Transaction Class")
plt.tight_layout()
plt.show()


# Violin plot (FIXED)
plt.figure(figsize=(8, 5))
sns.violinplot(
    x="FraudulentLabel",
    y="AnomalyScore",
    data=df,
    palette={"Fraud": "red", "Normal": "green"},
    inner="quartile"
)
plt.title("Violin Plot of Anomaly Scores")
plt.tight_layout()
plt.show()


# Pair plot
sns.pairplot(
    df[["Amount", "AnomalyScore", "FraudulentLabel"]],
    hue="FraudulentLabel",
    palette={"Fraud": "red", "Normal": "green"}
)
plt.suptitle("Pair Plot: Amount vs Anomaly", y=1.02)
plt.show()


# ------------------------------
# 12. Export Results
# ------------------------------
df.to_csv("fraud_detection_output.csv", index=False)
print("\n📁 Results saved to fraud_detection_output.csv")

print("\n✅ Script execution completed successfully")

# ------------------------------
# 13. Extract Potential Fraud Cases
# ------------------------------
fraud_cases = df[df["Fraudulent"] == -1].copy()

# Sort by most suspicious first
fraud_cases = fraud_cases.sort_values("AnomalyScore")

print("\n🚨 Top 10 Most Suspicious Transactions")
print(
    fraud_cases[
        ["Timestamp", "TransactionType", "Amount", "AnomalyScore"]
    ].head(10)
)
fraud_cases.to_csv("potential_fraud_cases.csv", index=False)

print("\n📁 potential_fraud_cases.csv saved")
print("\n🚨 Top 5 Most Suspicious Transactions:")
print(
    fraud_cases[
        ["Timestamp", "TransactionType", "Amount", "AnomalyScore"]
    ].head(5)
)
