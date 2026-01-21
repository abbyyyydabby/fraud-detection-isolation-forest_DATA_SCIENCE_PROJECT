import pandas as pd
import numpy as np

np.random.seed(42)

# Number of transactions
N = 5000

# Generate timestamps (random dates in 2023)
timestamps = pd.to_datetime(
    np.random.choice(
        pd.date_range("2023-01-01", "2023-12-31", freq="H"),
        size=N
    )
)

# Transaction types
transaction_types = np.random.choice(
    ["UPI", "Card", "NetBanking", "Wallet"],
    size=N,
    p=[0.45, 0.30, 0.15, 0.10]
)

# Normal transaction amounts
amounts = np.random.normal(loc=2000, scale=800, size=N)
amounts = np.abs(amounts)

# Inject fraud-like anomalies (very high amounts)
fraud_indices = np.random.choice(N, size=int(0.01 * N), replace=False)
amounts[fraud_indices] *= np.random.uniform(8, 15)

# Build DataFrame
df = pd.DataFrame({
    "Timestamp": timestamps,
    "TransactionType": transaction_types,
    "Amount": amounts.round(2)
})

# Save CSV
df.to_csv("financial_anomaly_data.csv", index=False)

print("✅ financial_anomaly_data.csv generated successfully")
print(df.head())
