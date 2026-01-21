Fraud Detection using Isolation Forest (Unsupervised)
📌 Project Overview

This project demonstrates an unsupervised fraud detection pipeline for financial transactions using Isolation Forest.
The goal is to identify high-risk anomalous transactions from unlabeled transactional data and prioritize them for further investigation.

⚠️ Note: This system flags potential fraud candidates, not confirmed fraud.

🎯 Problem Statement

In real-world financial systems:

Fraud labels are often unavailable or delayed

Manual review of all transactions is infeasible

Businesses need a triage mechanism to surface suspicious cases

This project addresses that by:

Learning normal transaction behavior

Identifying statistically rare deviations

Reducing review volume to the top 1% most suspicious transactions

🧠 Solution Approach

We use Isolation Forest, an unsupervised anomaly detection algorithm that:

Randomly partitions data

Isolates anomalies faster than normal observations

Does not require labeled fraud data

Why Isolation Forest?

Works well with skewed distributions

Scales efficiently

Suitable for real-world fraud screening pipelines

🗂️ Project Structure
fraud_detection_project/
│
├── fraud_detection.py              # Main pipeline
├── generate_data.py                # Synthetic data generator
├── financial_anomaly_data.csv      # Generated transaction data
├── potential_fraud_cases.csv       # Flagged anomalies (output)
├── fraud_detection_output.csv      # Full dataset with scores
└── README.md

📊 Dataset Description

The dataset is synthetically generated to mimic real-world digital payments.

Columns
Column	Description
Timestamp	Date & time of transaction
TransactionType	UPI / Card / NetBanking / Wallet
Amount	Transaction amount
Fraudulent	Model output (-1 anomaly, 1 normal)
AnomalyScore	Degree of abnormality
FraudulentLabel	Human-readable label
TransactionHour	Hour of transaction
TransactionDay	Day of week

Real financial data is sensitive, so synthetic data with injected anomalies is used for experimentation and validation.

⚙️ How the Pipeline Works

Data Generation

Simulates realistic transaction patterns

Injects rare, high-value anomalies

Preprocessing

Missing value imputation

Feature scaling (StandardScaler)

Model Training

Isolation Forest trained on transaction amounts

Anomaly Scoring

Each transaction assigned an anomaly score

Bottom 1% flagged as anomalous (-1)

Analysis & Visualization

Time-based anomaly plots

Distribution, box, violin, and pair plots

Export

potential_fraud_cases.csv for review

Full annotated dataset for analysis

🚀 How to Run the Project
1️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

2️⃣ Install Dependencies
pip install pandas scikit-learn matplotlib seaborn

3️⃣ Generate Dataset
python generate_data.py

4️⃣ Run Fraud Detection
python fraud_detection.py

📁 Outputs Explained
🔴 potential_fraud_cases.csv

Contains only flagged anomalous transactions, sorted by:

Lowest anomaly score first (most suspicious)

Used for:

Manual review

Rule-based validation

Downstream systems

🟢 fraud_detection_output.csv

Full dataset with:

Anomaly scores

Labels

Engineered features

📈 Visualizations

The project includes:

Anomaly scores over time

Score distribution

Box & violin plots (Normal vs Fraud)

Pair plots (Amount vs AnomalyScore)

These plots validate that anomalies are:

Statistically distinct

Not random noise

⚠️ Important Notes

-1 ≠ confirmed fraud

-1 = statistically rare / anomalous

This is a screening layer, not a final decision system

In production:

Flagged cases → Rules engine → Human review → Confirmation

🧪 Limitations

Uses only transaction amount as a feature

No ground-truth fraud labels

Synthetic data (not real banking data)

🔮 Future Improvements

Add customer & merchant features

Introduce time-window behavior analysis

Combine with supervised models (if labels available)

Deploy as a REST API (FastAPI)

Add business cost-based threshold tuning

🧠 Key Learnings

How unsupervised models fit into fraud systems

Importance of interpretability in anomaly detection

Difference between anomaly ≠ fraud

Debugging visualization & data type issues

Designing ML pipelines for real-world constraints

🏁 Final Takeaway

This project demonstrates how anomaly detection can significantly reduce fraud investigation effort by prioritizing high-risk transactions when labels are unavailable.
