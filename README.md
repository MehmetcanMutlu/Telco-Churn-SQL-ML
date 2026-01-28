# Telco Churn Prediction with Advanced SQL & CatBoost

This project implements an end-to-end churn prediction pipeline. It simulates a telecom company's relational database to perform feature engineering using **SQL Window Functions** and trains a **CatBoost** classifier to identify high-risk customers.

The primary goal is to demonstrate the ability to handle data transformation at the database level (ELT) rather than relying solely on in-memory processing, followed by state-of-the-art machine learning modeling.

## 📋 Project Overview

1.  **Data Simulation:** Generates a relational database (SQLite) with realistic patterns for customers, call logs, and complaints.
2.  **SQL Feature Engineering:** Uses CTEs, Aggregations, and Window Functions (e.g., calculating trend changes over time) to extract features directly from the raw database tables.
3.  **Machine Learning:** Trains a CatBoost Classifier to predict customer churn, optimizing for Recall to capture as many potential churners as possible.

## 🛠 Tech Stack

* **Language:** Python 3.x
* **Database:** SQLite
* **Data Manipulation:** SQL (Window Functions, Joins), Pandas
* **Machine Learning:** CatBoost, Scikit-learn
* **Version Control:** Git

## 📊 Key Results

The model achieved high performance in distinguishing between loyal and churning customers.

* **ROC-AUC Score:** 0.9367
* **Recall (Churn Class):** 0.78 (Correctly identified 78% of leaving customers)
* **Precision (Churn Class):** 0.70

**Top Predictive Features:**
1.  `calls_last_30_days` (Derived via SQL: Significant drop in usage)
2.  `total_complaints` (Customer dissatisfaction signal)
3.  `total_calls` (General usage volume)

## 📂 Project Structure

```text
├── data/                  # Stores generated database and model artifacts
├── src/
│   ├── db_generator.py    # Simulates customers, calls, and complaints data
│   ├── feature_store.py   # Extracts features using complex SQL queries
│   └── train_model.py     # Trains, evaluates, and saves the CatBoost model
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation