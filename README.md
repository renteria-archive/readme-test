---
title: Fraud Detection V1
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---
# 🛡️ Fraud Detection System - V1 (PaySim)

**End-to-end financial fraud detection using XGBoost, Streamlit, FastAPI & Docker**

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-red?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-V1_Complete-success?style=flat)

> **High-Recall Operational Fraud Detection Pipeline.**
>
> A production-ready Machine Learning system trained on the PaySim synthetic financial dataset. Designed to detect fraudulent mobile money transfers with a focus on modularity, reproducibility, and real-time inference.

---

## 📊 Executive Summary

The primary objective of V1 was to maximize **Recall** — capturing as many fraudulent transactions as possible — while maintaining an operationally manageable false positive rate.

**Key metrics:** Recall captures real frauds (higher → fewer losses). Precision measures alert quality (higher → fewer false alarms). PR-AUC measures overall fraud separation ability. Accuracy is discarded due to the 349:1 class imbalance.

### The Challenge: Extreme Imbalance

The dataset presents a severe imbalance — only 0.29% of transactions after filtering are fraudulent. Standard accuracy would reach 99.7% by predicting everything as legitimate, which is useless for fraud detection.

![Fig 1. The 349:1 ratio makes standard accuracy a misleading metric](https://github.com/renteria-luis/fraud-detection-v1/raw/main/assets/figures/class_distribution.png)

### Model Benchmarking (threshold = 0.2226 for all)

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.65 | 0.40 | 0.50 | 0.48 |
| Random Forest | 0.76 | 0.78 | 0.77 | 0.81 |
| **XGBoost (Default)** | **0.84** | **0.85** | **0.84** | **0.91** |
| XGBoost (Tuned) | 0.24 | 0.98 | 0.39 | 0.92 |

> XGBoost Tuned achieved the highest PR-AUC (0.92) but produced a distorted precision curve at low thresholds, making threshold selection unreliable. XGBoost Default was selected for its cleaner and more stable behavior.

### Production Performance (V1)

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.5 (default) | 0.93 | 0.79 | 0.85 |
| **0.2226 (operational)** | **0.84** | **0.85** | **0.84** |

- **PR-AUC:** `0.9079`
- **ROC-AUC:** `0.9285`
- **Operational Threshold:** `0.2226` — lowered from 0.5 to recover the 6% of fraud missed at default. In fraud detection, a missed fraud costs more than a false alarm.

![Fig 2. Confusion matrix at operational threshold 0.2226](https://github.com/renteria-luis/fraud-detection-v1/raw/main/assets/figures/confusion_matrix.png)

---

## 🔎 Key Data Insights (EDA)

**1. Fraud is type-specific.** Fraudulent activity occurs exclusively in `TRANSFER` (0.77% fraud rate) and `CASH_OUT` (0.18%) transactions. `PAYMENT`, `CASH_IN`, and `DEBIT` have zero fraud — they were removed from the dataset entirely to eliminate noise.

**2. Fraud is time-agnostic (bot behavior).** Legitimate transactions follow a human circadian cycle — clear peaks during business hours and a sharp drop at night. Fraud is flat and constant across all hours, consistent with automated bot activity. Transactions between 00:00–06:00 are disproportionately risky.

![Fig 3. Fraud is flat across all hours while legitimate traffic follows a circadian cycle](https://github.com/renteria-luis/fraud-detection-v1/raw/main/assets/figures/time_distribution.png)

**3. A simulation artifact was corrected.** Legitimate transactions drop sharply after day 17 of the simulation while fraud continues through day 30 — an artifact of the synthetic data generator. The dataset was truncated at the last step where legitimate traffic exists to prevent the model from learning a false "late-month = fraud" pattern.

---

## ⚙️ Feature Engineering

Raw columns are never fed directly to the model. The `src/features/engineering.py` module implements a custom `PaySimFeatures` sklearn transformer that fits on training data and transforms both splits cleanly.

**Behavioral aggregates** (fitted on train, applied to test):
- `dest_tx_count` — how many transactions has this destination received? Mule accounts accumulate many.
- `dest_unique_orig` — how many different senders? Mules receive from multiple sources.

**Transaction signals:**
- `is_transfer`, `is_cash_out` — type encoded as binary flags.
- `log_amount` — log-transformed amount to reduce right-skewness.
- `is_large_tx` — binary flag for amounts above the 95th percentile.
- `is_round_amount` — round amounts (mod 1000 = 0) are a weak but consistent signal.

**Temporal signals:**
- `hour_of_day` — derived from simulation step (step % 24).
- `is_night` — binary flag for hours 00–06.

**Balance signals** (intentional leakage — documented in notebook):
- `dest_was_empty` — destination account had $0 before receiving funds. Strong mule account signal.
- `log_dest_balance`, `amount_to_dest_ratio` — context around destination balance.
- `log_orig_balance` — origin account balance before transaction. Technically available pre-transaction, but artificially predictive in PaySim due to synthetic data patterns.

---

## 📁 Project Architecture

```
.
├── api/                        # FastAPI Application Layer
│   ├── main.py                 # Endpoints & Singleton Model Loader
│   └── schemas.py              # Pydantic Data Validation Schemas
├── app.py                      # Streamlit Interactive Demo
├── assets/figures/             # EDA & Model Evaluation Plots
├── data/                       # Data storage (gitignored)
├── docker-compose.yml          # Production Orchestration (FastAPI)
├── docker-compose.override.yml # Local Development (Hot-Reload)
├── Dockerfile                  # Multi-stage, Non-root, Slim Image
├── models/
│   ├── fraud_detection_v1_xgb.pkl  # Trained Pipeline
│   └── metadata_v1.json            # Training Metadata & Metrics
├── notebooks/
│   ├── 01_eda_paysim.ipynb     # Exploratory Data Analysis
│   └── 02_baseline_models.ipynb # Training, Evaluation & Threshold Analysis
├── params.yaml                 # Single Source of Truth for Config
├── src/
│   ├── config.py               # Feature lists, paths, constants
│   ├── data/                   # loader.py, splitter.py
│   ├── evaluation/             # metrics.py (PR curves, classification report)
│   ├── features/               # engineering.py (PaySimFeatures transformer)
│   ├── models/                 # builder.py (pipeline construction)
│   └── utils/                  # helpers.py
└── train.py                    # Standalone retraining script
```

---

## 🚀 Quick Start

### Option A: Docker

**Streamlit app (HuggingFace / local):**
```bash
docker build -t fraud-detection:v1 .
docker run -p 7860:7860 fraud-detection:v1
```

**FastAPI (local development):**
```bash
docker compose up --build
```

- Streamlit: `http://localhost:7860`
- API Health: `http://localhost:8000/health`
- Swagger: `http://localhost:8000/docs`

### Option B: Local (no Docker)

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📡 API Usage

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "step": 10,
       "type": "TRANSFER",
       "amount": 50000.0,
       "nameOrig": "C123456789",
       "oldbalanceOrg": 50000.0,
       "nameDest": "C987654321",
       "oldbalanceDest": 0.0
     }'
```

**Response:**
```json
{
  "fraud_probability": 0.9341,
  "is_fraud": true,
  "threshold_used": 0.2226,
  "version": "1.0.0"
}
```

---

## 📅 Roadmap

| Feature | V1 (Current) | V2 (Planned) |
|---|---|---|
| Dataset | PaySim (Synthetic) | Real transaction data |
| Algorithm | XGBoost (Default) | TBD — DL if dataset justifies it |
| Validation | Stratified random split | Out-of-time split |
| Explainability | Feature Importance | SHAP |
| Features | Static behavioral aggregates | Temporal velocity features |

V2 is contingent on finding a dataset with real transaction data. Further iteration on PaySim carries diminishing returns given its synthetic nature and single fraud pattern.

---

## 📓 Notebooks

### [`01_eda_paysim.ipynb`](notebooks/01_eda_paysim.ipynb) — Exploratory Data Analysis
- Class imbalance analysis (349:1)
- Transaction type fraud rates
- Temporal patterns (circadian cycle vs. flat fraud)
- Simulation artifact detection and correction (step 718 truncation)
- Cardinality analysis for nameOrig / nameDest
- Feature engineering motivation

### [`02_baseline_models.ipynb`](notebooks/02_baseline_models.ipynb) — Training & Evaluation
- Pipeline construction and feature sanity checks
- XGBoost training with `scale_pos_weight`
- Optuna hyperparameter search (100 trials, 5-fold CV)
- Threshold analysis and operational threshold selection
- Confusion matrix and PR curve analysis
- Model export and metadata generation

---

## 📬 Contact

**Luis Renteria**
*Machine Learning Engineer | Data Scientist*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/renteria-luis/)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:luis.renteria.dev@gmail.com)
