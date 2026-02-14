# 🛡️ Fraud Detection System - V1

**End-to-end credit card fraud detection using XGBoost, FastAPI & Docker**

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-red?style=flat)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-V1_Complete-success?style=flat)
> **High-Recall Operational Fraud Detection Pipeline.**
> 
> A production-ready Machine Learning system designed to identify fraudulent transactions in highly imbalanced datasets ($598:1$). Engineered with a focus on modularity, reproducibility, and real-time inference using Docker and FastAPI.

---

## 📊 Executive Summary & Model Selection

The primary objective of V1 was to minimize financial loss by maximizing **Recall** (capturing as many frauds as possible) while maintaining an operationally manageable False Positive Rate.

### The Challenge: Extreme Imbalance

The dataset presents a severe imbalance (0.17% Fraud vs 99.83% Legitimate), requiring specialized techniques like `scale_pos_weight` in XGBoost rather than standard accuracy metrics.

![Fig 1. The 598:1 ratio makes standard accuracy a misleading metric](https://github.com/renteria-luis/fraud-detection-v1/raw/main/assets/figures/class_distribution.png)

| Metric    | What It Measures | If It Increases     | If It Decreases     | Priority      |
|-----------|-----------------|-------------------|-------------------|---------------|
| Recall    | Real frauds      | Catch more fraud  | Miss more fraud   | #1 CRITICAL   |
| Precision | True alerts      | Fewer false alarms | More false alarms | #2 MAJOR  |
| PR-AUC    | Fraud separation | Better detection  | Worse detection   | #3 KEY        |
| F1-Score  | Precision+Recall | Redundant         | -                 | NOT USED      |
| ROC-AUC   | Class separability | PR-AUC better   | -                 | NOT USED      |
| Accuracy  | Overall correct  | Misleading        | -                 | NOT USED      |


---

### Model Benchmarking

Three architectures were evaluated during the experimentation phase. **XGBoost** was selected as the production model due to its superior handling of class imbalance via `scale_pos_weight` and inference speed.

| **Model**               | **Recall** | **Precision** | **PR-AUC** | **Verdict**                                  |
|-------------------------|------------|---------------|------------|---------------------------------------------|
| Logistic Regression      | 0.82       | 0.15          | 0.65       | Baseline. Many false positives              |
| Random Forest            | 0.78       | **0.88**      | 0.84       | High precision, missed crucial fraud        |
| XGBoost (Tuned)          | **0.87**   | 0.82          | **0.88**   | **Selected.** Best Recall/Precision balance|


### Production Performance (V1)

- **Operational Threshold:** `0.2072` (Optimized for F2-Score/Recall).
- **Business Impact:** The model captures **87%** of fraudulent transactions.
- **Latency:** ~10ms per transaction via FastAPI.

---
## 🔎 Key Data Insights (EDA)

Before modeling, an extensive exploratory analysis revealed critical patterns used for feature selection.

1. Feature Separation:
Variables V14, V10, and V12 showed the strongest discriminative power. As seen below, V14 provides a clear (though not perfect) separation boundary between classes compared to other features.

![Fig 2. Scatter plot showing V14 vs Top 3 correlated features. Fraud points are distinct outliers](https://github.com/renteria-luis/fraud-detection-v1/raw/main/assets/figures/scatter_v14_vs_top3.png)

---
## ⚙️ Feature Engineering Strategy

Raw data is never enough. The `src/features` module implements custom Scikit-learn transformers to extract signal from noise:

1. **Temporal Cyclical Encoding**
EDA revealed a distinct pattern: Fraudulent activity remains consistent during the night, while legitimate transactions drop drastically.

![Fig 3. Density plot showing the "Night Valley" where legitimate traffic drops, but fraud persists](https://github.com/renteria-luis/fraud-detection-v1/raw/main/assets/figures/time_distribution.png)

Based on this, we engineered:

  - **Cyclical Encoding:** Converted `Time` to Sine/Cosine components $(sin(2πt/24))$ to preserve 24h continuity.
  - **Is Night Flag:** Binary feature for transactions between 22:00–06:00.
    
2. **Amount Scaling**
  - **Log Transformation:** Applied $\log(1 + x)$ to `Amount` to handle extreme right-skewness.
  - **Micro/Macro Flags:** Binary features for very small $(<\$1)$ or large $(>95th percentile)$ transactions.

3. **Anonymized Features:**
  - `V1`...`V28` (PCA components) were left otherwise untouched to preserve their principal component properties.

---

## 📁 Project Architecture

This repository follows strict **MLOps principles**. Code is modularized into a source package (`src`) rather than living in notebooks.

> **Note on Language Stats:** You might notice GitHub reports this repo as ~94% Python. Jupyter Notebooks are explicitly marked as documentation in `.gitattributes` to reflect the engineering effort put into the `.py` source code.

```
.
├── api/                       # FastAPI Application Layer
│   ├── main.py                # Endpoints & Singleton Model Loader
│   └── schemas.py             # Pydantic Data Validation Schemas
├── data/                      # Data storage (gitignored)
├── docker-compose.yml         # Production Orchestration (Base)
├── docker-compose.override.yml# Local Development (Hot-Reload)
├── Dockerfile                 # Multi-stage, Non-root, Slim Image
├── models/                    # Serialized Artifacts
│   ├── fraud_detection_v1_xgb.pkl  # The Trained Pipeline
│   └── metadata_v1.json       # Training Metadata
├── notebooks/                 # Experimentation & Analysis
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   └── 02_baseline_models.ipynb # Model Training & Evaluation
├── params.yaml                # Single Source of Truth for Config
├── src/                       # Core Logic Package
│   ├── evaluation/            # Metrics & Visualization logic
│   ├── features/              # Custom Transformers (FE)
│   ├── models/                # Training Pipelines (sklearn/xgb)
│   └── utils/                 # Helpers
└── requirements.txt           # Dependencies
```

---

## 🚀 Quick Start

### Option A: Docker (Recommended)

Run the entire system (API + Environment) without installing Python locally.

```bash
# 1. Clone the repository
git clone https://github.com/renteria-luis/fraud-detection-v1.git
cd fraud-detection-v1

# 2. Build and Run
docker compose up --build
```

- **API Health Check:** `http://localhost:8000/health`
- **Interactive Docs (Swagger):** `http://localhost:8000/docs`
  

### Option B: Local Development

To run locally with **hot-reloading** enabled (changes in `src/` reflect immediately):

```bash
# Uses docker-compose.override.yml automatically
docker compose up
```

---

## 📡 API Usage Example

Once the container is running, you can detect fraud via `curl` or Python:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"Time":1000.0,"Amount":150.0,"V1":-1.3,"V2":1.1,"V3":-0.5,"V4":0.3,"V5":0.2,"V6":-0.1,"V7":0.5,"V8":0.2,"V9":-0.4,"V10":0.1,"V11":-0.5,"V12":0.3,"V13":0.1,"V14":-0.2,"V15":0.4,"V16":-0.3,"V17":0.2,"V18":0.1,"V19":-0.1,"V20":0.1,"V21":0.2,"V22":-0.1,"V23":0.1,"V24":0.1,"V25":-0.2,"V26":0.1,"V27":0.1,"V28":-0.1}'

```
**Response:**

```json
{
  "fraud_probability": 7.73e-07,
  "is_fraud": false,
  "threshold_used": 0.2072,
  "model_version": "1.0.0"
}
```

---

## 🔮 Roadmap: V1 vs V2

This project is evolving. V1 (Current) established a robust classical ML baseline with XGBoost. **V2 (Planned)** will introduce Deep Learning to capture more complex patterns and interactions that may be missed by tree-based models.

| **Feature** | **V1 (Current)** | **V2 (Planned)** |
| --- | --- | --- |
| **Algorithm** | XGBoost (eXtreme Gradient Boosting) | Deep Neural Network (PyTorch) |
| **Loss Function** | Binary cross-entropy (Weighted) | Focal Loss (Hard Example Mining) |
| **Explainability** | Feature Importance | SHAP (DeepExplainer) |
| **Compute** | CPU Optimized | GPU Accelerated (CUDA) |

---

## 📓 Notebooks Guide

### [`01_eda.ipynb`](notebooks/01_eda.ipynb) - Exploratory Data Analysis
**Key Findings:**
- Severe class imbalance (598:1 ratio)
- Time exhibits clear day/night patterns
- Amount is highly right-skewed
- V14, V12, V10 are most correlated with fraud

**Outputs:** Distribution plots, correlation heatmap, temporal analysis

### [`02_baseline_models.ipynb`](notebooks/02_baseline_models.ipynb) - Model Training & Evaluation
**Contents:**
1. Feature engineering implementation
2. Pipeline construction (preprocessing + model)
3. Model comparison (LogReg, RF, XGBoost)
4. Hyperparameter tuning (RandomizedSearchCV)
5. Threshold optimization for production
6. Model export & metadata generation

**Outputs:** Trained model (.pkl), metrics, confusion matrix, feature importance

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## 📬 Contact

**Luis Renteria**  

*Machine Learning Engineer | Data Scientist*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/renteria-luis/) 
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:luis.renteria.dev@gmail.com)
