# 🛡️ End-to-End Credit Card Fraud Detection System (V1)

> **High-Recall Operational Fraud Detection Pipeline.**
> 
> A production-ready Machine Learning system designed to identify fraudulent transactions in highly imbalanced datasets ($598:1$). Engineered with a focus on modularity, reproducibility, and real-time inference using Docker and FastAPI.

---

## 📊 Executive Summary & Model Selection

The primary objective of V1 was to minimize financial loss by maximizing **Recall** (capturing as many frauds as possible) while maintaining an operationally manageable False Positive Rate.

### Fraud Detection Metrics

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
- **Latency:** <50ms per transaction (p99) via FastAPI container.

---

## ⚙️ Feature Engineering Strategy

Raw data is never enough. The `src/features` module implements custom Scikit-learn transformers to extract signal from noise:

1. **Temporal Cyclical Encoding**
  
  - The `Time` variable (seconds elapsed) is converted into **hours of the day**.
  - Transformed into **sine/cosine** components to preserve the cyclical nature of time (23:00 is close to 00:00).
  - *Logic:* 
    $$
    \text{hour\_sin} = \sin\left(\frac{2\pi \cdot t}{24}\right),
    \quad
    \text{hour\_cos} = \cos\left(\frac{2\pi \cdot t}{24}\right)
    $$
    
2. **Amount Scaling & Flagging**
  - **Log Transformation:** Applied $\log(1 + x)$ to `Amount` to handle extreme right-skewness.
  - **Micro/Macro Flags:** Binary features for very small (`<1`) or large (`>95th percentile`) transactions.
  - **Night Transaction Flag:** Binary feature for transactions between 22:00–06:00.
    
3. **Anonymized Features:**
  
  - `V1`...`V28` (PCA components) were standardized but left otherwise untouched to preserve their principal component properties.

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
     -d '{           "Time": 1000.0,           "Amount": 150.0,           "V1": -1.3, "V2": 1.1, "V3": -0.5, "V4": 0.3, "V5": 0.2,           "V6": -0.1, "V7": 0.5, "V8": 0.2, "V9": -0.4, "V10": 0.1,           "V11": -0.5, "V12": 0.3, "V13": 0.1, "V14": -0.2, "V15": 0.4,           "V16": -0.3, "V17": 0.2, "V18": 0.1, "V19": -0.1, "V20": 0.1,           "V21": 0.2, "V22": -0.1, "V23": 0.1, "V24": 0.1, "V25": -0.2,           "V26": 0.1, "V27": 0.1, "V28": -0.1         }'
```

**Response:**

```json
{
  "fraud_probability": 0.89,
  "is_fraud": true,
  "threshold_used": 0.2072,
  "model_version": "1.0.0"
}
```

---

## 🔮 Roadmap: V1 vs V2

This project is evolving. **V1 (Current)** established a robust classical ML baseline. **V2 (Planned)** will introduce Deep Learning to capture non-linear patterns that XGBoost misses.

| **Feature** | **V1 (Current)** | **V2 (Planned)** |
| --- | --- | --- |
| **Algorithm** | XGBoost (Gradient Boosting) | Deep Neural Network (PyTorch) |
| **Loss Function** | LogLoss (Weighted) | Focal Loss (Hard Example Mining) |
| **Explainability** | Feature Importance | SHAP (DeepExplainer) |
| **Compute** | CPU Optimized | GPU Accelerated (CUDA) |
| **Infrastructure** | Docker Compose | Kubernetes / Cloud Run |

---

## 📬 Contact

**Luis Renteria**

*Machine Learning Engineer | Data Scientist*

[LinkedIn](https://www.linkedin.com/in/renteria-luis/) | [Email](mailto:luis.renteria.dev@gmail.com)
