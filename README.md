# Interpretable Shopper Profiles (Online Retail II)

## Goal
Build *interpretable* shopper segments from transaction logs and explain **why** a customer belongs to a segment.
This project aligns with both **research-driven** (feature design + evaluation) and **innovation-driven** (interactive web demo) tracks.

## Dataset
Online Retail II (UCI) – real transaction logs at invoice-line level.

## Method Overview
### 1) Data engineering & feature extraction
From raw transactions → one row per customer with:
- **RFM**: Recency, Frequency, Monetary
- **Behavioral**: Weekend_Ratio, Night_Shopper, Basket_Diversity, Return_Rate

Output:
- `data/processed/customer_features.csv` (5942 customers, 8 columns)

### 2) Unsupervised learning (pattern discovery)
- Standardize features (StandardScaler)
- **KMeans** clustering
- k scanned in [2..10] using **Silhouette score**
- Chosen k = **6** for more meaningful, presentable segments  
  (metrics saved in `reports/k_silhouette.csv`)

Outputs:
- `data/processed/customer_with_clusters.csv`
- `models/scaler.joblib`, `models/kmeans.joblib`

### 3) Interpretable ML (thesis alignment)
Train a **surrogate classifier (RandomForest)** to predict ClusterID from features to enable explanations.
- Metric: **Surrogate accuracy (fidelity proxy)** = **0.9807**
- Global interpretability: `reports/surrogate_feature_importance.csv`
- Local interpretability: **SHAP** (shown in the app)

Output:
- `models/surrogate_rf.joblib`

### 4) Web demo (Streamlit)
Select a customer → view:
- customer feature values
- segment mean
- local SHAP contributions + top drivers (text)

## How to run (local)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/make_features.py
python src/train_cluster.py
python src/train_surrogate.py

streamlit run app/app.py
## Run with Docker
```bash
docker build -t shopper-xai .
docker run --rm -p 9000:8501 shopper-xai
