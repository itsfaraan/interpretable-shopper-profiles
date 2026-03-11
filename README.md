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
- **KMeans** clustering (k=6) validated via Silhouette score, Davies-Bouldin, and bootstrap stability (ARI)
- Chosen k = **6** for more meaningful, presentable segments  
  (metrics saved in `reports/k_silhouette.csv`)

Outputs:
- `data/processed/customer_with_clusters.csv`
- `models/scaler.joblib`, `models/kmeans.joblib`

### 3) Interpretable ML (Surrogate Pipeline)
- Transparent Surrogate: Train a depth-constrained **Decision Tree Classifier (max_depth=5)** on the unscaled features to predict the unsupervised cluster labels.
- Chronological Validation: To prevent temporal data leakage, the dataset is sorted by LastInvoiceDate using an **Out-Of-Time (OOT) Split** and Time Series Cross-Validation (training on past behavior to predict the future)
- Metric (Fidelity Proxies): Out-Of-Time Test Accuracy: **0.9445**. Time Series CV Accuracy: **0.9111 ± 0.0295** 
- Global interpretability: `reports/surrogate_feature_importance.csv`
- Local interpretability: **SHAP** (shown in the app)

Output:
- `models/surrogate_rf.joblib`

### 4) Web demo (Streamlit CRM Interface)

**Tab 1 (Micro View):** Select a customer → view feature values vs. segment means, local SHAP impact charts, and run a counterfactual "What-If" slider simulator.

**Tab 2 (Macro View):** View global segment distribution and average spend ($K) per persona to drive marketing budget allocation.

**Tab 3 (QA View):** Raw data transparency explorer to audit inputs and identify extreme-value outliers.

## How to run (local)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Run the data and ML pipeline sequentially
python src/make_features.py
python src/train_cluster.py --feature_set full
python src/cluster_naming.py --feature_set full
python src/train_surrogate.py

# 2. Launch the CRM web interface
streamlit run src/app.py
## Run with Docker
```bash
docker build -t shopper-xai .
docker run --rm -p 9000:8501 shopper-xai

# Open your browser and navigate to http://localhost:9000
