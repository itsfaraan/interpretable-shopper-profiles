import numpy as np
import pandas as pd
import joblib
import streamlit as st
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Shopper Profiles (XAI)", layout="wide")

DATA_PATH = "data/processed/customer_with_clusters.csv"
SURROGATE_PATH = "models/surrogate_rf.joblib"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(SURROGATE_PATH)

df = load_data()
clf = load_model()

feature_cols = [c for c in df.columns if c not in ["CustomerID", "ClusterID"]]

st.title("Interpretable Shopper Profiles")
st.caption("KMeans clustering + Surrogate RandomForest + SHAP explanations")

with st.sidebar:
    st.header("Customer selection")
    cid = st.selectbox("CustomerID", df["CustomerID"].tolist())

row = df[df["CustomerID"] == cid].iloc[0]
cluster_id = int(row["ClusterID"])
st.subheader(f"Customer {cid}  |  Segment (ClusterID): {cluster_id}")

c1, c2 = st.columns(2)

with c1:
    st.markdown("### Customer features")
    st.dataframe(row[feature_cols].to_frame("value"))

with c2:
    st.markdown("### Segment mean")
    seg_mean = df[df["ClusterID"] == cluster_id][feature_cols].mean()
    st.dataframe(seg_mean.to_frame("mean"))

st.markdown("## Why this segment? (SHAP on surrogate model)")

X_one = pd.DataFrame([row[feature_cols].values], columns=feature_cols)
pred_class = int(clf.predict(X_one)[0])
st.write(f"Surrogate predicted class: {pred_class}")

# --- SHAP (robust across versions / multiclass) ---
explainer = shap.TreeExplainer(clf)

# newer SHAP returns an Explanation
try:
    exp = explainer(X_one)
    values = np.array(exp.values)
    if values.ndim == 3:
        # (n_samples, n_features, n_classes)
        vals_1d = values[0, :, pred_class].astype(float)
    else:
        # (n_samples, n_features)
        vals_1d = values[0, :].astype(float)
except Exception:
    # fallback for older versions: list of arrays
    shap_values = explainer.shap_values(X_one)
    if isinstance(shap_values, list):
        vals_1d = np.array(shap_values[pred_class][0], dtype=float)
    else:
        vals_1d = np.array(shap_values[0], dtype=float)

# --- Plot (bar) ---
fig = plt.figure()
try:
    # simple bar plot using matplotlib (always works)
    order = np.argsort(np.abs(vals_1d))[::-1][:8]
    plt.barh([feature_cols[i] for i in order][::-1], vals_1d[order][::-1])
    plt.title("Top SHAP contributions (local)")
    plt.tight_layout()
except Exception as e:
    plt.text(0.1, 0.5, f"Plot failed: {e}")
st.pyplot(fig, clear_figure=True)

# --- Text explanation ---
st.markdown("### Simple explanation (Top drivers)")
top_idx = np.argsort(np.abs(vals_1d))[::-1][:3]
for i in top_idx:
    st.write(f"- **{feature_cols[i]}** | SHAP = {vals_1d[i]:.4f} | value = {float(X_one.iloc[0, i]):.4f}")

