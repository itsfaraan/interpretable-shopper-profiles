import os
import pandas as pd

def test_files_exist():
    assert os.path.exists("data/processed/customer_features.csv")
    assert os.path.exists("data/processed/customer_with_clusters.csv")
    assert os.path.exists("models/kmeans.joblib")
    assert os.path.exists("models/scaler.joblib")
    assert os.path.exists("models/surrogate_rf.joblib")

def test_return_rate_range():
    df = pd.read_csv("data/processed/customer_features.csv")
    rr = df["Return_Rate"]
    assert ((rr >= 0) & (rr <= 1)).all()

def test_clusters_present():
    df = pd.read_csv("data/processed/customer_with_clusters.csv")
    assert "ClusterID" in df.columns
    assert df["ClusterID"].nunique() >= 2
