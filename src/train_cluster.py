from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

DATA = Path("data/processed/customer_features.csv")
OUT_WITH_LABELS = Path("data/processed/customer_with_clusters.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

# ✅ اگر می‌خوای k را دستی ثابت کنی (پیشنهاد: 6 یا 8)
FORCE_K = 6   # اگر نمی‌خوای دستی باشه، بذار None

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA)
    X = df.drop(columns=["CustomerID"]).values

    # استانداردسازی
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # محاسبه silhouette برای kهای مختلف (برای گزارش)
    results = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(Xs)
        score = silhouette_score(Xs, labels)
        results.append((k, score))

    rep = pd.DataFrame(results, columns=["k", "silhouette"])
    rep.to_csv(REPORTS_DIR / "k_silhouette.csv", index=False)

    # انتخاب k
    if FORCE_K is None:
        best_k = int(rep.sort_values("silhouette", ascending=False).iloc[0]["k"])
    else:
        best_k = int(FORCE_K)

    # مدل نهایی با k انتخابی
    best_model = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(Xs)
    best_score = silhouette_score(Xs, best_model.labels_)

    print("Chosen k:", best_k, "silhouette:", round(best_score, 4))
    print("All results:", [(k, round(s, 4)) for k, s in results])

    # برچسب‌گذاری و ذخیره دیتای خروجی
    df_out = df.copy()
    df_out["ClusterID"] = best_model.labels_
    df_out.to_csv(OUT_WITH_LABELS, index=False)
    print("Saved:", OUT_WITH_LABELS)

    # ذخیره مدل‌ها
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(best_model, MODELS_DIR / "kmeans.joblib")
    print("Saved models to:", MODELS_DIR)

    print("Saved:", REPORTS_DIR / "k_silhouette.csv")

if __name__ == "__main__":
    main()
