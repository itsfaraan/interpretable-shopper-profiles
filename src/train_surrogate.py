from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA = Path("data/processed/customer_with_clusters.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA)
    feature_cols = [c for c in df.columns if c not in ["CustomerID", "ClusterID"]]

    X = df[feature_cols]
    y = df["ClusterID"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Surrogate accuracy (fidelity proxy):", round(acc, 4))
    print(classification_report(y_test, preds))

    # ذخیره مدل
    joblib.dump(clf, MODELS_DIR / "surrogate_rf.joblib")
    print("Saved:", MODELS_DIR / "surrogate_rf.joblib")

    # ذخیره اهمیت ویژگی‌ها برای گزارش
    imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    imp.to_csv(REPORTS_DIR / "surrogate_feature_importance.csv")
    print("Saved:", REPORTS_DIR / "surrogate_feature_importance.csv")

if __name__ == "__main__":
    main()
