import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# PROFESSOR'S NOTE: Swapped RandomForest for DecisionTree and imported TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DATA_PATH = Path("data/processed/customer_with_clusters.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

MODEL_PATH = MODELS_DIR / "surrogate_rf.joblib" # Kept the same name so app.py doesn't break yet
CM_PATH = REPORTS_DIR / "surrogate_confusion_matrix.csv"
PER_CLASS_PATH = REPORTS_DIR / "surrogate_per_class_accuracy.csv"
CV_PATH = REPORTS_DIR / "surrogate_cv_scores.csv"
SUMMARY_PATH = REPORTS_DIR / "surrogate_eval_summary.txt"
FI_PATH = REPORTS_DIR / "surrogate_feature_importance.csv"

FEATURE_COLS = [
    "Recency",
    "Frequency",
    "Monetary",
    "Weekend_Ratio",
    "Night_Shopper",
    "Basket_Diversity",
    "Return_Rate",
]


def per_class_accuracy(y_true, y_pred):
    """Returns per-class accuracy = correct_in_class / total_in_class."""
    classes = np.unique(y_true)
    rows = []
    for c in classes:
        mask = (y_true == c)
        acc_c = float((y_pred[mask] == y_true[mask]).mean()) if mask.sum() > 0 else float("nan")
        rows.append({"class": int(c), "support": int(mask.sum()), "per_class_accuracy": acc_c})
    return pd.DataFrame(rows).sort_values("class")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv_folds", type=int, default=5)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # PROFESSOR'S NOTE: Chronological sorting applied before the split
    if "LastInvoiceDate" in df.columns:
        df["LastInvoiceDate"] = pd.to_datetime(df["LastInvoiceDate"])
        df = df.sort_values("LastInvoiceDate").dropna(subset=["LastInvoiceDate"])
    else:
        print("WARNING: 'LastInvoiceDate' not found. Splitting will be sequential but not guaranteed chronological.")

    X = df[FEATURE_COLS].copy()
    y = df["ClusterID"].astype(int).copy()

    # PROFESSOR'S NOTE: 1) Proper OUT-OF-TIME Split (Sequential instead of Random Shuffle)
    split_idx = int(len(df) * (1 - args.test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # PROFESSOR'S NOTE: 2) Train an interpretable surrogate (Decision Tree) on TRAIN only
    clf = DecisionTreeClassifier(
        max_depth=5, 
        random_state=args.seed,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    # 3) Evaluate on TEST (Out-Of-Time Holdout)
    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in sorted(y.unique())],
        columns=[f"pred_{c}" for c in sorted(y.unique())],
    )
    cm_df.to_csv(CM_PATH, index=True)

    per_cls_df = per_class_accuracy(y_test.to_numpy(), y_pred)
    per_cls_df.to_csv(PER_CLASS_PATH, index=False)

    report = classification_report(y_test, y_pred, digits=4)

    # PROFESSOR'S NOTE: 4) Time Series Cross-Validation to prevent temporal leakage across folds
    cv = TimeSeriesSplit(n_splits=args.cv_folds)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    cv_df = pd.DataFrame({"fold": np.arange(1, len(cv_scores) + 1), "accuracy": cv_scores})
    cv_df.to_csv(CV_PATH, index=False)

    # 5) Save feature importances (DecisionTree also supports feature_importances_)
    fi = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False)
    fi.to_csv(FI_PATH, index=False)

    # 6) Save model
    joblib.dump(clf, MODEL_PATH)

    # 7) Write summary text
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("Surrogate evaluation (Chronological Out-Of-Time)\n")
        f.write(f"Model: DecisionTreeClassifier (max_depth=5)\n")
        f.write(f"Test size: {args.test_size}\n")
        f.write(f"Random seed: {args.seed}\n\n")
        f.write(f"Test accuracy (OOT): {test_acc:.4f}\n")
        f.write(f"Time Series CV (mean ± std) accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
        f.write("Classification report (test):\n")
        f.write(report)
        f.write("\n")

    # Console output
    print(f"Test accuracy (Out-Of-Time fidelity proxy): {test_acc:.4f}")
    print(f"Time Series CV accuracy mean ± std: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("Saved model:", MODEL_PATH)
    print("Saved confusion matrix:", CM_PATH)
    print("Saved per-class accuracy:", PER_CLASS_PATH)
    print("Saved CV scores:", CV_PATH)
    print("Saved summary:", SUMMARY_PATH)
    print("Saved feature importance:", FI_PATH)
    print("\nClassification report (test):")
    print(report)


if __name__ == "__main__":
    main()