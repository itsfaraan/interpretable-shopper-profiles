import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


DATA_PATH = Path("data/processed/customer_with_clusters.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

MODEL_PATH = MODELS_DIR / "surrogate_rf.joblib"
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

    X = df[FEATURE_COLS].copy()
    y = df["ClusterID"].astype(int).copy()

    # 1) Proper train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y
    )

    # 2) Train surrogate on TRAIN only
    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    clf.fit(X_train, y_train)

    # 3) Evaluate on TEST
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

    # 4) Cross-validation score (Stratified CV)
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    cv_df = pd.DataFrame({"fold": np.arange(1, len(cv_scores) + 1), "accuracy": cv_scores})
    cv_df.to_csv(CV_PATH, index=False)

    # 5) Save feature importances
    fi = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False)
    fi.to_csv(FI_PATH, index=False)

    # 6) Save model
    joblib.dump(clf, MODEL_PATH)

    # 7) Write summary text
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("Surrogate evaluation (proper)\n")
        f.write(f"Test size: {args.test_size}\n")
        f.write(f"Random seed: {args.seed}\n\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write(f"CV (mean ± std) accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
        f.write("Classification report (test):\n")
        f.write(report)
        f.write("\n")

    # Console output (useful for you)
    print(f"Test accuracy (fidelity proxy): {test_acc:.4f}")
    print(f"CV accuracy mean ± std: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
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
