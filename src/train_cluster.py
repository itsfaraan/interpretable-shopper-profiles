import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)

# Input
DATA_PATH = Path("data/processed/customer_features.csv")

# Folders
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

# Feature sets for ablation
FEATURE_SETS = {
    "rfm": ["Recency", "Frequency", "Monetary"],
    "full": [
        "Recency",
        "Frequency",
        "Monetary",
        "Weekend_Ratio",
        "Night_Shopper",
        "Basket_Diversity",
        "Return_Rate",
    ],
}


def compute_k_metrics(X_scaled, k_min=2, k_max=10, seed=42):
    """
    For each k:
      - Silhouette (higher better)
      - Davies–Bouldin (lower better)
      - Calinski–Harabasz (higher better)
    """
    results = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X_scaled)

        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)

        results.append((k, float(sil), float(db), float(ch)))
    return results


def cluster_stability_ari(X_scaled, k, runs, bootstrap=False, bootstrap_frac=1.0):
    """
    Run KMeans multiple times and compare clusterings via pairwise ARI.

    If bootstrap=True:
      - fit on bootstrap sample
      - predict labels on FULL dataset
      so label arrays are comparable (same length).
    """
    rng = np.random.default_rng(0)
    n = X_scaled.shape[0]
    labels_list = []

    for s in range(1, runs + 1):
        km = KMeans(n_clusters=k, random_state=int(s), n_init=10)

        if bootstrap:
            m = int(max(2, min(n, round(n * bootstrap_frac))))
            idx = rng.choice(n, size=m, replace=True)
            km.fit(X_scaled[idx])
            labels_full = km.predict(X_scaled)
        else:
            labels_full = km.fit_predict(X_scaled)

        labels_list.append(labels_full)

    pairs = []
    aris = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            ari = adjusted_rand_score(labels_list[i], labels_list[j])
            pairs.append((i, j, float(ari)))
            aris.append(float(ari))

    mean_ari = float(np.mean(aris)) if aris else float("nan")
    std_ari = float(np.std(aris)) if aris else float("nan")
    return pairs, mean_ari, std_ari


def align_labels_bruteforce(y_ref, y_new, k):
    """
    Align cluster IDs in y_new to best match y_ref (handles label permutation).
    For k<=8 brute force is feasible; k=6 is fast (720 permutations).
    """
    y_ref = np.asarray(y_ref)
    y_new = np.asarray(y_new)

    if k > 8:
        match_rate = float((y_ref == y_new).mean())
        return y_new, match_rate

    labels = list(range(k))
    best_match = -1
    best_map = None

    for perm in itertools.permutations(labels):
        mapping = {old: perm[old] for old in labels}
        mapped = np.vectorize(mapping.get)(y_new)
        match = int((mapped == y_ref).sum())
        if match > best_match:
            best_match = match
            best_map = mapping

    aligned = np.vectorize(best_map.get)(y_new)
    return aligned, float(best_match / len(y_ref))


def noise_robustness(X_scaled, baseline_labels, k, eps, runs, seed):
    """
    Add Gaussian noise to SCALED features and re-run clustering.
    Report % customers whose cluster changes (after aligning labels).
    """
    rng = np.random.default_rng(seed)
    changed_pcts = []
    rows = []

    for r in range(1, runs + 1):
        X_noisy = X_scaled + rng.normal(loc=0.0, scale=eps, size=X_scaled.shape)

        km_noisy = KMeans(n_clusters=k, random_state=seed + r, n_init=10)
        noisy_labels = km_noisy.fit_predict(X_noisy)

        aligned_labels, _ = align_labels_bruteforce(baseline_labels, noisy_labels, k=k)
        changed_pct = float((aligned_labels != baseline_labels).mean() * 100.0)

        changed_pcts.append(changed_pct)
        rows.append({"run": r, "k": k, "noise_eps": float(eps), "changed_pct": changed_pct})

    mean_changed = float(np.mean(changed_pcts)) if changed_pcts else float("nan")
    std_changed = float(np.std(changed_pcts)) if changed_pcts else float("nan")
    return pd.DataFrame(rows), mean_changed, std_changed


def main():
    parser = argparse.ArgumentParser()

    # Ablation: choose feature set
    parser.add_argument("--feature_set", choices=["rfm", "full"], default="full",
                        help="rfm = [Recency, Frequency, Monetary], full = RFM + behavioral features")

    # k scan metrics
    parser.add_argument("--kmin", type=int, default=2)
    parser.add_argument("--kmax", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42, help="Seed for metrics scan + final model")

    # final chosen k
    parser.add_argument("--k", type=int, default=6, help="Chosen number of clusters")

    # stability (ARI)
    parser.add_argument("--stability_runs", type=int, default=20)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--bootstrap_frac", type=float, default=0.8)

    # robustness to noise
    parser.add_argument("--noise_eps", type=float, default=0.05, help="Gaussian noise std on SCALED features")
    parser.add_argument("--noise_runs", type=int, default=20, help="How many noisy reruns")
    parser.add_argument("--noise_seed", type=int, default=123, help="Seed for noise generation")

    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Output paths (avoid overwriting by feature_set)
    OUT_WITH_CLUSTERS = Path(f"data/processed/customer_with_clusters_{args.feature_set}.csv")
    K_METRICS_REPORT = REPORTS_DIR / f"k_metrics_{args.feature_set}.csv"
    STABILITY_REPORT = REPORTS_DIR / f"cluster_stability_ari_{args.feature_set}.csv"
    NOISE_REPORT = REPORTS_DIR / f"cluster_noise_robustness_{args.feature_set}.csv"

    KMEANS_MODEL_PATH = MODELS_DIR / f"kmeans_{args.feature_set}.joblib"
    SCALER_PATH = MODELS_DIR / f"scaler_{args.feature_set}.joblib"

    # Load data
    df = pd.read_csv(DATA_PATH)

    feature_cols = FEATURE_SETS[args.feature_set]
    X = df[feature_cols].copy()

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n=== Feature set: {args.feature_set} ===")
    print("Using features:", feature_cols)

    # 1) Multi-metric validation across k
    k_results = compute_k_metrics(X_scaled, k_min=args.kmin, k_max=args.kmax, seed=args.seed)
    k_df = pd.DataFrame(k_results, columns=["k", "silhouette", "davies_bouldin", "calinski_harabasz"])
    k_df.to_csv(K_METRICS_REPORT, index=False)

    print("\nk metrics (silhouette ↑, davies_bouldin ↓, calinski_harabasz ↑):")
    for k, sil, db, ch in k_results:
        print(f"  k={k:2d} | sil={sil:.4f} | db={db:.4f} | ch={ch:.2f}")
    print(f"Saved: {K_METRICS_REPORT}")

    # 2) Fit final model on full data
    kmeans = KMeans(n_clusters=args.k, random_state=args.seed, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    out_df = df.copy()
    out_df["ClusterID"] = labels
    out_df.to_csv(OUT_WITH_CLUSTERS, index=False)

    joblib.dump(kmeans, KMEANS_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\nChosen k: {args.k}")
    print(f"Saved: {OUT_WITH_CLUSTERS}")
    print(f"Saved models: {KMEANS_MODEL_PATH}, {SCALER_PATH}")

    # 3) Stability (ARI)
    pairs, mean_ari, std_ari = cluster_stability_ari(
        X_scaled,
        k=args.k,
        runs=args.stability_runs,
        bootstrap=args.bootstrap,
        bootstrap_frac=args.bootstrap_frac,
    )
    stab_df = pd.DataFrame(pairs, columns=["run_i", "run_j", "ARI"])
    stab_df["k"] = args.k
    stab_df["feature_set"] = args.feature_set
    stab_df["bootstrap"] = bool(args.bootstrap)
    stab_df["bootstrap_frac"] = float(args.bootstrap_frac)
    stab_df.to_csv(STABILITY_REPORT, index=False)

    print("\n=== Cluster Stability (ARI) ===")
    print(f"Runs: {args.stability_runs} | bootstrap={args.bootstrap} | frac={args.bootstrap_frac}")
    print(f"Pairwise ARI mean ± std: {mean_ari:.4f} ± {std_ari:.4f}")
    print(f"Saved: {STABILITY_REPORT}")

    # 4) Robustness to noise (% label changes)
    noise_df, mean_changed, std_changed = noise_robustness(
        X_scaled=X_scaled,
        baseline_labels=labels,
        k=args.k,
        eps=args.noise_eps,
        runs=args.noise_runs,
        seed=args.noise_seed,
    )
    noise_df["feature_set"] = args.feature_set
    noise_df.to_csv(NOISE_REPORT, index=False)

    print("\n=== Robustness to Noise (Label Changes) ===")
    print(f"Noise eps (on scaled features): {args.noise_eps}")
    print(f"Runs: {args.noise_runs}")
    print(f"% customers changing cluster (mean ± std): {mean_changed:.2f}% ± {std_changed:.2f}%")
    print(f"Saved: {NOISE_REPORT}")


if __name__ == "__main__":
    main()
