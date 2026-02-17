import argparse
from pathlib import Path
import numpy as np
import pandas as pd


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


def safe_zscore(x: pd.Series, mean: float, std: float) -> pd.Series:
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mean) / std


def descriptor_rules(feature: str, z: float) -> str | None:
    """
    Maps feature + direction to a human-friendly descriptor.
    """
    hi = z >= 0.75
    lo = z <= -0.75
    if not (hi or lo):
        return None

    if feature == "Recency":
        return "Churn Risk" if hi else "Recent Buyers"
    if feature == "Frequency":
        return "Frequent" if hi else "Occasional"
    if feature == "Monetary":
        return "High Spend" if hi else "Low Spend"
    if feature == "Weekend_Ratio":
        return "Weekend Shoppers" if hi else "Weekday Shoppers"
    if feature == "Night_Shopper":
        return "Night Shoppers" if hi else "Day Shoppers"
    if feature == "Basket_Diversity":
        return "Diverse Basket" if hi else "Narrow Basket"
    if feature == "Return_Rate":
        return "High Returns" if hi else "Low Returns"

    return None


def build_cluster_name(z_row: pd.Series, features: list[str], max_tags: int = 2) -> str:
    """
    Builds a short name from strongest rule-matched descriptors using cluster mean z-scores.
    Picks up to max_tags descriptors.
    """
    # Order features by absolute z-score (strongest first)
    ordered = sorted(features, key=lambda f: abs(float(z_row.get(f, 0.0))), reverse=True)

    tags = []
    for f in ordered:
        z = float(z_row.get(f, 0.0))
        tag = descriptor_rules(f, z)
        if tag and tag not in tags:
            tags.append(tag)
        if len(tags) >= max_tags:
            break

    # Fallback: if no tags passed threshold
    if not tags:
        # pick top 1-2 strongest features even if below threshold
        top = ordered[:max_tags]
        fallback = []
        for f in top:
            z = float(z_row.get(f, 0.0))
            direction = "High" if z >= 0 else "Low"
            fallback.append(f"{direction} {f}")
        tags = fallback

    return " • ".join(tags)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_set", choices=["rfm", "full"], default="full")
    parser.add_argument("--input", type=str, default=None,
                        help="Input clustered CSV. Default: data/processed/customer_with_clusters_<feature_set>.csv")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV with ClusterName. Default: data/processed/customer_with_cluster_names_<feature_set>.csv")
    parser.add_argument("--z_threshold", type=float, default=0.75, help="Threshold for tag rules (default 0.75)")
    parser.add_argument("--max_tags", type=int, default=2, help="Max descriptors per cluster name")
    args = parser.parse_args()

    features = FEATURE_SETS[args.feature_set]

    inp = Path(args.input) if args.input else Path(f"data/processed/customer_with_clusters_{args.feature_set}.csv")
    out = Path(args.output) if args.output else Path(f"data/processed/customer_with_cluster_names_{args.feature_set}.csv")

    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)

    names_path = reports / f"cluster_names_{args.feature_set}.csv"
    z_path = reports / f"cluster_zscores_{args.feature_set}.csv"

    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}. Run train_cluster.py first.")

    df = pd.read_csv(inp)

    if "ClusterID" not in df.columns:
        raise ValueError("Input file must contain ClusterID column.")

    # Global mean/std for z-scores (based on all customers)
    global_mean = df[features].mean(numeric_only=True)
    global_std = df[features].std(numeric_only=True).replace(0, np.nan)

    # Cluster means
    cluster_means = df.groupby("ClusterID")[features].mean(numeric_only=True)

    # Cluster z-scores of means
    cluster_z = cluster_means.copy()
    for f in features:
        cluster_z[f] = safe_zscore(cluster_means[f], float(global_mean[f]), float(global_std[f]))

    # Build names (use threshold in rules via global variable trick)
    # We'll temporarily override rule threshold by checking in descriptor_rules:
    # easiest is to adjust descriptor_rules logic, but we’ll apply threshold here by zeroing small z.
    # So rules only see big signals.
    z_for_rules = cluster_z.copy()
    for f in features:
        z_for_rules[f] = z_for_rules[f].where(z_for_rules[f].abs() >= args.z_threshold, 0.0)

    mapping_rows = []
    for cid in cluster_z.index:
        name = build_cluster_name(z_for_rules.loc[cid], features, max_tags=args.max_tags)
        mapping_rows.append({"ClusterID": int(cid), "ClusterName": name})

    mapping = pd.DataFrame(mapping_rows).sort_values("ClusterID")
    mapping.to_csv(names_path, index=False)
    cluster_z.reset_index().to_csv(z_path, index=False)

    # Attach names to dataset
    df = df.merge(mapping, on="ClusterID", how="left")
    df.to_csv(out, index=False)

    print("=== Automatic Cluster Naming ===")
    print(f"Feature set: {args.feature_set}")
    print(f"Input:  {inp}")
    print(f"Output: {out}")
    print(f"Saved names:   {names_path}")
    print(f"Saved zscores: {z_path}")
    print("\nCluster labels:")
    for row in mapping_rows:
        print(f"  Cluster {row['ClusterID']}: {row['ClusterName']}")


if __name__ == "__main__":
    main()
