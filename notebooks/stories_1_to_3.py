"""Feature Stories 1 to 3: EDA, preparation, K-means, and DBSCAN.

Run from the project root with: uv run python notebooks/stories_1_to_3.py
The comments marked ADDED identify the new work for these three stories.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # ADDED: save figures without requiring a notebook window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


# ADDED: use paths relative to this file so the script works outside Colab.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "repport" / "stories_1_to_3"
RANDOM_STATE = 42
NUMERIC_COLUMNS = [
    "BALANCE",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "CREDIT_LIMIT",
    "PAYMENTS",
]


def save_figure(name):
    """ADDED: apply the same layout and save each required plot."""
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / name, dpi=150)
    plt.close()


def clustering_scores(x, labels, ignore_noise=False):
    """ADDED: score a clustering only when it has at least two real groups."""
    labels = np.asarray(labels)
    noise_share = float(np.mean(labels == -1)) if ignore_noise else 0.0
    if ignore_noise:
        keep = labels != -1
        x = x[keep]
        labels = labels[keep]

    groups, counts = np.unique(labels, return_counts=True)
    if len(groups) < 2 or len(groups) >= len(labels):
        return np.nan, np.nan, np.nan, noise_share, np.nan, np.nan

    # ADDED: silhouette uses a fixed random sample to keep the grid search quick.
    silhouette = silhouette_score(
        x, labels, sample_size=min(1000, len(labels)), random_state=RANDOM_STATE
    )
    davies = davies_bouldin_score(x, labels)
    calinski = calinski_harabasz_score(x, labels)
    shares = counts / len(labels)
    return silhouette, davies, calinski, noise_share, shares.min(), shares.max()


def main():
    # ADDED: create the output folders before any file is written.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ADDED (Story 1): load and check the eight expected source columns.
    df = pd.read_csv(RAW_FILE)
    required = ["CUST_ID"] + NUMERIC_COLUMNS
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing dataset columns: {missing_columns}")
    if not all(pd.api.types.is_numeric_dtype(df[column]) for column in NUMERIC_COLUMNS):
        raise ValueError("The seven behavioral columns must contain numbers.")

    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = int(df["CUST_ID"].duplicated().sum())
    print(f"Rows: {len(df)}, columns: {len(df.columns)}")
    print(f"Duplicate rows: {duplicate_rows}, duplicate CUST_ID values: {duplicate_ids}")
    print(df.head())
    print(df.dtypes)
    df.describe(include="all").to_csv(PROCESSED_DIR / "eda_describe.csv")

    # ADDED (Story 1): include every column in the sorted missing-value table.
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": df.isna().mean() * 100,
    }).sort_values("missing_count", ascending=False)
    missing.to_csv(PROCESSED_DIR / "eda_missing_values.csv")

    # ADDED (Story 1): histogram, KDE, and skewness for all seven features.
    fig, axes = plt.subplots(4, 2, figsize=(13, 14))
    for axis, column in zip(axes.flat, NUMERIC_COLUMNS):
        sns.histplot(df[column].dropna(), bins=30, kde=True, ax=axis)
        axis.set_title(f"{column} | skewness = {df[column].skew():.2f}")
    axes.flat[-1].axis("off")
    save_figure("01_histograms_kde.png")

    # ADDED (Story 1): one boxplot per feature for visual outlier checks.
    fig, axes = plt.subplots(4, 2, figsize=(13, 14))
    for axis, column in zip(axes.flat, NUMERIC_COLUMNS):
        sns.boxplot(x=df[column], ax=axis)
        axis.set_title(column)
    axes.flat[-1].axis("off")
    save_figure("02_boxplots.png")

    # ADDED (Story 1): count observations outside the 1.5 x IQR bounds.
    outlier_rows = []
    for column in NUMERIC_COLUMNS:
        q1, q3 = df[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        outside = (df[column] < q1 - 1.5 * iqr) | (df[column] > q3 + 1.5 * iqr)
        outlier_rows.append({
            "feature": column,
            "outlier_count": int(outside.sum()),
            "outlier_pct": float(outside.mean() * 100),
        })
    pd.DataFrame(outlier_rows).to_csv(PROCESSED_DIR / "eda_outliers.csv", index=False)

    # ADDED (Story 1): correlation heatmap for the seven numeric features.
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[NUMERIC_COLUMNS].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    save_figure("03_correlations.png")

    # ADDED (Story 2): make a clean reference copy in original money units.
    # Only CREDIT_LIMIT is missing in the supplied dataset. Its median is a
    # simple choice that does not let one large credit limit dominate the fill.
    df_clean = df.drop_duplicates().copy()
    if df_clean["CUST_ID"].duplicated().any():
        raise ValueError("CUST_ID contains duplicates; inspect them before clustering.")
    other_missing = df_clean[NUMERIC_COLUMNS].drop(columns="CREDIT_LIMIT").isna().sum()
    if other_missing.any():
        raise ValueError(f"Unexpected missing values: {other_missing[other_missing > 0]}")
    df_clean["CREDIT_LIMIT"] = df_clean["CREDIT_LIMIT"].fillna(
        df_clean["CREDIT_LIMIT"].median()
    )
    df_clean.to_csv(PROCESSED_DIR / "df_clean.csv", index=False)

    # ADDED (Story 2): keep clustering transformations on a separate copy.
    # CUST_ID stays in df_clean for matching rows, but never enters a model.
    clustering_raw = df_clean[NUMERIC_COLUMNS].copy()
    if (clustering_raw < 0).any().any():
        raise ValueError("log1p needs nonnegative monetary values.")
    clustering_log = np.log1p(clustering_raw)

    # ADDED (Story 2): compare skewness before and after the actual log step.
    skew_table = pd.DataFrame({
        "feature": NUMERIC_COLUMNS,
        "skew_before": clustering_raw.skew().values,
        "skew_after_log": clustering_log.skew().values,
    })
    skew_table.to_csv(PROCESSED_DIR / "skew_before_after.csv", index=False)
    fig, axes = plt.subplots(4, 2, figsize=(13, 14))
    for axis, column in zip(axes.flat, NUMERIC_COLUMNS):
        sns.histplot(clustering_log[column], bins=30, kde=True, ax=axis)
        axis.set_title(f"{column} after log | skewness = {clustering_log[column].skew():.2f}")
    axes.flat[-1].axis("off")
    save_figure("04_after_log.png")

    # ADDED (Story 2): the order is log, StandardScaler, and then PCA.
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(clustering_log)
    df_prepare_clustering = pd.DataFrame(scaled_values, columns=NUMERIC_COLUMNS)
    df_prepare_clustering.to_csv(PROCESSED_DIR / "df_prepare_clustering.csv", index=False)
    if not np.isfinite(scaled_values).all():
        raise ValueError("Clustering data contains NaN or infinite values.")

    # ADDED (Story 2): full PCA shows the first count reaching 80% variance.
    full_pca = PCA().fit(scaled_values)
    cumulative = np.cumsum(full_pca.explained_variance_ratio_)
    first_80 = int(np.argmax(cumulative >= 0.80) + 1)
    variance_table = pd.DataFrame({
        "components": np.arange(1, len(cumulative) + 1),
        "explained_variance": full_pca.explained_variance_ratio_,
        "cumulative_variance": cumulative,
    })
    variance_table.to_csv(PROCESSED_DIR / "pca_variance.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.plot(variance_table["components"], cumulative, marker="o")
    plt.axhline(0.80, color="red", linestyle="--", label="80% threshold")
    plt.axvline(first_80, color="green", linestyle="--", label=f"First: {first_80}")
    plt.xlabel("PCA components")
    plt.ylabel("Cumulative explained variance")
    plt.legend()
    save_figure("05_pca_variance.png")
    print(f"First PCA count reaching 80% variance: {first_80}")

    # ADDED (Story 3): search PCA counts and K-means k together.
    # Save the model and PCA data for each pair so the winner is reproducible.
    kmeans_results = []
    kmeans_models = {}
    pca_data = {}
    for components in range(2, 6):
        pca_data[components] = PCA(n_components=components).fit_transform(scaled_values)
        for k in range(2, 8):
            model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
            labels = model.fit_predict(pca_data[components])
            scores = clustering_scores(pca_data[components], labels)
            kmeans_results.append({
                "components": components,
                "k": k,
                "inertia": model.inertia_,
                "silhouette": scores[0],
                "davies_bouldin": scores[1],
                "calinski_harabasz": scores[2],
                "smallest_cluster_pct": scores[4] * 100,
                "largest_cluster_pct": scores[5] * 100,
            })
            kmeans_models[(components, k)] = model
    kmeans_table = pd.DataFrame(kmeans_results)
    kmeans_table.to_csv(PROCESSED_DIR / "kmeans_grid.csv", index=False)

    # ADDED (Story 3): heatmap answers which PCA count and k work together.
    heatmap_values = kmeans_table.pivot(index="components", columns="k", values="silhouette")
    plt.figure(figsize=(9, 5))
    sns.heatmap(heatmap_values, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("K-means silhouette: PCA components x k")
    save_figure("06_kmeans_grid_heatmap.png")

    # ADDED (Story 3): reject candidates with a cluster below 10% or above 90%.
    balanced_kmeans = kmeans_table[
        (kmeans_table["smallest_cluster_pct"] >= 10)
        & (kmeans_table["largest_cluster_pct"] <= 90)
    ]
    if balanced_kmeans.empty:
        raise ValueError("No balanced K-means candidate. Review the grid and data.")
    best_kmeans = balanced_kmeans.sort_values("silhouette", ascending=False).iloc[0]
    best_components = int(best_kmeans["components"])
    best_k = int(best_kmeans["k"])
    best_kmeans_model = kmeans_models[(best_components, best_k)]
    df_clean["cluster_kmeans"] = best_kmeans_model.labels_

    # ADDED (Story 3): elbow and silhouette curves use the selected PCA count.
    curve = kmeans_table[kmeans_table["components"] == best_components].sort_values("k")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(curve["k"], curve["inertia"], marker="o")
    axes[0].set(title="Elbow curve", xlabel="k", ylabel="Inertia")
    axes[1].plot(curve["k"], curve["silhouette"], marker="o")
    axes[1].set(title="Silhouette curve", xlabel="k", ylabel="Silhouette")
    for axis in axes:
        axis.axvline(best_k, color="red", linestyle="--")
    save_figure("07_kmeans_curves.png")

    # ADDED (Story 3): show counts and percentages for the chosen K-means model.
    kmeans_sizes = df_clean["cluster_kmeans"].value_counts().sort_index()
    kmeans_size_table = pd.DataFrame({
        "cluster": kmeans_sizes.index,
        "customers": kmeans_sizes.values,
        "percent": kmeans_sizes.values / len(df_clean) * 100,
    })
    kmeans_size_table.to_csv(PROCESSED_DIR / "kmeans_cluster_sizes.csv", index=False)
    plt.figure(figsize=(7, 4))
    sns.barplot(data=kmeans_size_table, x="cluster", y="percent", color="steelblue")
    plt.axhline(10, color="red", linestyle="--", label="10% minimum")
    plt.ylabel("Customers (%)")
    plt.legend()
    save_figure("08_kmeans_cluster_sizes.png")

    # ADDED (Story 3): search DBSCAN across PCA count, eps, and min_samples.
    # Invalid results (one real group) stay in the CSV with empty scores.
    dbscan_results = []
    dbscan_labels = {}
    for components in range(2, 6):
        for eps in [0.4, 0.6, 0.8, 1.0, 1.2]:
            for min_samples in [5, 10]:
                labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(
                    pca_data[components]
                )
                scores = clustering_scores(pca_data[components], labels, ignore_noise=True)
                dbscan_results.append({
                    "components": components,
                    "eps": eps,
                    "min_samples": min_samples,
                    "n_clusters": len(set(labels) - {-1}),
                    "silhouette": scores[0],
                    "noise_pct": scores[3] * 100,
                    "davies_bouldin": scores[1],
                    "calinski_harabasz": scores[2],
                    "smallest_cluster_pct": scores[4] * 100,
                    "largest_cluster_pct": scores[5] * 100,
                })
                dbscan_labels[(components, eps, min_samples)] = labels
    dbscan_table = pd.DataFrame(dbscan_results).sort_values(
        "silhouette", ascending=False, na_position="last"
    )
    dbscan_table.to_csv(PROCESSED_DIR / "dbscan_grid_ranked.csv", index=False)

    # ADDED (Story 3): keep a representative valid DBSCAN result for comparison.
    valid_dbscan = dbscan_table.dropna(subset=["silhouette"])
    if valid_dbscan.empty:
        raise ValueError("The DBSCAN grid produced no result with two real clusters.")
    best_dbscan = valid_dbscan.iloc[0]
    db_key = (
        int(best_dbscan["components"]),
        float(best_dbscan["eps"]),
        int(best_dbscan["min_samples"]),
    )
    df_clean["cluster_dbscan"] = dbscan_labels[db_key]
    df_clean["est_atypique_dbscan"] = df_clean["cluster_dbscan"] == -1

    # ADDED (Story 3): compare separation, coverage, and balance side by side.
    comparison = pd.DataFrame([
        {
            "method": "K-means",
            "components": best_components,
            "parameters": f"k={best_k}",
            "silhouette": best_kmeans["silhouette"],
            "davies_bouldin": best_kmeans["davies_bouldin"],
            "calinski_harabasz": best_kmeans["calinski_harabasz"],
            "noise_pct": 0.0,
            "smallest_cluster_pct": best_kmeans["smallest_cluster_pct"],
            "largest_cluster_pct": best_kmeans["largest_cluster_pct"],
        },
        {
            "method": "DBSCAN",
            "components": int(best_dbscan["components"]),
            "parameters": f"eps={best_dbscan['eps']}, min_samples={db_key[2]}",
            "silhouette": best_dbscan["silhouette"],
            "davies_bouldin": best_dbscan["davies_bouldin"],
            "calinski_harabasz": best_dbscan["calinski_harabasz"],
            "noise_pct": best_dbscan["noise_pct"],
            "smallest_cluster_pct": best_dbscan["smallest_cluster_pct"],
            "largest_cluster_pct": best_dbscan["largest_cluster_pct"],
        },
    ])
    comparison.to_csv(PROCESSED_DIR / "clustering_comparison.csv", index=False)

    # ADDED (Story 3): every customer needs a final cluster. Prefer K-means.
    # Choose DBSCAN only if it covers everyone, has balanced groups, and its
    # silhouette is at least 0.05 higher. This makes the decision explicit.
    dbscan_wins = (
        best_dbscan["noise_pct"] == 0
        and best_dbscan["smallest_cluster_pct"] >= 10
        and best_dbscan["largest_cluster_pct"] <= 90
        and best_dbscan["silhouette"] >= best_kmeans["silhouette"] + 0.05
    )
    if dbscan_wins:
        final_method = "DBSCAN"
        final_components = int(best_dbscan["components"])
        df_clean["cluster_final"] = df_clean["cluster_dbscan"]
    else:
        final_method = "K-means"
        final_components = best_components
        df_clean["cluster_final"] = df_clean["cluster_kmeans"]
    if df_clean["cluster_final"].isna().any() or (df_clean["cluster_final"] == -1).any():
        raise ValueError("Some customers have no final cluster.")
    df_clean.to_csv(PROCESSED_DIR / "df_clean_with_clusters.csv", index=False)

    # ADDED (Story 3): show final groups on the first two PCA axes.
    final_pca = pca_data[final_components]
    plot_data = pd.DataFrame({
        "PC1": final_pca[:, 0],
        "PC2": final_pca[:, 1],
        "cluster_final": df_clean["cluster_final"].astype(str),
    })
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=plot_data, x="PC1", y="PC2", hue="cluster_final", s=15, alpha=0.5)
    if final_method == "K-means":
        centers = best_kmeans_model.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], marker="X", s=180, c="black", label="Centroids")
        plt.legend(title="Final clusters")  # ADDED: name the black X markers in the figure.
    plt.title(f"Final clusters: {final_method}")
    save_figure("09_final_clusters.png")

    # ADDED: record the modeling choice in plain language for the report.
    summary = (
        f"Rows after cleaning: {len(df_clean)}\n"
        f"PCA reaches 80% variance with {first_80} components.\n"
        f"K-means choice: {best_components} components, k={best_k}, "
        f"silhouette={best_kmeans['silhouette']:.3f}, "
        f"smallest cluster={best_kmeans['smallest_cluster_pct']:.1f}%.\n"
        f"DBSCAN best silhouette: {db_key[0]} components, eps={db_key[1]}, "
        f"min_samples={db_key[2]}, silhouette={best_dbscan['silhouette']:.3f}, "
        f"noise={best_dbscan['noise_pct']:.1f}%, "
        f"smallest cluster={best_dbscan['smallest_cluster_pct']:.1f}%, "
        f"largest cluster={best_dbscan['largest_cluster_pct']:.1f}%.\n"
        f"Final method: {final_method}. DBSCAN is selected only with zero noise, "
        f"balanced clusters, and silhouette at least 0.05 higher than K-means.\n"
        "Silhouette scores in both searches use a fixed sample of up to 1000 rows. "
        "DBSCAN scores exclude noise; its noise percentage is reported separately.\n"
    )
    (PROCESSED_DIR / "clustering_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
