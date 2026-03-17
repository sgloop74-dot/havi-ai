from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


NON_FEATURE_COLS = {
    "file_path",
    "file_name",
    "hive",
    "timestamp_utc",
    "date",
    "hour",
    "month_dir",
    "day_dir",
    "hour_bucket_utc",
}


def load_feature_frame(
    audio_path: str | Path = "outputs/features/audio_hourly_features.parquet",
    modulation_path: str | Path = "outputs/features/modulation_hourly_features.parquet",
    multimodal_path: str | Path = "outputs/features/multimodal_hourly_features.parquet",
) -> pd.DataFrame:
    """Load the best available feature table.

    Priority:
    1) multimodal_hourly_features.parquet
    2) merge audio + modulation on file_path
    3) audio only
    """
    audio_path = Path(audio_path)
    modulation_path = Path(modulation_path)
    multimodal_path = Path(multimodal_path)

    if multimodal_path.exists():
        df = pd.read_parquet(multimodal_path)
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        return df

    if audio_path.exists() and modulation_path.exists():
        audio_df = pd.read_parquet(audio_path)
        mod_df = pd.read_parquet(modulation_path)

        join_cols = [c for c in ["file_path", "file_name", "hive", "timestamp_utc"] if c in audio_df.columns and c in mod_df.columns]
        if "file_path" in join_cols:
            merged = audio_df.merge(mod_df, on=join_cols, how="left", suffixes=("", "_mod"))
        else:
            merged = audio_df.copy()

        if "timestamp_utc" in merged.columns:
            merged["timestamp_utc"] = pd.to_datetime(merged["timestamp_utc"], utc=True, errors="coerce")
        return merged

    if audio_path.exists():
        df = pd.read_parquet(audio_path)
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        return df

    raise FileNotFoundError("No feature table found. Run feature extraction first.")


def build_feature_matrix(
    df: pd.DataFrame,
    min_non_na_ratio: float = 0.7,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build standardized feature matrix from numeric columns with robust filtering and imputation."""
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    numeric_cols = [c for c in df.select_dtypes(include=[np.number, bool]).columns if c not in NON_FEATURE_COLS]
    if not numeric_cols:
        raise ValueError("No numeric feature columns found for clustering.")

    feature_df = df[numeric_cols].copy()

    min_non_na = int(np.ceil(len(feature_df) * min_non_na_ratio))
    feature_df = feature_df.dropna(axis=1, thresh=min_non_na)

    if feature_df.empty:
        raise ValueError("All feature columns were dropped due to missingness.")

    feature_cols = list(feature_df.columns)
    imputer = SimpleImputer(strategy="median")
    x_imputed = imputer.fit_transform(feature_df)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_imputed)

    return x_scaled, feature_cols, x_imputed


def evaluate_clustering(x_scaled: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute silhouette and Davies-Bouldin scores, handling noise labels for DBSCAN."""
    labels = np.asarray(labels)

    if labels.size == 0:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "n_clusters": 0}

    valid_mask = labels >= 0
    unique_non_noise = np.unique(labels[valid_mask]) if np.any(valid_mask) else np.array([])
    n_clusters = int(len(unique_non_noise))

    if n_clusters <= 1:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "n_clusters": n_clusters}

    x_eval = x_scaled[valid_mask]
    y_eval = labels[valid_mask]

    return {
        "silhouette": float(silhouette_score(x_eval, y_eval)),
        "davies_bouldin": float(davies_bouldin_score(x_eval, y_eval)),
        "n_clusters": n_clusters,
    }


def run_experiments(
    df: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
    dbscan_eps: float = 1.2,
    dbscan_min_samples: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run KMeans, Gaussian Mixture, and DBSCAN on standardized features and return assignments + metrics."""
    x_scaled, feature_cols, _ = build_feature_matrix(df)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels_km = km.fit_predict(x_scaled)

    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state, covariance_type="full")
    labels_gmm = gmm.fit_predict(x_scaled)

    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    labels_db = db.fit_predict(x_scaled)

    pca = PCA(n_components=2, random_state=random_state)
    pca_xy = pca.fit_transform(x_scaled)

    keep_cols = [c for c in ["file_path", "file_name", "hive", "timestamp_utc", "date", "hour"] if c in df.columns]
    assignments = df[keep_cols].copy()
    assignments["cluster_kmeans"] = labels_km
    assignments["cluster_gmm"] = labels_gmm
    assignments["cluster_dbscan"] = labels_db
    assignments["pca_1"] = pca_xy[:, 0]
    assignments["pca_2"] = pca_xy[:, 1]

    metrics_rows: list[dict[str, Any]] = []

    for model_name, labels in [
        ("kmeans", labels_km),
        ("gaussian_mixture", labels_gmm),
        ("dbscan", labels_db),
    ]:
        score = evaluate_clustering(x_scaled, labels)
        score["model"] = model_name
        score["n_samples"] = int(len(labels))
        score["n_noise"] = int(np.sum(np.asarray(labels) == -1))
        metrics_rows.append(score)

    metrics = pd.DataFrame(metrics_rows)[
        ["model", "n_samples", "n_clusters", "n_noise", "silhouette", "davies_bouldin"]
    ]

    if "timestamp_utc" in assignments.columns:
        assignments["timestamp_utc"] = pd.to_datetime(assignments["timestamp_utc"], utc=True, errors="coerce")

    assignments.attrs["feature_columns"] = feature_cols
    assignments.attrs["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()

    return assignments, metrics


def save_clustering_outputs(
    assignments: pd.DataFrame,
    metrics: pd.DataFrame,
    assignments_path: str | Path = "outputs/models/cluster_assignments.parquet",
    metrics_path: str | Path = "outputs/models/cluster_metrics.csv",
) -> None:
    """Persist clustering outputs to parquet/csv."""
    assignments_path = Path(assignments_path)
    metrics_path = Path(metrics_path)
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    assignments.to_parquet(assignments_path, index=False)
    metrics.to_csv(metrics_path, index=False)


def plot_pca_scatter(
    assignments: pd.DataFrame,
    color_col: str,
    title: str,
    output_path: str | Path | None = None,
) -> None:
    """Create PCA scatter plot colored by a selected column."""
    if assignments.empty:
        raise ValueError("assignments DataFrame is empty")

    fig, ax = plt.subplots(figsize=(9, 6))
    values = assignments[color_col].astype(str)

    for value in sorted(values.unique()):
        mask = values == value
        ax.scatter(assignments.loc[mask, "pca_1"], assignments.loc[mask, "pca_2"], s=16, alpha=0.7, label=value)

    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)


def summarize_cluster_distribution(assignments: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """Return per-hive cluster counts and proportions for interpretability."""
    if cluster_col not in assignments.columns:
        raise ValueError(f"Missing cluster column: {cluster_col}")

    if "hive" not in assignments.columns:
        raise ValueError("assignments must include hive column")

    counts = (
        assignments.groupby(["hive", cluster_col], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["hive", cluster_col])
    )

    totals = counts.groupby("hive")["count"].transform("sum")
    counts["pct_within_hive"] = np.where(totals > 0, counts["count"] / totals, np.nan)
    return counts
