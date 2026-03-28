from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

METRICS_PATH = MODELS_DIR / "cluster_metrics.csv"
ASSIGNMENTS_PATH = MODELS_DIR / "cluster_assignments.parquet"
CANDIDATES_PATH = MODELS_DIR / "queenless_window_candidates.csv"
TOP20_PATH = MODELS_DIR / "queenless_top20.csv"
MULTIMODAL_PATH = PROJECT_ROOT / "outputs" / "features" / "multimodal_hourly_features.parquet"
MODULATION_PATH = PROJECT_ROOT / "outputs" / "features" / "modulation_hourly_features.parquet"

PCA_FILES = {
    "By Hive": FIGURES_DIR / "pca_by_hive.png",
    "KMeans Clusters": FIGURES_DIR / "pca_by_kmeans_cluster.png",
    "GMM Clusters": FIGURES_DIR / "pca_by_gmm_cluster.png",
    "DBSCAN Clusters": FIGURES_DIR / "pca_by_dbscan_cluster.png",
    "KMeans (full features)": FIGURES_DIR / "pca_kmeans.png",
    "GMM (full features)": FIGURES_DIR / "pca_gmm.png",
    "DBSCAN (full features)": FIGURES_DIR / "pca_dbscan.png",
}


@st.cache_data(show_spinner=False)
def load_metrics(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_assignments(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "hour_bucket_utc" in df.columns:
        df["hour_bucket_utc"] = pd.to_datetime(df["hour_bucket_utc"], utc=True, errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_feature_timeseries(multimodal_path: Path, modulation_path: Path) -> pd.DataFrame:
    multi = pd.read_parquet(multimodal_path)
    mod = pd.read_parquet(modulation_path)

    join_cols = [
        c
        for c in ["file_path", "file_name", "hive", "timestamp_utc"]
        if c in multi.columns and c in mod.columns
    ]
    if "file_path" not in join_cols:
        return multi

    # Keep only modulation columns that don't already exist in multimodal table.
    mod_keep = join_cols + [c for c in mod.columns if c not in join_cols and c not in multi.columns]
    merged = multi.merge(mod[mod_keep], on=join_cols, how="left")

    if "timestamp_utc" in merged.columns:
        merged["timestamp_utc"] = pd.to_datetime(merged["timestamp_utc"], utc=True, errors="coerce")
    return merged


def missing_files() -> list[Path]:
    required = [
        METRICS_PATH,
        ASSIGNMENTS_PATH,
        CANDIDATES_PATH,
        TOP20_PATH,
        MULTIMODAL_PATH,
        MODULATION_PATH,
    ]
    return [p for p in required if not p.exists()]


def kpi_metrics(metrics: pd.DataFrame, candidates: pd.DataFrame) -> None:
    kmeans_row = metrics.loc[metrics["model"] == "kmeans"]
    gmm_row = metrics.loc[metrics["model"] == "gaussian_mixture"]
    dbscan_row = metrics.loc[metrics["model"] == "dbscan"]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        value = float(kmeans_row["silhouette"].iloc[0]) if not kmeans_row.empty else float("nan")
        st.metric("KMeans Silhouette", f"{value:.3f}")

    with c2:
        value = float(gmm_row["silhouette"].iloc[0]) if not gmm_row.empty else float("nan")
        st.metric("GMM Silhouette", f"{value:.3f}")

    with c3:
        if not dbscan_row.empty:
            n_noise = int(dbscan_row["n_noise"].iloc[0])
            n_samples = int(dbscan_row["n_samples"].iloc[0])
            noise_ratio = n_noise / n_samples if n_samples else 0.0
            st.metric("DBSCAN Noise Ratio", f"{noise_ratio:.1%}")
        else:
            st.metric("DBSCAN Noise Ratio", "N/A")

    with c4:
        n_candidates = int(candidates.get("queenless_candidate", pd.Series(dtype=bool)).sum())
        st.metric("Queenless Candidates", f"{n_candidates}")


def _available_numeric_features(df: pd.DataFrame) -> list[str]:
    blocked = {
        "hour",
        "month_dir",
        "day_dir",
        "pca_1",
        "pca_2",
        "cluster_kmeans",
        "cluster_gmm",
        "cluster_dbscan",
    }
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    return [c for c in numeric_cols if c not in blocked]


def _sensor_accel_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("sens_")]


def _render_time_series(
    df: pd.DataFrame,
    feature: str,
    overlay_cluster: str | None,
    title: str,
) -> None:
    if df.empty:
        st.info("No rows match the selected filters.")
        return

    plot_df = df[["timestamp_utc", "hive", feature] + ([overlay_cluster] if overlay_cluster else [])].copy()
    plot_df = plot_df.dropna(subset=["timestamp_utc", feature])
    if plot_df.empty:
        st.info("No valid points available for this feature in the selected range.")
        return

    if overlay_cluster:
        fig = px.scatter(
            plot_df,
            x="timestamp_utc",
            y=feature,
            color=overlay_cluster,
            symbol="hive",
            title=title,
            opacity=0.75,
        )
    else:
        fig = px.line(
            plot_df,
            x="timestamp_utc",
            y=feature,
            color="hive",
            title=title,
        )

    fig.update_layout(legend_title_text="Legend", margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="HiveNavigator Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("HiveNavigator: Unsupervised Hive State Dashboard")
    st.caption("Audio + modulation + sensor clustering with queenless-window candidate surfacing")

    missing = missing_files()
    if missing:
        st.error("Missing required output files. Run the feature and notebook pipeline first.")
        for p in missing:
            st.write(f"- {p.relative_to(PROJECT_ROOT)}")
        st.stop()

    metrics = load_metrics(METRICS_PATH)
    assignments = load_assignments(ASSIGNMENTS_PATH)
    candidates = load_candidates(CANDIDATES_PATH)
    top20 = load_candidates(TOP20_PATH)
    ts_df = load_feature_timeseries(MULTIMODAL_PATH, MODULATION_PATH)

    # Join cluster labels into time-series frame for overlay.
    if "file_path" in ts_df.columns and "file_path" in assignments.columns:
        label_cols = [c for c in ["file_path", "cluster_kmeans", "cluster_gmm", "cluster_dbscan"] if c in assignments.columns]
        if label_cols:
            ts_df = ts_df.merge(assignments[label_cols], on="file_path", how="left")

    if "timestamp_utc" in ts_df.columns:
        ts_df = ts_df.dropna(subset=["timestamp_utc"]).copy()

    st.sidebar.header("Filters")
    hive_options = sorted(ts_df["hive"].dropna().unique().tolist()) if "hive" in ts_df.columns else []
    selected_hives = st.sidebar.multiselect("Hive toggles", hive_options, default=hive_options)

    min_date = ts_df["timestamp_utc"].dt.date.min() if "timestamp_utc" in ts_df.columns and not ts_df.empty else None
    max_date = ts_df["timestamp_utc"].dt.date.max() if "timestamp_utc" in ts_df.columns and not ts_df.empty else None
    if min_date is not None and max_date is not None:
        date_range = st.sidebar.date_input("Date range filter", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range
    else:
        start_date = end_date = None

    filtered_ts = ts_df.copy()
    if selected_hives and "hive" in filtered_ts.columns:
        filtered_ts = filtered_ts[filtered_ts["hive"].isin(selected_hives)]
    if start_date is not None and end_date is not None and "timestamp_utc" in filtered_ts.columns:
        mask = (filtered_ts["timestamp_utc"].dt.date >= start_date) & (filtered_ts["timestamp_utc"].dt.date <= end_date)
        filtered_ts = filtered_ts[mask]

    filtered_ts = filtered_ts.sort_values("timestamp_utc") if "timestamp_utc" in filtered_ts.columns else filtered_ts

    st.subheader("Model Performance Snapshot")
    kpi_metrics(metrics, candidates)
    st.dataframe(metrics, use_container_width=True)

    st.subheader("Interactive Time-Series Explorer")
    feature_options = _available_numeric_features(filtered_ts)
    default_primary = [
        c
        for c in [
            "band_mid_180_280_power_mean",
            "band_high_450_530_power_mean",
            "mfcc_01_mean",
        ]
        if c in feature_options
    ]
    selected_features = st.multiselect(
        "Feature selector",
        options=feature_options,
        default=default_primary[:2] if default_primary else feature_options[:1],
    )

    cluster_overlay = st.selectbox(
        "Cluster overlay",
        options=["None", "cluster_kmeans", "cluster_gmm", "cluster_dbscan"],
        index=1,
    )
    overlay_col = None if cluster_overlay == "None" else cluster_overlay

    for feature in selected_features:
        _render_time_series(
            filtered_ts,
            feature=feature,
            overlay_cluster=overlay_col if overlay_col in filtered_ts.columns else None,
            title=f"{feature} over time",
        )

    st.subheader("Optional Secondary Panel: Sensor and Accelerometer")
    show_secondary = st.checkbox("Show sensor/accel panel", value=True)
    if show_secondary:
        sensor_features = _sensor_accel_features(filtered_ts)
        if sensor_features:
            secondary_feature = st.selectbox("Secondary panel feature", options=sensor_features, index=0)
            _render_time_series(
                filtered_ts,
                feature=secondary_feature,
                overlay_cluster=overlay_col if overlay_col in filtered_ts.columns else None,
                title=f"Secondary panel: {secondary_feature}",
            )
        else:
            st.info("No sensor/accel columns found in the loaded feature table.")

    st.subheader("PCA Visual Diagnostics")
    selected_plot = st.selectbox("Select PCA view", list(PCA_FILES.keys()), index=0)
    pca_path = PCA_FILES[selected_plot]
    if pca_path.exists():
        st.image(str(pca_path), caption=selected_plot, use_container_width=True)
    else:
        st.warning(f"Image not found: {pca_path.relative_to(PROJECT_ROOT)}")

    st.subheader("Cluster Assignment Explorer")
    assignment_hives = sorted(assignments["hive"].dropna().unique().tolist()) if "hive" in assignments.columns else []

    filtered_assignments = assignments.copy()
    if selected_hives and "hive" in filtered_assignments.columns:
        filtered_assignments = filtered_assignments[filtered_assignments["hive"].isin(selected_hives)]

    if start_date is not None and end_date is not None and "timestamp_utc" in filtered_assignments.columns:
        t = pd.to_datetime(filtered_assignments["timestamp_utc"], utc=True, errors="coerce")
        mask = (t.dt.date >= start_date) & (t.dt.date <= end_date)
        filtered_assignments = filtered_assignments[mask]

    if "timestamp_utc" in filtered_assignments.columns:
        filtered_assignments = filtered_assignments.sort_values("timestamp_utc")

    st.dataframe(filtered_assignments.head(200), use_container_width=True)
    st.caption("Showing first 200 filtered rows for quick inspection.")

    st.subheader("Queenless Candidate Windows")
    only_flagged = st.checkbox("Show only flagged candidate windows", value=True)
    candidate_view = candidates.copy()

    if selected_hives and "hive" in candidate_view.columns:
        candidate_view = candidate_view[candidate_view["hive"].isin(selected_hives)]

    if start_date is not None and end_date is not None and "hour_bucket_utc" in candidate_view.columns:
        t = pd.to_datetime(candidate_view["hour_bucket_utc"], utc=True, errors="coerce")
        mask = (t.dt.date >= start_date) & (t.dt.date <= end_date)
        candidate_view = candidate_view[mask]

    if only_flagged and "queenless_candidate" in candidate_view.columns:
        candidate_view = candidate_view[candidate_view["queenless_candidate"] == True]

    if "tvd_zscore" in candidate_view.columns:
        candidate_view = candidate_view.sort_values("tvd_zscore", ascending=False)

    st.dataframe(candidate_view, use_container_width=True)

    st.subheader("Top 20 Candidate Windows")
    top20_view = top20.copy()
    if selected_hives and "hive" in top20_view.columns:
        top20_view = top20_view[top20_view["hive"].isin(selected_hives)]
    st.dataframe(top20_view, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            label="Download Cluster Metrics CSV",
            data=METRICS_PATH.read_bytes(),
            file_name="cluster_metrics.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            label="Download Queenless Top20 CSV",
            data=TOP20_PATH.read_bytes(),
            file_name="queenless_top20.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
