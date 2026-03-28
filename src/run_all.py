from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from clustering import (
    load_feature_frame,
    plot_pca_scatter,
    run_experiments,
    save_clustering_outputs,
    summarize_cluster_distribution,
)
from io_utils import build_audio_inventory, save_inventory_outputs


VALID_STAGES = ["inventory", "audio", "modulation", "integration", "clustering"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HiveNavigator pipeline end-to-end with cache-aware stage skipping."
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=",".join(VALID_STAGES),
        help="Comma-separated stages to run. Choices: inventory,audio,modulation,integration,clustering",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run selected stages even if expected outputs already exist.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Pass skip-errors to audio/modulation extractors.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("data/audio"),
        help="Audio root folder used by inventory/audio/modulation stages.",
    )
    parser.add_argument(
        "--sensors-root",
        type=Path,
        default=Path("data/sensors"),
        help="Sensor root folder used by integration stage.",
    )
    parser.add_argument(
        "--audio-limit",
        type=int,
        default=None,
        help="Optional file limit passed to audio and modulation extractors for quick tests.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=4,
        help="Cluster count for KMeans/GMM in clustering stage.",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=1.2,
        help="DBSCAN eps for clustering stage.",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=12,
        help="DBSCAN min_samples for clustering stage.",
    )
    return parser.parse_args()


def parse_stages(raw: str) -> list[str]:
    stages = [s.strip().lower() for s in raw.split(",") if s.strip()]
    invalid = [s for s in stages if s not in VALID_STAGES]
    if invalid:
        raise ValueError(f"Invalid stage(s): {invalid}. Valid stages: {VALID_STAGES}")
    # Preserve execution order by filtering VALID_STAGES
    return [s for s in VALID_STAGES if s in stages]


def run_subprocess(command: list[str], project_root: Path) -> None:
    print(f"[run] {' '.join(command)}")
    subprocess.run(command, cwd=project_root, check=True)


def should_skip(stage: str, force: bool, output_paths: list[Path]) -> bool:
    if force:
        return False
    return all(p.exists() for p in output_paths)


def run_inventory_stage(project_root: Path, audio_root: Path, force: bool) -> None:
    output_csv = project_root / "outputs" / "data_inventory.csv"
    report_md = project_root / "outputs" / "data_quality_report.md"

    if should_skip("inventory", force, [output_csv, report_md]):
        print("[skip] inventory (cached outputs found)")
        return

    print("[stage] inventory")
    inv = build_audio_inventory(audio_root)
    save_inventory_outputs(inv, output_csv, report_md)
    print(f"[done] inventory -> {output_csv} and {report_md}")


def run_audio_stage(
    project_root: Path,
    audio_root: Path,
    force: bool,
    skip_errors: bool,
    audio_limit: int | None,
) -> None:
    output_path = project_root / "outputs" / "features" / "audio_hourly_features.parquet"

    if should_skip("audio", force, [output_path]):
        print("[skip] audio (cached outputs found)")
        return

    print("[stage] audio")
    command = [
        sys.executable,
        "src/run_audio_features.py",
        "--audio-root",
        str(audio_root),
        "--output-path",
        str(output_path),
    ]
    if audio_limit is not None:
        command.extend(["--limit", str(audio_limit)])
    if skip_errors:
        command.append("--skip-errors")

    run_subprocess(command, project_root)
    print(f"[done] audio -> {output_path}")


def run_modulation_stage(
    project_root: Path,
    audio_root: Path,
    force: bool,
    skip_errors: bool,
    audio_limit: int | None,
) -> None:
    output_path = project_root / "outputs" / "features" / "modulation_hourly_features.parquet"

    if should_skip("modulation", force, [output_path]):
        print("[skip] modulation (cached outputs found)")
        return

    print("[stage] modulation")
    command = [
        sys.executable,
        "src/run_modulation_features.py",
        "--audio-root",
        str(audio_root),
        "--output-path",
        str(output_path),
    ]
    if audio_limit is not None:
        command.extend(["--limit", str(audio_limit)])
    if skip_errors:
        command.append("--skip-errors")

    run_subprocess(command, project_root)
    print(f"[done] modulation -> {output_path}")


def run_integration_stage(project_root: Path, sensors_root: Path, force: bool) -> None:
    multimodal_path = project_root / "outputs" / "features" / "multimodal_hourly_features.parquet"
    sensor_path = project_root / "outputs" / "features" / "sensor_hourly_features.parquet"
    accel_path = project_root / "outputs" / "features" / "accel_hourly_features.parquet"

    if should_skip("integration", force, [multimodal_path, sensor_path, accel_path]):
        print("[skip] integration (cached outputs found)")
        return

    print("[stage] integration")
    command = [
        sys.executable,
        "src/run_sensor_integration.py",
        "--sensors-root",
        str(sensors_root),
        "--audio-features",
        str(project_root / "outputs" / "features" / "audio_hourly_features.parquet"),
        "--output-path",
        str(multimodal_path),
        "--sensor-hourly-out",
        str(sensor_path),
        "--accel-hourly-out",
        str(accel_path),
        "--summary-out",
        str(project_root / "outputs" / "features" / "sensor_integration_summary.md"),
    ]
    run_subprocess(command, project_root)
    print(f"[done] integration -> {multimodal_path}")


def run_clustering_stage(
    project_root: Path,
    force: bool,
    n_clusters: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
) -> None:
    assignments_path = project_root / "outputs" / "models" / "cluster_assignments.parquet"
    metrics_path = project_root / "outputs" / "models" / "cluster_metrics.csv"

    if should_skip("clustering", force, [assignments_path, metrics_path]):
        print("[skip] clustering (cached outputs found)")
        return

    print("[stage] clustering")
    feature_df = load_feature_frame(
        audio_path=project_root / "outputs" / "features" / "audio_hourly_features.parquet",
        modulation_path=project_root / "outputs" / "features" / "modulation_hourly_features.parquet",
        multimodal_path=project_root / "outputs" / "features" / "multimodal_hourly_features.parquet",
    )

    assignments, metrics = run_experiments(
        feature_df,
        n_clusters=n_clusters,
        random_state=42,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
    )
    save_clustering_outputs(assignments, metrics, assignments_path=assignments_path, metrics_path=metrics_path)

    figures_dir = project_root / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_pca_scatter(assignments, "hive", "PCA colored by hive", figures_dir / "pca_by_hive.png")
    plot_pca_scatter(
        assignments,
        "cluster_kmeans",
        "PCA colored by KMeans cluster",
        figures_dir / "pca_by_kmeans_cluster.png",
    )
    plot_pca_scatter(
        assignments,
        "cluster_gmm",
        "PCA colored by GMM cluster",
        figures_dir / "pca_by_gmm_cluster.png",
    )
    plot_pca_scatter(
        assignments,
        "cluster_dbscan",
        "PCA colored by DBSCAN cluster",
        figures_dir / "pca_by_dbscan_cluster.png",
    )

    for col, name in [
        ("cluster_kmeans", "kmeans"),
        ("cluster_gmm", "gmm"),
        ("cluster_dbscan", "dbscan"),
    ]:
        dist = summarize_cluster_distribution(assignments, col)
        dist.to_csv(project_root / "outputs" / "models" / f"cluster_distribution_{name}.csv", index=False)

    print(f"[done] clustering -> {assignments_path} and {metrics_path}")


def main() -> None:
    args = parse_args()
    stages = parse_stages(args.stages)

    project_root = Path(__file__).resolve().parents[1]

    audio_root = args.audio_root if args.audio_root.is_absolute() else project_root / args.audio_root
    sensors_root = args.sensors_root if args.sensors_root.is_absolute() else project_root / args.sensors_root

    print(f"Project root: {project_root}")
    print(f"Stages: {stages}")
    print(f"Force rerun: {args.force}")

    for stage in stages:
        if stage == "inventory":
            run_inventory_stage(project_root, audio_root, args.force)
        elif stage == "audio":
            run_audio_stage(project_root, audio_root, args.force, args.skip_errors, args.audio_limit)
        elif stage == "modulation":
            run_modulation_stage(project_root, audio_root, args.force, args.skip_errors, args.audio_limit)
        elif stage == "integration":
            run_integration_stage(project_root, sensors_root, args.force)
        elif stage == "clustering":
            run_clustering_stage(
                project_root,
                args.force,
                n_clusters=args.n_clusters,
                dbscan_eps=args.dbscan_eps,
                dbscan_min_samples=args.dbscan_min_samples,
            )

    print("Pipeline finished.")


if __name__ == "__main__":
    main()
