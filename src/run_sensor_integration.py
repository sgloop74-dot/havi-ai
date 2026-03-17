from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sensor_features import (
    build_hourly_sensor_accel_features,
    merge_with_audio_hourly,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge hourly sensor and accel features with audio features.")
    parser.add_argument("--sensors-root", type=Path, default=Path("data/sensors"), help="Root of sensor CSV files.")
    parser.add_argument(
        "--audio-features",
        type=Path,
        default=Path("outputs/features/audio_hourly_features.parquet"),
        help="Input parquet with per-file audio features.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/features/multimodal_hourly_features.parquet"),
        help="Output parquet with merged multimodal features.",
    )
    parser.add_argument(
        "--sensor-hourly-out",
        type=Path,
        default=Path("outputs/features/sensor_hourly_features.parquet"),
        help="Optional output parquet for hourly environmental sensor features.",
    )
    parser.add_argument(
        "--accel-hourly-out",
        type=Path,
        default=Path("outputs/features/accel_hourly_features.parquet"),
        help="Optional output parquet for hourly accelerometer FFT features.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("outputs/features/sensor_integration_summary.md"),
        help="Markdown summary of sensor and accel coverage.",
    )
    return parser.parse_args()


def _summary_to_markdown(summary_tables: dict[str, pd.DataFrame]) -> str:
    lines: list[str] = ["# Sensor Integration Summary", ""]
    for name, df in summary_tables.items():
        lines.append(f"## {name}")
        if df.empty:
            lines.append("No data available.")
        else:
            lines.append(df.to_markdown(index=False))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    if not args.audio_features.exists():
        raise SystemExit(f"Audio feature file not found: {args.audio_features}")

    audio_df = pd.read_parquet(args.audio_features)
    sensor_hourly, accel_hourly, summary_tables = build_hourly_sensor_accel_features(args.sensors_root)

    merged_df = merge_with_audio_hourly(audio_df, sensor_hourly, accel_hourly)

    args.sensor_hourly_out.parent.mkdir(parents=True, exist_ok=True)
    args.accel_hourly_out.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)

    sensor_hourly.to_parquet(args.sensor_hourly_out, index=False)
    accel_hourly.to_parquet(args.accel_hourly_out, index=False)
    merged_df.to_parquet(args.output_path, index=False)
    args.summary_out.write_text(_summary_to_markdown(summary_tables), encoding="utf-8")

    print(f"Saved sensor hourly: {args.sensor_hourly_out}")
    print(f"Saved accel hourly: {args.accel_hourly_out}")
    print(f"Saved multimodal merged features: {args.output_path}")
    print(f"Saved integration summary: {args.summary_out}")
    print(f"Merged rows: {len(merged_df)}")


if __name__ == "__main__":
    main()
