from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


HIVE_ID_REGEX = re.compile(r"hive_(\d+)", re.IGNORECASE)


def parse_hive_from_path(path: str | Path) -> str:
    """Extract normalized hive id (e.g., hive_03) from a file path."""
    match = HIVE_ID_REGEX.search(str(path))
    if not match:
        raise ValueError(f"Could not parse hive id from path: {path}")
    return f"hive_{int(match.group(1)):02d}"


def load_sensor_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load environmental sensor CSV and parse timestamp as UTC."""
    path = Path(csv_path)
    df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        raise ValueError(f"Missing timestamp column in {path}")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).copy()

    hive = parse_hive_from_path(path)
    df["hive"] = hive

    # Keep numeric columns only for aggregation.
    value_cols = [
        "sht_t",
        "sht_h",
        "co2",
        "scd_t",
        "scd_h",
        "accel_x",
        "accel_y",
        "accel_z",
        "rssi",
    ]
    for col in value_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_accel_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load FFT-based accelerometer CSV and parse timestamp as UTC."""
    path = Path(csv_path)
    df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        raise ValueError(f"Missing timestamp column in {path}")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).copy()

    hive = parse_hive_from_path(path)
    df["hive"] = hive

    for col in ["f1", "m1", "f2", "m2", "f3", "m3"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _hourly_trend(series: pd.Series, times: pd.Series) -> float:
    """Compute linear trend slope per hour using simple least squares."""
    valid = ~(series.isna() | times.isna())
    if valid.sum() < 2:
        return np.nan

    y = series[valid].to_numpy(dtype=float)
    t = (times[valid] - times[valid].min()).dt.total_seconds().to_numpy(dtype=float)

    if np.allclose(t.std(), 0):
        return np.nan

    slope_per_second = np.polyfit(t, y, 1)[0]
    return float(slope_per_second * 3600.0)


def resample_hourly_features(
    df: pd.DataFrame,
    value_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Aggregate a timestamped table to hourly stats with trend features."""
    if df.empty:
        return pd.DataFrame(columns=["hive", "hour_bucket_utc"])

    work = df.copy()
    work["hour_bucket_utc"] = work["timestamp_utc"].dt.floor("h")

    rows: list[dict[str, Any]] = []

    for (hive, hour), g in work.groupby(["hive", "hour_bucket_utc"], dropna=False):
        row: dict[str, Any] = {
            "hive": hive,
            "hour_bucket_utc": hour,
            f"{prefix}_samples": int(len(g)),
        }

        for col in value_columns:
            if col not in g.columns:
                continue
            s = g[col]
            row[f"{prefix}_{col}_mean"] = float(s.mean()) if s.notna().any() else np.nan
            row[f"{prefix}_{col}_std"] = float(s.std(ddof=0)) if s.notna().any() else np.nan
            row[f"{prefix}_{col}_min"] = float(s.min()) if s.notna().any() else np.nan
            row[f"{prefix}_{col}_max"] = float(s.max()) if s.notna().any() else np.nan
            row[f"{prefix}_{col}_trend_per_hour"] = _hourly_trend(s, g["timestamp_utc"])

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["hour_bucket_utc"] = pd.to_datetime(out["hour_bucket_utc"], utc=True, errors="coerce")
    return out.sort_values(["hive", "hour_bucket_utc"]).reset_index(drop=True)


def build_hourly_sensor_accel_features(sensors_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load hive_03/hive_04 sensor+accel CSVs and return hourly feature tables plus summaries."""
    root = Path(sensors_root)
    sensor_paths = sorted(root.glob("hive_*/*/*_sensors.csv"))
    accel_paths = sorted(root.glob("hive_*/*/*_accel.csv"))

    sensor_frames = [load_sensor_csv(p) for p in sensor_paths]
    accel_frames = [load_accel_csv(p) for p in accel_paths]

    sensor_raw = pd.concat(sensor_frames, ignore_index=True) if sensor_frames else pd.DataFrame()
    accel_raw = pd.concat(accel_frames, ignore_index=True) if accel_frames else pd.DataFrame()

    sensor_cols = ["sht_t", "sht_h", "co2", "scd_t", "scd_h", "accel_x", "accel_y", "accel_z", "rssi"]
    accel_cols = ["f1", "m1", "f2", "m2", "f3", "m3"]

    sensor_hourly = resample_hourly_features(sensor_raw, sensor_cols, prefix="sens")
    accel_hourly = resample_hourly_features(accel_raw, accel_cols, prefix="acc")

    summary_tables: dict[str, pd.DataFrame] = {}

    if not sensor_raw.empty:
        summary_tables["sensor_summary_by_hive"] = (
            sensor_raw.groupby("hive", dropna=False)
            .agg(rows=("timestamp_utc", "count"), min_ts=("timestamp_utc", "min"), max_ts=("timestamp_utc", "max"))
            .reset_index()
            .sort_values("hive")
        )
    else:
        summary_tables["sensor_summary_by_hive"] = pd.DataFrame()

    if not accel_raw.empty:
        summary_tables["accel_summary_by_hive"] = (
            accel_raw.groupby("hive", dropna=False)
            .agg(rows=("timestamp_utc", "count"), min_ts=("timestamp_utc", "min"), max_ts=("timestamp_utc", "max"))
            .reset_index()
            .sort_values("hive")
        )
    else:
        summary_tables["accel_summary_by_hive"] = pd.DataFrame()

    return sensor_hourly, accel_hourly, summary_tables


def merge_with_audio_hourly(
    audio_features_df: pd.DataFrame,
    sensor_hourly_df: pd.DataFrame,
    accel_hourly_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join audio with hourly sensor and accel features on hive + UTC hour bucket."""
    if audio_features_df.empty:
        return audio_features_df.copy()

    audio = audio_features_df.copy()
    audio["timestamp_utc"] = pd.to_datetime(audio["timestamp_utc"], utc=True, errors="coerce")
    audio["hour_bucket_utc"] = audio["timestamp_utc"].dt.floor("h")

    merged = audio.merge(sensor_hourly_df, on=["hive", "hour_bucket_utc"], how="left")
    merged = merged.merge(accel_hourly_df, on=["hive", "hour_bucket_utc"], how="left")

    merged["has_sensor_data"] = merged.filter(regex=r"^sens_").notna().any(axis=1)
    merged["has_accel_data"] = merged.filter(regex=r"^acc_").notna().any(axis=1)

    return merged.sort_values(["hive", "timestamp_utc", "file_name"]).reset_index(drop=True)
