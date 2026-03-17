from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import pandas as pd
import soundfile as sf

AUDIO_FILE_REGEX = re.compile(r"(?P<hive>hive_\d+)_(?P<date>\d{8})_(?P<time>\d{6})\.flac$", re.IGNORECASE)


@dataclass
class AudioPathInfo:
    hive: str
    timestamp_utc: pd.Timestamp
    month_dir: int | None
    day_dir: int | None


def scan_audio_files(audio_root: str | Path) -> list[Path]:
    """Recursively return all FLAC files under the audio root."""
    root = Path(audio_root)
    return sorted(root.rglob("*.flac"))


def parse_audio_path(file_path: str | Path) -> AudioPathInfo:
    """Parse hive and UTC timestamp from path/filename metadata."""
    path = Path(file_path)

    match = AUDIO_FILE_REGEX.match(path.name)
    if not match:
        raise ValueError(f"Unexpected audio filename format: {path.name}")

    hive_from_name = match.group("hive").lower()
    dt = pd.to_datetime(
        f"{match.group('date')}{match.group('time')}",
        format="%Y%m%d%H%M%S",
        utc=True,
    )

    parts_lower = [part.lower() for part in path.parts]
    hive_from_dir = next((part for part in parts_lower if part.startswith("hive_")), None)
    if hive_from_dir and hive_from_dir != hive_from_name:
        raise ValueError(
            f"Hive mismatch between folder ({hive_from_dir}) and filename ({hive_from_name}) in {path}"
        )

    month_dir = None
    day_dir = None
    try:
        if path.parent.name.isdigit():
            day_dir = int(path.parent.name)
        if path.parent.parent.name.isdigit():
            month_dir = int(path.parent.parent.name)
    except (IndexError, ValueError):
        month_dir = None
        day_dir = None

    return AudioPathInfo(
        hive=hive_from_name,
        timestamp_utc=dt,
        month_dir=month_dir,
        day_dir=day_dir,
    )


def get_audio_metadata(file_path: str | Path) -> dict:
    """Read basic FLAC metadata. Raises RuntimeError for unreadable/corrupt files."""
    try:
        info = sf.info(str(file_path))
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to read audio metadata: {exc}") from exc

    duration_s = info.frames / info.samplerate if info.samplerate else None
    return {
        "samplerate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_s": float(duration_s) if duration_s is not None else None,
        "format": info.format,
        "subtype": info.subtype,
    }


def build_audio_inventory(
    audio_root: str | Path,
    expected_sr: int = 16000,
    expected_channels: int = 1,
) -> pd.DataFrame:
    """Build a per-file inventory table and validate key metadata constraints."""
    records: list[dict] = []

    for file_path in scan_audio_files(audio_root):
        record = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "valid": True,
            "error": None,
        }

        try:
            parsed = parse_audio_path(file_path)
            meta = get_audio_metadata(file_path)
            record.update(
                {
                    "hive": parsed.hive,
                    "timestamp_utc": parsed.timestamp_utc,
                    "date": parsed.timestamp_utc.date(),
                    "hour": int(parsed.timestamp_utc.hour),
                    "month_dir": parsed.month_dir,
                    "day_dir": parsed.day_dir,
                    **meta,
                }
            )

            issues = []
            if meta["samplerate"] != expected_sr:
                issues.append(f"samplerate={meta['samplerate']} (expected {expected_sr})")
            if meta["channels"] != expected_channels:
                issues.append(f"channels={meta['channels']} (expected {expected_channels})")
            if meta["duration_s"] is None or meta["duration_s"] <= 0:
                issues.append("duration_s missing or non-positive")

            if issues:
                record["valid"] = False
                record["error"] = "; ".join(issues)

        except Exception as exc:
            record["valid"] = False
            record["error"] = str(exc)

        records.append(record)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

    return df.sort_values(["hive", "timestamp_utc", "file_name"], na_position="last").reset_index(drop=True)


def summarize_inventory(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create summary tables used in QA reporting and plotting."""
    if df.empty:
        empty = pd.DataFrame()
        return {
            "by_hive": empty,
            "by_day": empty,
            "by_hive_day": empty,
            "by_hive_hour": empty,
            "invalid": empty,
        }

    by_hive = (
        df.groupby("hive", dropna=False)
        .agg(
            files=("file_path", "count"),
            invalid_files=("valid", lambda s: int((~s).sum())),
            min_ts=("timestamp_utc", "min"),
            max_ts=("timestamp_utc", "max"),
            median_duration_s=("duration_s", "median"),
        )
        .reset_index()
        .sort_values("hive")
    )

    by_day = (
        df.assign(day=df["timestamp_utc"].dt.date)
        .groupby("day", dropna=False)
        .agg(files=("file_path", "count"), invalid_files=("valid", lambda s: int((~s).sum())))
        .reset_index()
        .sort_values("day")
    )

    by_hive_day = (
        df.assign(day=df["timestamp_utc"].dt.date)
        .groupby(["hive", "day"], dropna=False)
        .agg(files=("file_path", "count"), invalid_files=("valid", lambda s: int((~s).sum())))
        .reset_index()
        .sort_values(["hive", "day"])
    )

    by_hive_hour = (
        df.groupby(["hive", "hour"], dropna=False)
        .agg(files=("file_path", "count"), invalid_files=("valid", lambda s: int((~s).sum())))
        .reset_index()
        .sort_values(["hive", "hour"])
    )

    invalid = df.loc[~df["valid"], ["file_path", "hive", "timestamp_utc", "error"]].copy()

    return {
        "by_hive": by_hive,
        "by_day": by_day,
        "by_hive_day": by_hive_day,
        "by_hive_hour": by_hive_hour,
        "invalid": invalid,
    }


def inventory_markdown_report(df: pd.DataFrame, summaries: dict[str, pd.DataFrame]) -> str:
    """Generate a compact markdown quality report from inventory results."""
    total_files = int(len(df))
    invalid_files = int((~df["valid"]).sum()) if not df.empty else 0
    valid_pct = 100.0 * (total_files - invalid_files) / total_files if total_files else 0.0

    sr_counts = df["samplerate"].value_counts(dropna=False).sort_index() if "samplerate" in df else pd.Series(dtype=int)
    ch_counts = df["channels"].value_counts(dropna=False).sort_index() if "channels" in df else pd.Series(dtype=int)

    lines: list[str] = []
    lines.append("# Data Quality Report")
    lines.append("")
    lines.append("## Overall")
    lines.append(f"- Total audio files scanned: {total_files}")
    lines.append(f"- Invalid files: {invalid_files}")
    lines.append(f"- Valid files: {total_files - invalid_files} ({valid_pct:.2f}%)")

    if total_files:
        lines.append("")
        lines.append("## Metadata Distribution")
        lines.append("### Sample rates")
        for sr, n in sr_counts.items():
            lines.append(f"- {sr}: {int(n)}")

        lines.append("### Channels")
        for ch, n in ch_counts.items():
            lines.append(f"- {ch}: {int(n)}")

    by_hive = summaries.get("by_hive", pd.DataFrame())
    if not by_hive.empty:
        lines.append("")
        lines.append("## Files by Hive")
        lines.append(by_hive.to_markdown(index=False))

    by_day = summaries.get("by_day", pd.DataFrame())
    if not by_day.empty:
        lines.append("")
        lines.append("## Files by Day")
        lines.append(by_day.to_markdown(index=False))

    invalid = summaries.get("invalid", pd.DataFrame())
    lines.append("")
    lines.append("## Invalid File Details")
    if invalid.empty:
        lines.append("No invalid files detected.")
    else:
        lines.append(invalid.head(100).to_markdown(index=False))
        if len(invalid) > 100:
            lines.append("")
            lines.append(f"Showing first 100 invalid rows out of {len(invalid)}.")

    return "\n".join(lines) + "\n"


def save_inventory_outputs(
    inventory_df: pd.DataFrame,
    output_csv: str | Path,
    report_path: str | Path,
) -> None:
    """Persist inventory table and markdown report to disk."""
    output_csv = Path(output_csv)
    report_path = Path(report_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    inventory_df.to_csv(output_csv, index=False)
    summaries = summarize_inventory(inventory_df)
    report_path.write_text(inventory_markdown_report(inventory_df, summaries), encoding="utf-8")


def iter_sensor_csv_files(sensors_root: str | Path) -> Iterable[Path]:
    """Utility iterator for sensor CSV files."""
    yield from sorted(Path(sensors_root).rglob("*.csv"))
