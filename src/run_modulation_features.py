from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from io_utils import scan_audio_files
from modulation_features import extract_modulation_feature_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract modulation spectrogram features for all hive FLAC files.")
    parser.add_argument("--audio-root", type=Path, default=Path("data/audio"), help="Root directory containing hive FLAC files.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/features/modulation_hourly_features.parquet"),
        help="Destination parquet path for modulation features.",
    )
    parser.add_argument("--target-sr", type=int, default=16000, help="Target sample rate.")
    parser.add_argument("--low-hz", type=float, default=100.0, help="Bandpass low cutoff.")
    parser.add_argument("--high-hz", type=float, default=2000.0, help="Bandpass high cutoff.")
    parser.add_argument("--frame-length-s", type=float, default=1.0, help="STFT frame length in seconds.")
    parser.add_argument("--overlap", type=float, default=0.5, help="Frame overlap ratio [0,1).")
    parser.add_argument("--mod-low-hz", type=float, default=1.0, help="Minimum modulation frequency to retain.")
    parser.add_argument("--mod-high-hz", type=float, default=30.0, help="Maximum modulation frequency to retain.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for quick test runs.")
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue processing when a file fails; failed files are omitted from output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    file_paths = scan_audio_files(args.audio_root)
    if args.limit is not None:
        file_paths = file_paths[: args.limit]

    if not file_paths:
        raise SystemExit(f"No FLAC files found under: {args.audio_root}")

    rows: list[dict] = []
    failed: list[tuple[str, str]] = []

    for file_path in tqdm(file_paths, desc="Extracting modulation features", unit="file"):
        try:
            row = extract_modulation_feature_row(
                file_path=file_path,
                target_sr=args.target_sr,
                low_hz=args.low_hz,
                high_hz=args.high_hz,
                frame_length_s=args.frame_length_s,
                overlap=args.overlap,
                mod_low_hz=args.mod_low_hz,
                mod_high_hz=args.mod_high_hz,
            )
            rows.append(row)
        except Exception as exc:  # pragma: no cover
            failed.append((str(file_path), str(exc)))
            if not args.skip_errors:
                raise

    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df = df.sort_values(["hive", "timestamp_utc", "file_name"]).reset_index(drop=True)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_path, index=False)

    print(f"Saved modulation features: {args.output_path}")
    print(f"Rows written: {len(df)}")

    if failed:
        failed_path = args.output_path.with_suffix(".errors.csv")
        pd.DataFrame(failed, columns=["file_path", "error"]).to_csv(failed_path, index=False)
        print(f"Failed files: {len(failed)}")
        print(f"Error log: {failed_path}")


if __name__ == "__main__":
    main()
