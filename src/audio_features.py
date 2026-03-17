from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from io_utils import parse_audio_path

# Domain bands from assignment guidance.
DEFAULT_BANDS_HZ: dict[str, tuple[float, float]] = {
    "band_low_30_90": (30.0, 90.0),
    "band_mid_180_280": (180.0, 280.0),
    "band_high_450_530": (450.0, 530.0),
}


def load_audio(file_path: str | Path, target_sr: int = 16000, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load audio, optionally convert to mono and resample to target sample rate."""
    y, sr = sf.read(str(file_path), always_2d=True)
    y = y.astype(np.float32)

    if mono:
        y = y.mean(axis=1)
    else:
        y = y[:, 0]

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
        sr = target_sr

    return y.astype(np.float32), int(sr)


def bandpass_filter_audio(
    y: np.ndarray,
    sr: int,
    low_hz: float = 100.0,
    high_hz: float = 2000.0,
    order: int = 6,
) -> np.ndarray:
    """Apply Butterworth bandpass filtering (default 100-2000 Hz)."""
    nyquist = 0.5 * sr
    low = max(1.0, low_hz)
    high = min(high_hz, nyquist - 1.0)

    if low >= high:
        raise ValueError(f"Invalid bandpass bounds: low={low}, high={high}, sr={sr}")

    sos = butter(order, [low, high], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def frame_audio(y: np.ndarray, sr: int, frame_length_s: float = 1.0, overlap: float = 0.5) -> tuple[np.ndarray, int, int]:
    """Frame waveform into Hann-window-compatible slices with configurable overlap."""
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")

    frame_length = int(sr * frame_length_s)
    if frame_length <= 0:
        raise ValueError("frame_length_s produced non-positive frame length")

    hop_length = max(1, int(frame_length * (1.0 - overlap)))

    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)), mode="constant")

    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    return frames.astype(np.float32), frame_length, hop_length


def _safe_feature_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": np.nan, "std": np.nan, "p10": np.nan, "p50": np.nan, "p90": np.nan}

    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "p10": float(np.nanpercentile(arr, 10)),
        "p50": float(np.nanpercentile(arr, 50)),
        "p90": float(np.nanpercentile(arr, 90)),
    }


def aggregate_feature_matrix(feature_name: str, matrix: np.ndarray) -> dict[str, float]:
    """Aggregate per-frame feature matrices with mean/std/p10/p50/p90."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]

    result: dict[str, float] = {}
    n_rows = arr.shape[0]

    for i in range(n_rows):
        base_name = f"{feature_name}_{i + 1:02d}" if n_rows > 1 else feature_name
        stats = _safe_feature_stats(arr[i])
        for stat_name, stat_val in stats.items():
            result[f"{base_name}_{stat_name}"] = stat_val

    return result


def extract_core_features(
    y: np.ndarray,
    sr: int,
    frame_length_s: float = 1.0,
    overlap: float = 0.5,
    n_mfcc: int = 13,
) -> dict[str, np.ndarray]:
    """Extract frame-level MFCC and spectral features using 1 s Hann windows and 50% overlap by default."""
    _, n_fft, hop_length = frame_audio(y, sr=sr, frame_length_s=frame_length_s, overlap=overlap)

    feature_kwargs = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": n_fft,
        "window": "hann",
        "center": False,
    }

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, **feature_kwargs)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, **feature_kwargs)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, **feature_kwargs)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, **feature_kwargs)
    flatness = librosa.feature.spectral_flatness(y=y, **feature_kwargs)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, **feature_kwargs)
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop_length, center=False)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length, center=False)

    return {
        "mfcc": mfcc,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
        "spectral_flatness": flatness,
        "chroma": chroma,
        "zero_crossing_rate": zcr,
        "rms": rms,
    }


def compute_band_power_features(
    y: np.ndarray,
    sr: int,
    frame_length_s: float = 1.0,
    overlap: float = 0.5,
    bands_hz: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Compute per-band power and variability across time frames for domain-relevant frequency bands."""
    bands_hz = bands_hz or DEFAULT_BANDS_HZ
    _, n_fft, hop_length = frame_audio(y, sr=sr, frame_length_s=frame_length_s, overlap=overlap)

    stft = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window="hann",
        center=False,
    )
    power = np.abs(stft) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    features: dict[str, float] = {}

    for band_name, (f_low, f_high) in bands_hz.items():
        mask = (freqs >= f_low) & (freqs < f_high)
        if not np.any(mask):
            features[f"{band_name}_power_mean"] = np.nan
            features[f"{band_name}_power_std"] = np.nan
            features[f"{band_name}_power_p10"] = np.nan
            features[f"{band_name}_power_p50"] = np.nan
            features[f"{band_name}_power_p90"] = np.nan
            features[f"{band_name}_variability_cv"] = np.nan
            continue

        # Average power in selected acoustic band at each frame.
        band_power_t = power[mask, :].mean(axis=0)
        stats = _safe_feature_stats(band_power_t)

        features[f"{band_name}_power_mean"] = stats["mean"]
        features[f"{band_name}_power_std"] = stats["std"]
        features[f"{band_name}_power_p10"] = stats["p10"]
        features[f"{band_name}_power_p50"] = stats["p50"]
        features[f"{band_name}_power_p90"] = stats["p90"]

        mean_val = stats["mean"]
        std_val = stats["std"]
        cv = std_val / mean_val if np.isfinite(mean_val) and mean_val > 0 else np.nan
        features[f"{band_name}_variability_cv"] = float(cv) if np.isfinite(cv) else np.nan

    return features


def extract_audio_feature_row(
    file_path: str | Path,
    target_sr: int = 16000,
    low_hz: float = 100.0,
    high_hz: float = 2000.0,
    frame_length_s: float = 1.0,
    overlap: float = 0.5,
    n_mfcc: int = 13,
) -> dict[str, Any]:
    """End-to-end per-file feature extraction with metadata and aggregated feature statistics."""
    path = Path(file_path)
    parsed = parse_audio_path(path)

    y, sr = load_audio(path, target_sr=target_sr, mono=True)
    y = bandpass_filter_audio(y, sr=sr, low_hz=low_hz, high_hz=high_hz, order=6)

    core = extract_core_features(
        y,
        sr=sr,
        frame_length_s=frame_length_s,
        overlap=overlap,
        n_mfcc=n_mfcc,
    )

    row: dict[str, Any] = {
        "file_path": str(path),
        "file_name": path.name,
        "hive": parsed.hive,
        "timestamp_utc": parsed.timestamp_utc,
        "date": parsed.timestamp_utc.date(),
        "hour": int(parsed.timestamp_utc.hour),
        "month_dir": parsed.month_dir,
        "day_dir": parsed.day_dir,
        "samplerate": sr,
        "duration_s": float(len(y) / sr),
        "n_samples": int(len(y)),
    }

    for feature_name, values in core.items():
        row.update(aggregate_feature_matrix(feature_name, values))

    row.update(
        compute_band_power_features(
            y,
            sr=sr,
            frame_length_s=frame_length_s,
            overlap=overlap,
            bands_hz=DEFAULT_BANDS_HZ,
        )
    )

    return row


def process_audio_files(
    file_paths: list[str | Path],
    target_sr: int = 16000,
    low_hz: float = 100.0,
    high_hz: float = 2000.0,
    frame_length_s: float = 1.0,
    overlap: float = 0.5,
    n_mfcc: int = 13,
) -> pd.DataFrame:
    """Process multiple files and return a feature DataFrame."""
    rows: list[dict[str, Any]] = []

    for file_path in file_paths:
        rows.append(
            extract_audio_feature_row(
                file_path=file_path,
                target_sr=target_sr,
                low_hz=low_hz,
                high_hz=high_hz,
                frame_length_s=frame_length_s,
                overlap=overlap,
                n_mfcc=n_mfcc,
            )
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    return df.sort_values(["hive", "timestamp_utc", "file_name"]).reset_index(drop=True)
