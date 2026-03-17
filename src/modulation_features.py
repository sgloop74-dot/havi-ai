from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd

from audio_features import DEFAULT_BANDS_HZ, bandpass_filter_audio, load_audio
from io_utils import parse_audio_path


def compute_stft_magnitude(
    y: np.ndarray,
    sr: int,
    frame_length_s: float = 1.0,
    overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute magnitude spectrogram S(f, t) using Hann-window STFT.

    Notes on assumptions:
    - We use a fixed analysis frame (default 1 s) and overlap (default 50%)
      to be consistent with the core spectral feature pipeline.
    - STFT is computed with center=False to keep frame timing explicit.

    Returns:
    - magnitude: |STFT| with shape (n_freq_bins, n_time_frames)
    - freqs_hz: frequency value for each STFT row
    - hop_length: frame hop in samples
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0,1)")

    n_fft = int(sr * frame_length_s)
    if n_fft <= 0:
        raise ValueError("frame_length_s produced non-positive n_fft")

    hop_length = max(1, int(n_fft * (1.0 - overlap)))

    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size), mode="constant")

    stft = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window="hann",
        center=False,
    )
    magnitude = np.abs(stft).astype(np.float32)
    freqs_hz = librosa.fft_frequencies(sr=sr, n_fft=n_fft).astype(np.float32)
    return magnitude, freqs_hz, hop_length


def compute_modulation_spectrum(
    magnitude_spectrogram: np.ndarray,
    sr: int,
    hop_length: int,
    mod_low_hz: float = 1.0,
    mod_high_hz: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute modulation spectrum M(f, w_m) via FFT along the time axis.

    Mathematical idea:
    - Given STFT magnitude S(f, t), for each acoustic frequency row f,
      compute a temporal FFT across t: M(f, w_m) = |FFT_t(S(f, t))|.
    - This captures how quickly energy in each acoustic band fluctuates over time.

    Assumptions:
    - Temporal sample rate of S(f, t) is frame_rate = sr / hop_length.
    - We retain modulation frequencies in [mod_low_hz, mod_high_hz], default 1-30 Hz.
    """
    if hop_length <= 0:
        raise ValueError("hop_length must be > 0")

    n_time = magnitude_spectrogram.shape[1]
    if n_time < 2:
        # Not enough frames to estimate temporal modulation.
        return np.zeros((magnitude_spectrogram.shape[0], 0), dtype=np.float32), np.array([], dtype=np.float32)

    mod_complex = np.fft.rfft(magnitude_spectrogram, axis=1)
    mod_mag = np.abs(mod_complex).astype(np.float32)

    frame_rate_hz = sr / float(hop_length)
    mod_freqs_hz = np.fft.rfftfreq(n_time, d=1.0 / frame_rate_hz).astype(np.float32)

    mask = (mod_freqs_hz >= mod_low_hz) & (mod_freqs_hz <= mod_high_hz)
    return mod_mag[:, mask], mod_freqs_hz[mask]


def _spectral_entropy(x: np.ndarray, eps: float = 1e-12) -> float:
    """Compute normalized Shannon entropy for non-negative vector x."""
    arr = np.asarray(x, dtype=np.float64)
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0:
        return float("nan")

    p = arr / (total + eps)
    p = np.clip(p, eps, 1.0)
    h = -np.sum(p * np.log(p))
    h_max = np.log(len(p)) if len(p) > 1 else 1.0
    return float(h / h_max)


def summarize_modulation_by_bands(
    mod_mag: np.ndarray,
    acoustic_freqs_hz: np.ndarray,
    mod_freqs_hz: np.ndarray,
    bands_hz: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Summarize modulation behavior by acoustic bands.

    For each acoustic band B, we first average M(f, w_m) over f in B,
    producing a 1D modulation profile P_B(w_m). We then extract:
    - modulation energy: mean and total of P_B
    - dominant modulation frequency: argmax_wm P_B(w_m)
    - modulation entropy: normalized spectral entropy of P_B
    """
    bands_hz = bands_hz or DEFAULT_BANDS_HZ
    features: dict[str, float] = {}

    for band_name, (f_low, f_high) in bands_hz.items():
        band_mask = (acoustic_freqs_hz >= f_low) & (acoustic_freqs_hz < f_high)
        if not np.any(band_mask) or mod_mag.size == 0 or mod_freqs_hz.size == 0:
            features[f"{band_name}_mod_energy_mean"] = np.nan
            features[f"{band_name}_mod_energy_total"] = np.nan
            features[f"{band_name}_mod_peak_hz"] = np.nan
            features[f"{band_name}_mod_entropy"] = np.nan
            continue

        profile = mod_mag[band_mask, :].mean(axis=0)
        profile = np.asarray(profile, dtype=np.float64)

        features[f"{band_name}_mod_energy_mean"] = float(np.nanmean(profile))
        features[f"{band_name}_mod_energy_total"] = float(np.nansum(profile))

        peak_idx = int(np.nanargmax(profile)) if profile.size else 0
        features[f"{band_name}_mod_peak_hz"] = float(mod_freqs_hz[peak_idx]) if profile.size else np.nan
        features[f"{band_name}_mod_entropy"] = _spectral_entropy(profile)

    return features


def extract_modulation_feature_row(
    file_path: str | Path,
    target_sr: int = 16000,
    low_hz: float = 100.0,
    high_hz: float = 2000.0,
    frame_length_s: float = 1.0,
    overlap: float = 0.5,
    mod_low_hz: float = 1.0,
    mod_high_hz: float = 30.0,
) -> dict[str, Any]:
    """End-to-end modulation feature extraction for a single audio file."""
    path = Path(file_path)
    parsed = parse_audio_path(path)

    y, sr = load_audio(path, target_sr=target_sr, mono=True)
    y = bandpass_filter_audio(y, sr=sr, low_hz=low_hz, high_hz=high_hz, order=6)

    magnitude, acoustic_freqs_hz, hop_length = compute_stft_magnitude(
        y,
        sr,
        frame_length_s=frame_length_s,
        overlap=overlap,
    )

    mod_mag, mod_freqs_hz = compute_modulation_spectrum(
        magnitude,
        sr=sr,
        hop_length=hop_length,
        mod_low_hz=mod_low_hz,
        mod_high_hz=mod_high_hz,
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
        "mod_freq_min_hz": float(mod_low_hz),
        "mod_freq_max_hz": float(mod_high_hz),
        "n_mod_bins": int(mod_freqs_hz.size),
    }

    # Global modulation summaries across all acoustic bins.
    if mod_mag.size and mod_freqs_hz.size:
        global_profile = mod_mag.mean(axis=0)
        row["mod_global_energy_mean"] = float(np.nanmean(global_profile))
        row["mod_global_energy_total"] = float(np.nansum(global_profile))
        peak_idx = int(np.nanargmax(global_profile))
        row["mod_global_peak_hz"] = float(mod_freqs_hz[peak_idx])
        row["mod_global_entropy"] = _spectral_entropy(global_profile)
    else:
        row["mod_global_energy_mean"] = np.nan
        row["mod_global_energy_total"] = np.nan
        row["mod_global_peak_hz"] = np.nan
        row["mod_global_entropy"] = np.nan

    row.update(
        summarize_modulation_by_bands(
            mod_mag=mod_mag,
            acoustic_freqs_hz=acoustic_freqs_hz,
            mod_freqs_hz=mod_freqs_hz,
            bands_hz=DEFAULT_BANDS_HZ,
        )
    )
    return row


def process_modulation_files(
    file_paths: list[str | Path],
    target_sr: int = 16000,
    low_hz: float = 100.0,
    high_hz: float = 2000.0,
    frame_length_s: float = 1.0,
    overlap: float = 0.5,
    mod_low_hz: float = 1.0,
    mod_high_hz: float = 30.0,
) -> pd.DataFrame:
    """Process multiple files and return a modulation feature DataFrame."""
    rows: list[dict[str, Any]] = []
    for file_path in file_paths:
        rows.append(
            extract_modulation_feature_row(
                file_path=file_path,
                target_sr=target_sr,
                low_hz=low_hz,
                high_hz=high_hz,
                frame_length_s=frame_length_s,
                overlap=overlap,
                mod_low_hz=mod_low_hz,
                mod_high_hz=mod_high_hz,
            )
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    return df.sort_values(["hive", "timestamp_utc", "file_name"]).reset_index(drop=True)
