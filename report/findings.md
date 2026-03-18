# HiveNavigator Findings (Draft)

## Objective
Identify unsupervised acoustic and multimodal patterns that may indicate queenless hive windows, focusing on `hive_03` and `hive_04`, without supervised labels.

## Data and Features
- Audio hourly features extracted from FLAC recordings (spectral, MFCC, band-power statistics).
- Modulation features extracted from time-varying spectra.
- Sensor and accelerometer hourly aggregates merged where available.
- Final analysis frame used for clustering: 1316 hourly rows.

## Clustering Results (Tuned Run)
From `outputs/models/cluster_metrics.csv`:

- `kmeans`: silhouette = 0.3225, Davies-Bouldin = 1.1329
- `gaussian_mixture`: silhouette = 0.3193, Davies-Bouldin = 1.3609
- `dbscan` (eps=3.0, min_samples=5):
  - clusters found = 15
  - noise points = 927 / 1316 (~70%)
  - silhouette (non-noise) = 0.2698
  - Davies-Bouldin (non-noise) = 0.8506

Interpretation:
- KMeans/GMM provide stable coarse partitioning.
- Tuned DBSCAN is useful for anomaly-oriented segmentation, but high noise fraction suggests sparse local density structure in this feature space.

## Queenless-Window Hypothesis
Heuristic:
1. Compute hourly cluster composition per hive.
2. Measure hour-to-hour change with total variation distance (TVD).
3. Compute rolling z-score per hive.
4. Mark candidate windows where z-score >= 2.0.

Outputs:
- Full candidate table: `outputs/models/queenless_window_candidates.csv`
- Ranked top windows: `outputs/models/queenless_top20.csv`

Top candidate windows by z-score include:
- `hive_04` at `2026-03-13 02:00:00+00:00` (z=4.6949)
- `hive_04` at `2026-03-13 03:00:00+00:00` (z=3.2468)
- `hive_01` at `2026-03-08 15:00:00+00:00` (z=2.6950)

Operational reading:
- These are high-priority anomaly windows for inspection, not confirmed labels.
- Candidate windows should be validated against beekeeper logs, weather/context events, or manual review.

## Limitations
- No ground-truth labels for queen loss, so findings are hypothesis-generating.
- DBSCAN sensitivity to `eps` and `min_samples` is high in high-dimensional feature spaces.
- Sensor data availability is uneven across hives.

## Recommended Next Actions
1. Build a weak-label validation set from expert review of top candidate windows.
2. Add temporal smoothing/hysteresis to reduce isolated one-hour spikes.
3. Compare multiple anomaly scores (e.g., isolation forest, one-class SVM, HMM state changes) against the TVD-zscore heuristic.
4. Integrate confidence bands and candidate ranking into the dashboard.
