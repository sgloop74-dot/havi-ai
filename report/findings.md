# HiveNavigator Findings

## Data
The analysis uses hive audio recordings plus available sensor and accelerometer logs. Audio-derived and multimodal features were merged to hourly granularity in UTC and used for unsupervised modeling. The final clustering frame contains 1316 hourly rows.

Primary data artifacts:
- [outputs/features/audio_hourly_features.parquet](outputs/features/audio_hourly_features.parquet)
- [outputs/features/modulation_hourly_features.parquet](outputs/features/modulation_hourly_features.parquet)
- [outputs/features/multimodal_hourly_features.parquet](outputs/features/multimodal_hourly_features.parquet)

## Features
The feature set includes spectral descriptors, MFCC statistics, custom acoustic band summaries, and sensor aggregates.

Key biologically relevant groups:
- Acoustic texture around the 200-600 Hz target region, represented here by 180-280 Hz and 450-530 Hz band-power summaries.
- Cluster-transition behavior over time (state switching frequency).
- Modulation features from the modulation pipeline, included where values were available.
- Sensor and accelerometer context features for multimodal interpretation.

## Methods
Unsupervised methods were run on standardized features:
- KMeans for baseline partitioning.
- Gaussian Mixture for soft clustering.
- DBSCAN for anomaly/state discovery.

Model evaluation used silhouette and Davies-Bouldin metrics from [outputs/models/cluster_metrics.csv](outputs/models/cluster_metrics.csv). PCA-based cluster/hive projections were reviewed using:
- [outputs/figures/pca_by_hive.png](outputs/figures/pca_by_hive.png)
- [outputs/figures/pca_by_kmeans_cluster.png](outputs/figures/pca_by_kmeans_cluster.png)
- [outputs/figures/pca_by_gmm_cluster.png](outputs/figures/pca_by_gmm_cluster.png)
- [outputs/figures/pca_by_dbscan_cluster.png](outputs/figures/pca_by_dbscan_cluster.png)

## Cluster Findings
Latest metrics:
- KMeans: silhouette 0.3225, Davies-Bouldin 1.1329.
- Gaussian Mixture: silhouette 0.3193, Davies-Bouldin 1.3609.
- DBSCAN: 15 clusters, 927 noise points of 1316 (~70%), silhouette 0.2698, Davies-Bouldin 0.8506.

Interpretation:
- KMeans and Gaussian Mixture provide stable coarse state segmentation.
- DBSCAN isolates denser local regimes and anomalies, but with a high noise fraction that limits direct state labeling.

Supporting tables:
- [outputs/models/cluster_assignments.parquet](outputs/models/cluster_assignments.parquet)
- [outputs/models/cluster_distribution_kmeans.csv](outputs/models/cluster_distribution_kmeans.csv)
- [outputs/models/cluster_distribution_gmm.csv](outputs/models/cluster_distribution_gmm.csv)
- [outputs/models/cluster_distribution_dbscan.csv](outputs/models/cluster_distribution_dbscan.csv)

## Estimated Queenless Windows for Hive 3 and 4
Queenless-window inference was generated from a combined evidence score using:
- matched-hour cluster transition excess versus control hives,
- 200-600 Hz proxy acoustic deviation,
- modulation evidence contribution (neutral fallback where modulation values were unavailable in this run).

Ranked candidates and hypothesis artifacts:
- [outputs/models/queenless_window_ranked.csv](outputs/models/queenless_window_ranked.csv)
- [outputs/models/queenless_window_candidates.csv](outputs/models/queenless_window_candidates.csv)
- [outputs/models/queenless_top20.csv](outputs/models/queenless_top20.csv)
- [outputs/queenless_window_hypothesis.md](outputs/queenless_window_hypothesis.md)

Top-ranked windows currently include repeated high-confidence periods in both target hives, with the highest ranks concentrated in hive_03 and hive_04 morning and transition hours.

## Limitations
- No ground-truth intervention labels were available, so conclusions are hypothesis-driven.
- The current modulation table in this run contains missing numerical modulation summaries, reducing modulation evidence strength.
- DBSCAN behavior is parameter-sensitive and yields high noise in this feature geometry.
- Sensor coverage is partial across hives and time.

## Next Steps
1. Repair and re-run modulation feature extraction so modulation evidence is fully populated, then re-rank queenless windows.
2. Validate top-ranked candidate windows against beekeeper logs or manual inspection notes.
3. Add temporal persistence constraints or change-point logic to reduce isolated spikes.
4. Compare the current ranking with a second unsupervised anomaly model for robustness.
