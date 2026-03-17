# HiveNavigator Take-Home: Execution Plan + Copilot Prompt Pack

## 1) What this assignment is asking

You need to do three things end-to-end:

1. Build a robust feature extraction pipeline from hive audio (and use vibration/sensor data as optional overlays).
2. Use unsupervised learning to discover colony states and infer when Hive 3 and Hive 4 were likely queenless.
3. Deliver results as:
   - a reproducible notebook or scripts,
   - an interactive time-series dashboard,
   - a short report with conclusions.

Core scientific target:
- Detect shifts in spectral behavior, especially in roughly 200-600 Hz, plus modulation behavior in about 1-30 Hz modulation rates.
- Compare Hive 3 and Hive 4 against control hives.
- Identify likely intervention windows (queen removal/reintroduction period).

## 2) What success looks like

By the end, your submission should include:

1. Reproducible pipeline from raw FLAC + CSV to hourly features.
2. Unsupervised clusters with interpretable feature signatures.
3. Time-resolved visualizations showing day/night patterns and hive differences.
4. A defensible estimate of queenless windows for Hive 3 and Hive 4.
5. Concise report (max about 2 pages worth of content).

## 3) Step-by-step execution plan

## Phase A: Project setup and data audit

1. Create project structure:
   - notebooks/
   - src/
   - outputs/features/
   - outputs/figures/
   - outputs/models/
   - app/ (for dashboard)
2. Create requirements file and install dependencies.
3. Build a data inventory script:
   - count audio files per hive/day/hour,
   - verify sample rates/channels,
   - log missing/corrupt files.
4. Normalize timestamps (UTC) and generate a master index table.

Expected output:
- outputs/data_inventory.csv
- outputs/data_quality_report.md

## Phase B: Audio preprocessing + feature extraction

For each audio file:

1. Load FLAC at 16 kHz mono.
2. Apply bandpass filter (start with 100-2000 Hz, keep configurable).
3. Frame signal (1 s windows, 50 percent overlap; later test 2 s).
4. Extract per-frame features, then aggregate to per-file summary (mean/std/percentiles):
   - MFCC (13 or 20)
   - spectral centroid
   - spectral bandwidth
   - spectral rolloff
   - spectral flatness
   - chroma
   - zero-crossing rate
   - RMS
   - optional: spectral contrast, flux, band-energy ratios
5. Track domain bands from figure guidance:
   - low band around 30-90 Hz
   - mid buzz around 180-280 Hz
   - high buzz around 450-530 Hz
   - compute each band power, variance, and temporal stability

Expected output:
- outputs/features/audio_hourly_features.parquet
- outputs/features/audio_feature_dictionary.md

## Phase C: Modulation spectrogram features

1. Compute STFT magnitude S(t,f).
2. FFT along time axis for each frequency row to get modulation representation M(f, wm).
3. Focus modulation rates near 1-30 Hz.
4. Summarize into interpretable features:
   - modulation energy in selected acoustic bands,
   - peak modulation frequency per band,
   - modulation entropy/flatness.
5. Optionally store compact 2D maps for plotting.

Expected output:
- outputs/features/modulation_hourly_features.parquet
- outputs/figures/modulation_examples.png

## Phase D: Sensor and accelerometer integration

1. Parse sensor and accel CSV files.
2. Resample to hourly features (mean/std/min/max + trend).
3. Join with audio features on hive + hour.
4. Flag missing values and add data-quality columns.

Expected output:
- outputs/features/multimodal_hourly_features.parquet

## Phase E: Unsupervised modeling

1. Build feature matrix options:
   - audio only,
   - audio + modulation,
   - multimodal (when available).
2. Standardize features.
3. Dimensionality reduction for visualization:
   - PCA first,
   - optional UMAP/t-SNE for cluster view.
4. Cluster methods:
   - KMeans (baseline),
   - Gaussian Mixture (soft assignment),
   - HDBSCAN/DBSCAN (anomaly/state discovery).
5. Evaluate cluster quality:
   - silhouette score,
   - Davies-Bouldin,
   - stability across random seeds.
6. Interpret clusters:
   - top distinguishing features per cluster,
   - distribution by hive and date,
   - relation to day/night.

Expected output:
- outputs/models/cluster_assignments.parquet
- outputs/models/cluster_metrics.csv
- outputs/figures/cluster_projection.png

## Phase F: Queenless window inference

1. For Hive 3 and Hive 4, plot cluster timeline across days.
2. Compare against control hives at same hours (normalize by diurnal pattern).
3. Define candidate queenless window rules:
   - persistent shift in cluster membership,
   - sustained change in 200-600 Hz energy texture,
   - modulation pattern shift.
4. Produce confidence score per day/hour.
5. Select best window and justify with evidence plots.

Expected output:
- outputs/figures/hive3_hive4_state_timeline.png
- outputs/queenless_window_hypothesis.md

## Phase G: Interactive dashboard

Build a Streamlit (or Dash) app with:

1. Feature selector (multi-select).
2. Hive toggles.
3. Date-range zoom.
4. Cluster/state overlay.
5. Optional secondary panel for sensor/accel.
6. Export selected view to CSV/PNG.

Expected output:
- app/streamlit_app.py
- README run instructions.

## Phase H: Report writing

Report structure:

1. Data and preprocessing summary.
2. Features and why they are biologically relevant.
3. Unsupervised methods and selected model.
4. Cluster characteristics.
5. Estimated queenless windows for Hive 3 and Hive 4.
6. Limitations (cold weather, sparse sensors) and next steps.

Expected output:
- report/findings.md or report/findings.pdf

## 4) Suggested file layout

Use this target layout:

```text
hive ai/
  data/
  notebooks/
    01_data_audit.ipynb
    02_feature_extraction.ipynb
    03_unsupervised_analysis.ipynb
    04_report.ipynb
  src/
    io_utils.py
    audio_features.py
    modulation_features.py
    sensor_features.py
    clustering.py
    plotting.py
  app/
    streamlit_app.py
  outputs/
    features/
    figures/
    models/
  report/
    findings.md
  requirements.txt
  README.md
```

## 5) Risks and mitigation

1. Audio volume and runtime are large.
   - Mitigation: cache per-file features and use parquet.
2. Day/night confounding can mimic state change.
   - Mitigation: compare hives at matched hour-of-day and include diurnal baseline.
3. Cluster instability.
   - Mitigation: try multiple algorithms, check stability, prefer interpretable clusters.
4. Sensor coverage is incomplete.
   - Mitigation: keep audio-only baseline and mark multimodal analysis as partial.

## 6) Prompt pack for GitHub Copilot (use in order)

Copy and send prompts one by one.

Prompt 1: Project scaffold

Create a clean project scaffold in this workspace for the HiveNavigator assignment.
Make folders: notebooks, src, outputs/features, outputs/figures, outputs/models, app, report.
Create requirements.txt with needed libraries: numpy, pandas, scipy, librosa, soundfile, scikit-learn, plotly, streamlit, pyarrow, tqdm, matplotlib, seaborn.
Create a README.md with setup and run commands.
Do not overwrite existing data folder.

Prompt 2: Data inventory and QA

Implement src/io_utils.py and a notebook notebooks/01_data_audit.ipynb.
The notebook should scan data/audio recursively, parse hive/date/hour from file paths, validate audio metadata (sample rate, channels, duration), and save outputs/data_inventory.csv and outputs/data_quality_report.md.
Include clear summary tables and plots by hive and day.

Prompt 3: Core audio feature extraction

Implement src/audio_features.py with functions to:
1) load and bandpass filter audio (default 100-2000 Hz),
2) frame at 1 s with 50 percent overlap,
3) extract MFCC, centroid, bandwidth, rolloff, flatness, chroma, zero-crossing rate, RMS,
4) compute per-file aggregated stats (mean, std, p10, p50, p90),
5) compute custom band features for 30-90, 180-280, 450-530 Hz (band power and variability).
Also create a runner script to process all files and save outputs/features/audio_hourly_features.parquet.

Prompt 4: Modulation spectrogram features

Implement src/modulation_features.py.
For each audio file:
- compute STFT magnitude,
- compute FFT along time axis to get modulation representation,
- keep modulation frequencies 1-30 Hz,
- summarize per acoustic band using modulation energy, dominant modulation frequency, and entropy.
Save outputs/features/modulation_hourly_features.parquet.
Add docstrings explaining the math and assumptions.

Prompt 5: Sensor feature integration

Implement src/sensor_features.py to load sensor and accel CSVs for Hive 3 and Hive 4.
Parse timestamps as UTC, resample to hourly features, and output summary stats.
Create a merge pipeline that joins sensor/accel features with audio features by hive and hour.
Save outputs/features/multimodal_hourly_features.parquet.

Prompt 6: Unsupervised clustering pipeline

Implement src/clustering.py and notebook notebooks/03_unsupervised_analysis.ipynb.
Build clustering experiments with standardized features using KMeans, GaussianMixture, and DBSCAN or HDBSCAN.
Compute silhouette and Davies-Bouldin scores.
Create PCA plots colored by cluster and hive.
Save cluster assignments to outputs/models/cluster_assignments.parquet and metrics to outputs/models/cluster_metrics.csv.

Prompt 7: Queenless window detection logic

In notebooks/03_unsupervised_analysis.ipynb, add a section that infers likely queenless windows for Hive 3 and Hive 4.
Use evidence from cluster transitions, 200-600 Hz related features, and modulation features.
Compare against control hives matched by hour of day.
Output a ranked table of candidate windows with confidence scores and explanatory notes.
Save outputs/queenless_window_hypothesis.md.

Prompt 8: Interactive dashboard

Create app/streamlit_app.py with an interactive dashboard:
- feature selector,
- hive toggles,
- date range filter,
- time series chart,
- optional secondary panel for sensor/accel,
- cluster overlay.
Load from parquet outputs and ensure app runs with streamlit run app/streamlit_app.py.

Prompt 9: Report generation

Create report/findings.md as a concise technical report with sections:
Data, Features, Methods, Cluster Findings, Estimated Queenless Windows for Hive 3 and 4, Limitations, Next Steps.
Reference generated figures and tables from outputs.
Keep it concise but defensible.

Prompt 10: Reproducibility and polish

Add a single orchestration script src/run_all.py that executes the pipeline in order (inventory, feature extraction, integration, clustering).
Add argument flags for partial runs and caching.
Update README.md with end-to-end commands and troubleshooting notes.

## 7) How to use this practically

1. Run Prompt 1.
2. Execute created setup commands.
3. Run Prompts 2 through 10 sequentially.
4. After each prompt, run and verify outputs before moving on.
5. If something fails, ask Copilot to fix only that module, then continue.

## 8) Optional high-impact enhancements

1. Add diurnal normalization features (deviation from hive-hour baseline).
2. Add anomaly scores (Isolation Forest) as a secondary unsupervised signal.
3. Add per-hive change-point detection on key spectral and modulation features.
4. Add confidence intervals using bootstrap over hourly aggregates.

This plan is intentionally scoped to complete the assignment in a structured, submission-ready way while preserving interpretability.