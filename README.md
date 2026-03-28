# HiveNavigator Acoustic and Sensor Analysis

Project scaffold for the HiveNavigator take-home exercise on unsupervised analysis of queenright vs. queenless colonies.

## Project Structure

- `data/` - provided dataset (audio and sensors). This folder is not modified.
- `notebooks/` - exploratory and reporting notebooks.
- `src/` - reusable pipeline modules.
- `outputs/features/` - extracted feature tables.
- `outputs/figures/` - generated plots.
- `outputs/models/` - clustering outputs and metrics.
- `app/` - dashboard application code.
- `report/` - final written findings.

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## Run Workflow (after implementation)

This scaffold supports the planned phases:

1. Data audit notebook: `notebooks/01_data_audit.ipynb`
2. Feature extraction modules in `src/`
3. Clustering analysis notebook: `notebooks/03_unsupervised_analysis.ipynb`
4. Dashboard app:

```powershell
streamlit run app/streamlit_app.py
```

## End-to-End Orchestration

Use the single orchestration script to run the pipeline in order:

- inventory
- audio feature extraction
- modulation feature extraction
- sensor integration
- clustering outputs

```powershell
.\.venv\Scripts\python.exe src\run_all.py
```

### Partial Runs

Run only selected stages:

```powershell
.\.venv\Scripts\python.exe src\run_all.py --stages audio,modulation
```

Quick test run on a subset of audio files:

```powershell
.\.venv\Scripts\python.exe src\run_all.py --stages audio,modulation --audio-limit 50 --skip-errors
```

Rerun stages even if cached outputs exist:

```powershell
.\.venv\Scripts\python.exe src\run_all.py --stages clustering --force --dbscan-eps 3.0 --dbscan-min-samples 5
```

### Common Commands

Run dashboard:

```powershell
streamlit run app/streamlit_app.py
```

Run only inventory refresh:

```powershell
.\.venv\Scripts\python.exe src\run_all.py --stages inventory --force
```

## Troubleshooting

1. `ModuleNotFoundError: tabulate`
	- Install optional markdown table dependency:

```powershell
.\.venv\Scripts\python.exe -m pip install tabulate
```

2. No files found under `data/audio`
	- Verify dataset path and run from project root (`hive ai/`).

3. Long runtime for extraction stages
	- Use `--audio-limit` for dry runs and remove it for full runs.
	- Keep caching enabled (default) to skip completed stages.

4. Streamlit port already in use
	- Use a different port:

```powershell
streamlit run app/streamlit_app.py --server.port 8502
```

5. Clustering output missing
	- Ensure at least `outputs/features/audio_hourly_features.parquet` exists, then run:

```powershell
.\.venv\Scripts\python.exe src\run_all.py --stages clustering --force
```

## Notes

- Keep all timestamps in UTC for merges across audio and sensor data.
- Save intermediate feature tables to Parquet under `outputs/features/` for caching and faster iteration.
- Do not overwrite raw files under `data/`.
