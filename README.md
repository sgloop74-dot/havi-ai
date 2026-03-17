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

## Notes

- Keep all timestamps in UTC for merges across audio and sensor data.
- Save intermediate feature tables to Parquet under `outputs/features/` for caching and faster iteration.
- Do not overwrite raw files under `data/`.
