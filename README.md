# Manuscript Moisture-Risk MLOps

An end-to-end, reproducible MLOps project that predicts whether a museum manuscript storage room is likely to face moisture-related damage risk in the **next 48 hours**.

> **Synthetic-data notice:** Both datasets are generated for coursework and demonstration. They are not measurements from a real museum and must not be used as a real conservation alarm without validation on museum sensor data and review by conservation specialists.

## 1. What this project delivers

- Two versioned synthetic datasets with 300 observations each.
- Version 2 simulates monsoon conditions: higher relative humidity and lower ventilation duration.
- A preprocessing and feature-engineering pipeline.
- A reusable SQLite feature store for persisted engineered features.
- Decision Tree and Random Forest comparison, with recall prioritized.
- MLflow tracking and model-registration code for `ManuscriptMoistureRiskModel`.
- FastAPI `/predict` and `/health` endpoints with validated inputs.
- Pytest checks and a GitHub Actions CI workflow.
- A final project report in `reports/` (the PDF is supplied alongside this project ZIP).

## 2. Project structure

```text
manuscript_moisture_mlops/
├── .github/workflows/ci.yml
├── artifacts/
│   ├── best_model.joblib
│   ├── feature_importance.csv
│   ├── metrics.json
│   └── model_metadata.json
├── data/
│   ├── raw/
│   │   ├── manuscript_rooms_v1.csv
│   │   ├── manuscript_rooms_v2_monsoon.csv
│   │   └── dataset_versions.json
│   └── processed/feature_store.sqlite
├── evidence/
│   ├── api_prediction_screenshot.png
│   └── fastapi_docs_screenshot.png
├── reports/
│   ├── risk_distribution.png
│   └── model_comparison.png
├── src/
│   ├── api.py
│   ├── config.py
│   ├── feature_pipeline.py
│   ├── feature_store.py
│   ├── features.py
│   ├── generate_data.py
│   ├── run_pipeline.py
│   └── train.py
├── tests/
│   ├── test_api.py
│   ├── test_features.py
│   └── test_validation.py
├── requirements.txt
└── README.md
```

## 3. Run on Windows (PowerShell)

Open PowerShell in the extracted project directory.

```powershell
# Create a virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install packages
python -m pip install --upgrade pip
pip install -r requirements.txt

# Run all pipeline steps: data -> features -> model training and MLflow registration
python -m src.run_pipeline

# Run tests
python -m pytest -q

# Start the REST API
uvicorn src.api:app --reload
```

Open the interactive API page at `http://127.0.0.1:8000/docs`.

## 4. Run the stages separately

```powershell
# 1) Generate 300 records per dataset version
python -m src.generate_data

# 2) Engineer and persist reusable features in SQLite
python -m src.feature_pipeline

# 3) Compare Decision Tree and Random Forest; save best model; log/register with MLflow
python -m src.train
```

The feature store table is `room_features`. Training reads the persisted table instead of recalculating features from the raw CSV files.

## 5. Features

Raw columns:

- `relative_humidity_pct`
- `room_temperature_c`
- `wall_surface_temperature_c`
- `door_openings_per_hour`
- `ventilation_duration_min_per_hour`
- `dehumidifier_operating_hours_per_day`
- `room_occupancy`

Engineered columns:

- `dew_point_c`: Magnus dew-point approximation from room temperature and relative humidity.
- `dew_point_margin_c`: wall surface temperature minus dew point. Smaller or negative margins mean condensation is more plausible.
- `condensation_risk_index`: normalized 0-1 score combining dew-point proximity and high relative humidity.
- `ventilation_effectiveness_ratio`: ventilation minutes divided by a demand-adjusted denominator that accounts for door traffic and occupancy.

The same `engineer_features()` function is used for offline feature-store materialization and live API inference to reduce training-serving skew.

## 6. Model strategy

- Train/validation split: Version 1 only, stratified 75/25.
- External stress evaluation: all Version 2 monsoon observations.
- Models: Decision Tree and Random Forest.
- Operating threshold: 0.35 (selected to favor sensitivity / recall for the early-warning scenario).
- Selection rule: highest Version 1 validation recall; F1 and ROC-AUC are tie-breakers.

Metrics are saved to `artifacts/metrics.json`. The best estimator is stored in `artifacts/best_model.joblib`, and `artifacts/model_metadata.json` records its threshold and feature schema.

## 7. MLflow registry

When dependencies are installed, `python -m src.train` uses a local SQLite-backed MLflow tracking store at `artifacts/mlflow.db`, creates the `manuscript-moisture-risk` experiment, logs evaluation metrics and model artifacts, and registers `ManuscriptMoistureRiskModel`.

Launch the UI with:

```powershell
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --host 127.0.0.1 --port 5000
```

Browse to `http://127.0.0.1:5000`.

To serve a registered version instead of the local Joblib copy, set the model URI and then start the API:

```powershell
$env:MLFLOW_MODEL_URI="models:/ManuscriptMoistureRiskModel/1"
uvicorn src.api:app --reload
```

The API uses `mlflow.sklearn.load_model()` for this option, preserving `predict_proba()` behavior.

## 8. API examples

### Health

```http
GET http://127.0.0.1:8000/health
```

### Prediction

```http
POST http://127.0.0.1:8000/predict
Content-Type: application/json
```

High-risk example:

```json
{
  "relative_humidity_pct": 91,
  "room_temperature_c": 18,
  "wall_surface_temperature_c": 14,
  "door_openings_per_hour": 10,
  "ventilation_duration_min_per_hour": 8,
  "dehumidifier_operating_hours_per_day": 2,
  "room_occupancy": 4
}
```

Expected response shape:

```json
{
  "prediction": "Moisture Risk",
  "moisture_risk_probability": 0.9637,
  "decision_threshold": 0.35,
  "forecast_horizon": "Next 48 hours",
  "engineered_features": {
    "condensation_risk_index": 0.9426,
    "ventilation_effectiveness_ratio": 0.0678,
    "dew_point_c": 16.508
  }
}
```

Probabilities shown above are local demonstration outputs for the generated model artifact; exact values can change if the random seed, dataset, model parameters or scikit-learn version changes.

Input constraints include humidity 0-100%, room temperature 5-40 C, wall surface temperature 0-45 C, door openings 0-60 per hour, ventilation 0-60 minutes per hour, dehumidifier 0-24 hours per day, and occupancy 0-30.

## 9. CI/CD

The GitHub Actions workflow at `.github/workflows/ci.yml` runs on pushes and pull requests to `main` or `master`. It installs dependencies, regenerates both datasets, materializes the feature store, trains/registers the model, runs tests, and uploads metrics and dataset metadata as a workflow artifact.

To activate hosted CI:

1. Create a new GitHub repository.
2. Extract this project and run `git init` in its root folder.
3. Commit and push to the repository's `main` branch.
4. Open the repository's **Actions** tab and inspect the `Manuscript Moisture MLOps CI` workflow.

A real GitHub Actions run cannot be created from this local project package; the first hosted run happens after you push it to your GitHub account.

## 10. Important limitations

1. Labels and sensor data are synthetic. The high evaluation scores show the code works against its synthetic labeling rule; they do not establish real-world predictive accuracy.
2. The training data is relatively small. A museum deployment needs months of sensor records, room-specific validation, alert calibration and human review.
3. The output is an early-warning risk estimate, not a guarantee of damage and not a replacement for conservation staff.
4. Missing sensors, sensor calibration drift, equipment failure and manuscript-specific material sensitivity are not modeled in this coursework prototype.
