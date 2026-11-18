$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$VenvMlflow = Join-Path $ProjectRoot ".venv\Scripts\mlflow.exe"

Set-Location $ProjectRoot

function Write-Step {
    param([int]$Number, [string]$Name)
    Write-Host ""
    Write-Host ("{0}/14 - {1}" -f $Number, $Name) -ForegroundColor Cyan
}

function Invoke-PythonStep {
    param([string]$Description, [string[]]$Arguments)

    Write-Host $Description
    & $VenvPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

function Wait-ForUrl {
    param([string]$Url, [int]$TimeoutSeconds = 60)

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            return Invoke-RestMethod -Uri $Url -Method Get -TimeoutSec 5
        } catch {
            Start-Sleep -Seconds 2
        }
    }
    throw "Timed out waiting for $Url"
}

Write-Step 1 "Prepare project folder"
Write-Host "Project root: $ProjectRoot"

Write-Step 2 "Check or create Python 3.11 virtual environment"
if (-not (Test-Path $VenvPython)) {
    python -m venv ".venv"
}
if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment Python was not created at $VenvPython"
}
& $VenvPython --version

Write-Step 3 "Upgrade pip"
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

Write-Step 4 "Install project dependencies"
& $VenvPython -m pip install -r "requirements.txt"
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }

Write-Step 5 "Generate Version 1 and Version 2 datasets"
Invoke-PythonStep "Generating synthetic datasets..." @("-m", "src.generate_data")

Write-Step 6 "Build SQLite feature store"
Invoke-PythonStep "Materializing engineered features..." @("-m", "src.feature_pipeline")

Write-Step 7 "Train and compare Decision Tree and Random Forest"
Invoke-PythonStep "Training models and registering with MLflow..." @("-m", "src.train")

Write-Step 8 "Run the complete reproducible pipeline"
Invoke-PythonStep "Verifying the one-command pipeline..." @("-m", "src.run_pipeline")

Write-Step 9 "Run pytest"
Invoke-PythonStep "Running automated tests..." @("-m", "pytest", "-q")

Write-Step 10 "Build the final PDF report"
Invoke-PythonStep "Building final project report..." @("build_report.py")

$ApiProcess = $null
$MlflowProcess = $null

try {
    Write-Step 11 "Start the FastAPI server"
    $ApiProcess = Start-Process -FilePath $VenvPython `
        -ArgumentList "-m", "uvicorn", "src.api:app", "--host", "127.0.0.1", "--port", "8000" `
        -WorkingDirectory $ProjectRoot -PassThru

    $health = Wait-ForUrl "http://127.0.0.1:8000/health"
    Write-Host ($health | ConvertTo-Json -Depth 5)

    Write-Step 12 "Test the /predict endpoint"
    $payload = @{
        relative_humidity_pct = 91
        room_temperature_c = 18
        wall_surface_temperature_c = 14
        door_openings_per_hour = 10
        ventilation_duration_min_per_hour = 8
        dehumidifier_operating_hours_per_day = 2
        room_occupancy = 4
    } | ConvertTo-Json

    $prediction = Invoke-RestMethod `
        -Uri "http://127.0.0.1:8000/predict" `
        -Method Post `
        -ContentType "application/json" `
        -Body $payload

    Write-Host ($prediction | ConvertTo-Json -Depth 5)

    $evidence = @{
        generated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        health = $health
        prediction = $prediction
    }
    New-Item -ItemType Directory -Force -Path "evidence" | Out-Null
    $evidence | ConvertTo-Json -Depth 8 |
        Set-Content "evidence\api_live_evidence.json" -Encoding UTF8

    Write-Step 13 "Start the MLflow dashboard"
    $MlflowProcess = Start-Process -FilePath $VenvMlflow `
        -ArgumentList "ui", "--backend-store-uri", "sqlite:///artifacts/mlflow.db", "--host", "127.0.0.1", "--port", "5000" `
        -WorkingDirectory $ProjectRoot -PassThru
    Wait-ForUrl "http://127.0.0.1:5000" | Out-Null

    Write-Host "FastAPI:  http://127.0.0.1:8000/docs"
    Write-Host "MLflow:   http://127.0.0.1:5000"

    Write-Step 14 "Display final output folders"
    Write-Host ""
    Write-Host "Final deliverables:"
    Get-Item "artifacts", "data\raw", "data\processed", "reports", "evidence" |
        ForEach-Object { Write-Host $_.FullName }

    $summary = @"
Project root: $ProjectRoot
FastAPI docs: http://127.0.0.1:8000/docs
MLflow dashboard: http://127.0.0.1:5000
Final report: $ProjectRoot\reports\Manuscript_Moisture_MLOps_Final_Report.pdf
Metrics: $ProjectRoot\artifacts\metrics.json
Model: $ProjectRoot\artifacts\best_model.joblib
Feature store: $ProjectRoot\data\processed\feature_store.sqlite
"@
    Set-Content "evidence\execution_status.txt" $summary -Encoding UTF8

    Start-Process "explorer.exe" -ArgumentList $ProjectRoot
    Start-Process "http://127.0.0.1:8000/docs"
    Start-Process "http://127.0.0.1:5000"
} catch {
    Write-Error $_
    throw
}
