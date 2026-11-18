"""Build the final coursework PDF report from actual project outputs."""
from __future__ import annotations
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    Image, Preformatted, KeepTogether
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "Manuscript_Moisture_MLOps_Final_Report.pdf"
metrics = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
versions = json.loads((ROOT / "data" / "raw" / "dataset_versions.json").read_text())

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=24, leading=30, alignment=TA_CENTER, textColor=colors.HexColor("#16324F"), spaceAfter=14))
styles.add(ParagraphStyle(name="SubTitleX", parent=styles["Normal"], fontSize=11, leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#53657A"), spaceAfter=8))
styles.add(ParagraphStyle(name="H1X", parent=styles["Heading1"], fontSize=16, leading=20, textColor=colors.HexColor("#16324F"), spaceBefore=8, spaceAfter=8))
styles.add(ParagraphStyle(name="H2X", parent=styles["Heading2"], fontSize=11, leading=14, textColor=colors.HexColor("#215A7A"), spaceBefore=7, spaceAfter=5))
styles.add(ParagraphStyle(name="BodyX", parent=styles["BodyText"], fontSize=9, leading=13, spaceAfter=6))
styles.add(ParagraphStyle(name="SmallX", parent=styles["BodyText"], fontSize=7.5, leading=10, textColor=colors.HexColor("#53657A"), spaceAfter=3))
styles.add(ParagraphStyle(name="CodeX", parent=styles["Code"], fontName="Courier", fontSize=7.2, leading=9, leftIndent=7, rightIndent=7, borderPadding=6, backColor=colors.HexColor("#F1F4F8"), textColor=colors.HexColor("#243447")))
styles.add(ParagraphStyle(name="CalloutX", parent=styles["BodyText"], fontSize=8.5, leading=12, borderColor=colors.HexColor("#D5A642"), borderWidth=0.7, borderPadding=7, backColor=colors.HexColor("#FFF8E7"), spaceBefore=5, spaceAfter=8))


def P(text, style="BodyX"):
    return Paragraph(text, styles[style])


def make_table(rows, widths=None, header=True, font_size=7.5):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#D4DDE7")),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("FONTSIZE", (0,0), (-1,-1), font_size),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E8F0F6")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#16324F")),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(commands))
    return t


def footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(colors.HexColor("#D6DEE7"))
    canvas.line(1.7*cm, 1.35*cm, w-1.7*cm, 1.35*cm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#65758B"))
    canvas.drawString(1.7*cm, 0.95*cm, "Manuscript Moisture-Risk MLOps | Synthetic coursework prototype")
    canvas.drawRightString(w-1.7*cm, 0.95*cm, f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=1.7*cm, leftMargin=1.7*cm, topMargin=1.5*cm, bottomMargin=1.8*cm, title="Predicting Moisture Damage in Rare Manuscript Storage Rooms", author="MLOps Project Report")
story=[]

# PAGE 1: Cover + executive summary + architecture.
story += [Spacer(1, 1.1*cm), P("PREDICTING MOISTURE DAMAGE", "CoverTitle"), P("IN RARE MANUSCRIPT STORAGE ROOMS", "CoverTitle"),
          P("End-to-end Python MLOps project report", "SubTitleX"), Spacer(1, 0.35*cm),
          P("Project objective", "H1X"),
          P("Develop a machine-learning early-warning system that estimates whether a climate-controlled manuscript storage room is likely to experience moisture-related damage risk within the next 48 hours. The solution uses relative humidity, room and wall temperatures, door traffic, ventilation, dehumidifier operation and occupancy.", "BodyX"),
          P("Executive summary", "H1X")]
rf=metrics["candidate_metrics"]["Random Forest"]
v2=rf["external_stress_test_v2"]
summary_rows=[
 ["Project item","Local result"],
 ["Synthetic observations", "600 total (300 in Version 1 + 300 in Version 2)"],
 ["Selected classifier", metrics["selected_model"]],
 ["Version 1 validation recall", f"{rf['validation_v1']['recall']*100:.1f}%"],
 ["Version 2 monsoon stress recall", f"{v2['recall']*100:.1f}%"],
 ["Local automated tests", "9 passed"],
 ["API smoke test", "GET /health and POST /predict returned HTTP 200"],
]
story += [make_table(summary_rows,[5.1*cm,11.0*cm]), Spacer(1,0.25*cm),
          P("Important evidence note: the datasets and target labels are synthetic. The metrics demonstrate that the pipeline runs and learns the synthetic labeling rule; they are not proof of real-world museum accuracy. MLflow registry execution and GitHub-hosted Actions execution remain pending in this build environment, as detailed later.","CalloutX"),
          P("Workflow architecture", "H1X"),
          Image(str(ROOT/"reports"/"workflow_architecture.png"), width=17.0*cm, height=6.25*cm), PageBreak()]

# PAGE 2: Problem, data versions.
story += [P("1. Problem Definition", "H1X"),
          P("High humidity, low wall-surface temperature, repeated door openings and poor ventilation can cause water vapour to condense on cold surfaces. The project frames the task as binary classification: <b>Moisture Risk</b> versus <b>Storage Conditions Acceptable</b> for a 48-hour warning horizon.","BodyX"),
          P("1.1 Inputs and target", "H2X"),
          P("Each observation contains relative humidity (%), room temperature (C), wall surface temperature (C), door openings per hour, ventilation duration (minutes per hour), dehumidifier operating hours per day and room occupancy. The target column is <b>moisture_risk_status</b>.","BodyX"),
          P("1.2 Dataset version management", "H2X")]
vers_rows=[["Version","Rows","Mean RH","Mean ventilation","Synthetic risk rate","Purpose"]]
for name,item in versions["versions"].items():
    short="V1 - Baseline" if "Version 1" in name else "V2 - Monsoon"
    vers_rows.append([short,str(item["rows"]),f"{item['mean_relative_humidity_pct']:.2f}%",f"{item['mean_ventilation_min_per_hour']:.2f} min/h",f"{item['risk_rate']*100:.1f}%", "Controlled baseline" if short.startswith("V1") else "Higher humidity; reduced ventilation"])
story += [make_table(vers_rows,[2.2*cm,1.2*cm,2.0*cm,2.4*cm,2.0*cm,6.3*cm],font_size=7),
          Spacer(1,0.2*cm),
          P("Version 1 is the baseline set. Version 2 applies monsoon-like distribution shifts: higher humidity (about 75.87% mean) and lower ventilation duration (about 20.13 minutes per hour) than Version 1 (about 63.79% RH and 33.18 ventilation minutes per hour). Both datasets are seeded and reproducible.","BodyX"),
          Image(str(ROOT/"reports"/"risk_distribution.png"),width=16.6*cm,height=8.6*cm),
          P("Data files: data/raw/manuscript_rooms_v1.csv, data/raw/manuscript_rooms_v2_monsoon.csv and data/raw/dataset_versions.json. Each CSV contains 300 records, exceeding the minimum 250-record requirement per version.","SmallX"), PageBreak()]

# PAGE 3: pipeline and feature engineering.
story += [P("2. Data Pipeline and Feature Store", "H1X"),
          P("The preprocessing pipeline loads both versioned CSV files, applies the same feature-engineering functions, and upserts the resulting rows into a SQLite table named <b>room_features</b>. Training then reads the materialized feature table, which makes the engineered values reusable rather than recomputing them for each experiment.","BodyX"),
          P("2.1 Engineered features", "H2X")]
feature_rows=[
 ["Feature","Definition / intuition"],
 ["dew_point_c","Magnus approximation from room temperature and relative humidity."],
 ["dew_point_margin_c","Wall surface temperature - dew point. A small or negative margin indicates greater condensation plausibility."],
 ["condensation_risk_index","0-1 score combining dew-point proximity (70%) and high relative humidity (30%)."],
 ["ventilation_effectiveness_ratio","Ventilation minutes divided by (60 + 5 x door openings + 2 x occupancy), clipped to 0-1."],
]
story += [make_table(feature_rows,[4.1*cm,12.0*cm]), P("2.2 Formula summary", "H2X"),
          Preformatted("gamma = ln(RH / 100) + (17.625 * T) / (243.04 + T)\ndew_point_c = (243.04 * gamma) / (17.625 - gamma)\n\nmargin_pressure = clip((3 - dew_point_margin_c) / 6, 0, 1)\nhumidity_pressure = clip((RH - 55) / 35, 0, 1)\ncondensation_risk_index = 0.70 * margin_pressure + 0.30 * humidity_pressure\n\nventilation_effectiveness_ratio = ventilation_minutes /\n    (60 + 5 * door_openings_per_hour + 2 * room_occupancy)",styles["CodeX"]),
          P("The shared function engineer_features() is used in both offline preprocessing and API inference. This is a simple way to reduce training-serving feature skew. The SQLite store provides an offline materialization for the coursework system; a production deployment could replace it with Feast or a managed online/offline feature store.","BodyX"),
          P("2.3 Code locations", "H2X"),
          P("src/features.py contains feature formulas; src/feature_store.py materializes and retrieves features from SQLite; src/feature_pipeline.py runs the store build; src/config.py defines the reusable feature schema.","BodyX"),
          Image(str(ROOT/"reports"/"feature_importance.png"),width=16.4*cm,height=8.6*cm), PageBreak()]

# PAGE 4: model evaluation.
dt=metrics["candidate_metrics"]["Decision Tree"]["validation_v1"]
rfv=rf["validation_v1"]
story += [P("3. Model Training and Evaluation", "H1X"),
          P("Two classifiers were trained: Decision Tree and Random Forest. Both use a scikit-learn Pipeline with median imputation and the same 11 model features. The split is stratified: 75% of Version 1 is used for fitting and 25% for validation. Version 2 is kept separate as an external monsoon stress evaluation.","BodyX"),
          P("3.1 Classifier comparison - Version 1 validation", "H2X"),
          make_table([["Model","Accuracy","Precision","Recall","F1","ROC-AUC"],
                      ["Decision Tree",f"{dt['accuracy']:.3f}",f"{dt['precision']:.3f}",f"{dt['recall']:.3f}",f"{dt['f1']:.3f}",f"{dt['roc_auc']:.3f}"],
                      ["Random Forest",f"{rfv['accuracy']:.3f}",f"{rfv['precision']:.3f}",f"{rfv['recall']:.3f}",f"{rfv['f1']:.3f}",f"{rfv['roc_auc']:.3f}"]],
                     [3.2*cm,2.2*cm,2.2*cm,2.0*cm,2.0*cm,2.0*cm]),
          Image(str(ROOT/"reports"/"model_comparison.png"),width=16.4*cm,height=8.5*cm),
          P("3.2 Selection and recall priority", "H2X"),
          P("The selected operating threshold is 0.35. The decision rule prefers the highest Version 1 validation recall, then F1 and ROC-AUC as tie-breakers. Random Forest was selected because its validation recall is 0.90 (versus 0.50 for Decision Tree). Recall is emphasized because missing a true moisture-risk condition can be costly for fragile manuscripts.","BodyX"),
          P("3.3 Version 2 monsoon stress evaluation", "H2X"),
          make_table([["Metric","Random Forest on Version 2"],
                      ["Accuracy",f"{v2['accuracy']:.4f}"],["Precision",f"{v2['precision']:.4f}"],
                      ["Recall",f"{v2['recall']:.4f}"],["F1-score",f"{v2['f1']:.4f}"],["ROC-AUC",f"{v2['roc_auc']:.4f}"],
                      ["Confusion matrix",str(v2["confusion_matrix"]) ]],[5.2*cm,10.9*cm]),
          P("Version 2 includes 209 moisture-risk and 91 acceptable observations. The confusion matrix is [[80, 11], [5, 204]] using [acceptable, moisture-risk] as the class order. These results are internal to a synthetic scenario and should not be generalized to real storage facilities.","SmallX"),PageBreak()]

# PAGE 5: MLflow + CI/CD + tests.
story += [P("4. Model Registry, CI/CD and Testing", "H1X"),
          P("4.1 MLflow integration", "H2X"),
          P("src/train.py contains the MLflow integration. With MLflow installed, it creates the experiment manuscript-moisture-risk, logs hyperparameters and evaluation metrics, logs the scikit-learn pipeline, and registers the selected model under the name <b>ManuscriptMoistureRiskModel</b>. The tracking store is configured as a local SQLite database at artifacts/mlflow.db.","BodyX"),
          Preformatted('mlflow.set_tracking_uri("sqlite:///artifacts/mlflow.db")\nmlflow.set_experiment("manuscript-moisture-risk")\nwith mlflow.start_run(run_name="Random Forest-v1"):\n    mlflow.log_metric("recall", validation_recall)\n    mlflow.sklearn.log_model(\n        sk_model=best_model,\n        artifact_path="model",\n        registered_model_name="ManuscriptMoistureRiskModel"\n    )',styles["CodeX"]),
          P("Execution status: MLflow registration was <b>not executed in this build environment</b> because the runtime could not download/install MLflow from the package index. The registration code, dependency and command are included in the project. Install requirements.txt in an internet-enabled Python environment and rerun python -m src.train to create the actual run and registered model version. The report does not claim a successful registration or invent a run ID.","CalloutX"),
          P("4.2 GitHub Actions workflow", "H2X"),
          P(".github/workflows/ci.yml runs on pushes and pull requests to main/master. It checks out the repository, sets up Python 3.11, installs requirements, regenerates both dataset versions, materializes the feature store, trains/registers the model, runs pytest and uploads metrics/metadata as a workflow artifact.","BodyX"),
          P("Execution status: GitHub-hosted Actions was <b>not run</b>, because the project has not been pushed to a GitHub repository from this environment. The workflow is included and ready to run after the user pushes the folder. The Actions URL and hosted-run screenshot should be added after the first push.","CalloutX"),
          P("4.3 Local test results", "H2X"),
          make_table([["Test group","Checks","Result"],
                      ["Feature engineering","Dew point and engineered-feature ranges","Passed"],
                      ["Input validation","Humidity and room-temperature boundary checks","Passed"],
                      ["API","Health route and prediction response behavior","Passed"],
                      ["Full pytest suite","9 tests total","9 passed"]],[3.3*cm,9.0*cm,3.8*cm]),
          P("Command executed locally: <b>python -m pytest -q</b> - output: 9 passed. The API tests allow a 503 when no model artifact exists, while the current local run has a loaded model and returned successful predictions.","BodyX"), PageBreak()]

# PAGE 6: API details and screenshot.
story += [P("5. Model Serving with FastAPI", "H1X"),
          P("src/api.py loads the local Joblib model by default. It can also load a specific registered model through the MLFLOW_MODEL_URI environment variable. The same engineer_features() function runs on incoming requests before the model receives the feature vector.","BodyX"),
          make_table([["Endpoint","Purpose","Result"],
                      ["GET /health","Checks service and model readiness","Returns status and model version"],
                      ["POST /predict","Predicts the 48-hour moisture-risk class","Returns class, probability, threshold and key features"]],[3.2*cm,6.8*cm,6.1*cm]),
          P("Input validation", "H2X"),
          make_table([["Input","Accepted range"],
                      ["Relative humidity","0 to 100 percent"],["Room temperature","5 to 40 C"],
                      ["Wall surface temperature","0 to 45 C"],["Door openings","0 to 60 per hour"],
                      ["Ventilation duration","0 to 60 minutes per hour"],["Dehumidifier operating time","0 to 24 hours per day"],
                      ["Room occupancy","0 to 30"]],[7.5*cm,8.6*cm]),
          P("5.1 Live local API prediction evidence", "H2X"),
          Image(str(ROOT/"evidence"/"api_prediction_screenshot.png"),width=17.0*cm,height=11.05*cm),
          P("The screenshot was created from actual HTTP responses returned by the locally running service. The high-risk test case returned <b>Moisture Risk</b> with probability 0.9637; the acceptable test case returned <b>Storage Conditions Acceptable</b> with probability 0.0. The health endpoint reported model_loaded=true.","SmallX"), PageBreak()]

# PAGE 7: How to run + limitations + checklist.
story += [P("6. Reproducibility and Submission Checklist", "H1X"),
          P("Run these commands from the extracted project root on Windows PowerShell:","BodyX"),
          Preformatted("py -3.11 -m venv .venv\n.\\.venv\\Scripts\\Activate.ps1\npip install -r requirements.txt\npython -m src.run_pipeline\npython -m pytest -q\nuvicorn src.api:app --reload",styles["CodeX"]),
          P("Open http://127.0.0.1:8000/docs for the interactive API documentation. For MLflow, use mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --host 127.0.0.1 --port 5000 after the registry run has been created.","BodyX"),
          P("Files included", "H2X"),
          make_table([["Requirement","Included file(s)"],
                      ["Dataset versions","data/raw/manuscript_rooms_v1.csv; manuscript_rooms_v2_monsoon.csv"],
                      ["Feature pipeline / store","src/features.py; src/feature_store.py; data/processed/feature_store.sqlite"],
                      ["Model comparison","src/train.py; artifacts/metrics.json; artifacts/best_model.joblib"],
                      ["API","src/api.py"],
                      ["CI/CD",".github/workflows/ci.yml"],
                      ["Tests","tests/test_features.py; tests/test_validation.py; tests/test_api.py"],
                      ["Evidence","evidence/api_prediction_screenshot.png; evidence/execution_status.txt"]],[4.0*cm,12.1*cm]),
          P("Before final college submission", "H2X"),
          P("1. Push the extracted project to your GitHub repository. 2. Open the Actions tab and wait for the workflow run to complete. 3. Add the real Actions run URL/screenshot to the evidence folder. 4. Install MLflow in an internet-enabled environment and rerun training to create the registered model; add a screenshot of the experiment and registered model version. 5. Replace synthetic data with approved museum sensor readings before any real operational use.","BodyX"),
          P("Limitations and responsible use", "H2X"),
          P("All observations and labels are synthetic. The risk labels are generated from a transparent rule combining dew-point margin, relative humidity, ventilation, door activity and dehumidifier operation; the model therefore learns a simulated heuristic. Real deployment requires sensor calibration, room-specific history, confirmed conservation outcomes, leakage-safe temporal validation, alert threshold review, and assessment by museum conservation staff. A model probability is not a certainty that manuscript damage will occur.","BodyX"),
          Spacer(1,0.2*cm), P("Project conclusion", "H2X"),
          P("The deliverable implements the requested end-to-end Python MLOps structure: data versioning, reusable engineered features, model comparison, registry integration code, API serving, automated tests and GitHub CI configuration. Local data generation, feature materialization, model training, API inference and 9 tests were executed successfully. Hosted MLflow registration and GitHub Actions remain the two external steps that require an internet-enabled environment and the student's GitHub account.","BodyX")]

doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)
