# KNN Admission Predictor

A lightweight three-page web app that predicts graduate **Chance of Admit** with a
scikit-learn `KNeighborsRegressor`. FastAPI + Jinja2 on the server; plain HTML, CSS
and vanilla JavaScript (with a locally bundled Chart.js) in the browser. No Node.js,
no build step, no database.

## Setup and run

```bash
cd admission-knn
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open <http://127.0.0.1:8000>. The API docs are at <http://127.0.0.1:8000/docs>.

Run the tests with `pytest`.

## Pages

| Page | Route | What it does |
|---|---|---|
| Train Model | `/train` | Upload a CSV/XLSX/XLS file (drag and drop; you can pick the Excel sheet), see a preview and validation report, set K, the test %, the distance metric, the weights and the features, then train and view the metrics. |
| Predict Admission | `/predict` | Single-student form and bulk prediction from a file, with CSV/XLSX download. |
| Model Evaluation | `/evaluation` | Train vs test metrics, permutation importance, a cross-validated K curve (with an optional retrain), a one-feature-change consistency experiment, and the largest errors. |

## Data format

The required columns are `GRE Score`, `TOEFL Score`, `University Rating`, `SOP`, `LOR`, `CGPA` and
`Research`. Training also needs the target `Chance of Admit` (0–1). Names are matched without regard to
case or extra spaces, so the original `Admission_Predict_Ver1.1.csv` headers (`LOR `, `Chance of Admit `)
work as they are. `Research` also accepts Yes/No.

## How the model is built

- `Pipeline([StandardScaler(), KNeighborsRegressor()])` with a `train_test_split` (`random_state=42`).
  The scaler is fitted on the training rows only.
- Permutation importance is computed on the held-out test set. K is analysed with 5-fold CV on the
  training data only.
- The active model is stored in `saved_models/active/` (`pipeline.joblib`, `metadata.json`,
  `evaluation.json`, `dataset.csv`). New models are written to a temporary folder, verified and then
  swapped in, so a failed training run never replaces a working model. The pipeline's SHA-256 hash
  is checked before it is loaded, and the app only loads models it saved itself.
- Uploaded files and bulk-prediction results in `uploads/` are deleted after 2 hours.

## Project layout

```
main.py          FastAPI routes, Pydantic schemas, error handlers
ml_service.py    validation, training, persistence, prediction, evaluation
file_service.py  upload checks, CSV/Excel reading, temporary storage
templates/       base.html + train/predict/evaluation pages
static/          style.css, app.js, chart.umd.min.js
tests/           pytest suite
```
