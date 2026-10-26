import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

import ml_service
from conftest import STUDENT, make_dataset, xlsx_bytes

REAL_DATASET = Path(__file__).resolve().parents[2] / "Admission_Predict_Ver1.1.csv"


# ---------------------------------------------------------------- upload


def test_upload_csv_summary(uploaded):
    assert uploaded["rows"] == 200
    assert uploaded["columns"] == 9
    assert "LOR" in uploaded["column_names"] and "Chance of Admit" in uploaded["column_names"]
    assert len(uploaded["preview"]) == 10
    assert uploaded["missing_values"]["CGPA"] == 0
    assert uploaded["validation"]["valid"] is True
    assert uploaded["validation"]["extra_columns"] == ["Serial No."]


def test_upload_rejects_bad_files(client):
    r = client.post("/api/upload", files={"file": ("data.txt", b"a,b\n1,2", "text/plain")})
    assert r.status_code == 415
    r = client.post("/api/upload", files={"file": ("fake.xlsx", b"GRE Score,CGPA\n1,2", "application/octet-stream")})
    assert r.status_code == 400 and "not a valid .xlsx" in r.json()["detail"]
    r = client.post("/api/upload", files={"file": ("empty.csv", b"", "text/csv")})
    assert r.status_code == 400 and "empty" in r.json()["detail"]


def test_upload_reports_missing_columns(client, dataset):
    data = dataset.drop(columns=["CGPA"]).to_csv(index=False).encode()
    r = client.post("/api/upload", files={"file": ("d.csv", data, "text/csv")})
    assert r.status_code == 200
    v = r.json()["validation"]
    assert v["valid"] is False and v["missing_columns"] == ["CGPA"]


def test_upload_excel_sheet_selection(client, dataset):
    data = xlsx_bytes({"Notes": pd.DataFrame({"info": ["readme"]}), "Data": dataset})
    r = client.post("/api/upload", files={"file": ("book.xlsx", data, "application/vnd.ms-excel")})
    assert r.status_code == 200
    body = r.json()
    assert body["sheets"] == ["Notes", "Data"] and body["validation"]["valid"] is False
    r = client.post("/api/upload/sheet", json={"upload_id": body["upload_id"], "sheet": "Data"})
    assert r.status_code == 200 and r.json()["validation"]["valid"] is True
    t = client.post("/api/train", json={"upload_id": body["upload_id"], "sheet": "Data"})
    assert t.status_code == 200, t.text


# ---------------------------------------------------------------- training


def test_training_70_30_split(trained):
    model = trained["model"]
    assert model["config"]["test_percent"] == 30 and model["config"]["train_percent"] == 70
    assert model["n_train"] == 140 and model["n_test"] == 60
    assert model["config"]["k"] == 5 and model["config"]["metric"] == "euclidean"
    for split in ("train", "test"):
        assert set(model["metrics"][split]) == {"r2", "mae", "rmse"}
    assert len(trained["scatter"]["actual"]) == 60
    active = ml_service.MODEL_DIR / "active"
    for name in ("pipeline.joblib", "metadata.json", "evaluation.json", "dataset.csv"):
        assert (active / name).exists()


def test_invalid_k(client, uploaded):
    r = client.post("/api/train", json={"upload_id": uploaded["upload_id"], "k": 141})
    assert r.status_code == 400 and "larger than the number of training samples" in r.json()["detail"]
    r = client.post("/api/train", json={"upload_id": uploaded["upload_id"], "k": 0})
    assert r.status_code == 422


def test_invalid_test_percentage(client, uploaded):
    r = client.post("/api/train", json={"upload_id": uploaded["upload_id"], "test_percent": 80})
    assert r.status_code == 422


def test_scaler_fitted_on_training_data_only(trained):
    bundle = ml_service.load_model()
    scaler = bundle["pipeline"].named_steps["scaler"]
    data = pd.read_csv(ml_service.MODEL_DIR / "active" / "dataset.csv")
    features = bundle["metadata"]["features"]
    train_rows = data.loc[bundle["evaluation"]["train_index"], features]
    np.testing.assert_allclose(scaler.mean_, train_rows.mean().values)
    assert not np.allclose(scaler.mean_, data[features].mean().values)
    assert scaler.n_samples_seen_ == bundle["metadata"]["n_train"]


def test_model_save_and_reload_gives_identical_prediction(client, trained):
    before = client.post("/api/predict", json=STUDENT).json()["chance_of_admit"]
    ml_service.clear_cache()  # simulate a server restart
    from main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as fresh:
        after = fresh.post("/api/predict", json=STUDENT).json()["chance_of_admit"]
    assert before == after


def test_failed_training_keeps_working_model(client, trained, uploaded):
    model_id = trained["model"]["model_id"]
    r = client.post("/api/train", json={"upload_id": uploaded["upload_id"], "k": 200})
    assert r.status_code == 400
    assert client.get("/api/model").json()["model_id"] == model_id


def test_retrain_from_active_model_dataset(client, trained):
    r = client.post("/api/train", json={"source": "active_model", "k": 9})
    assert r.status_code == 200
    assert r.json()["model"]["config"]["k"] == 9
    assert r.json()["model"]["model_id"] != trained["model"]["model_id"]


def test_feature_subset_training(client, uploaded):
    r = client.post("/api/train", json={"upload_id": uploaded["upload_id"], "features": ["CGPA", "GRE Score"]})
    assert r.status_code == 200
    assert r.json()["model"]["features"] == ["GRE Score", "CGPA"]  # canonical order kept
    assert client.post("/api/predict", json=STUDENT).status_code == 200


def test_corrupted_model_reports_clear_error(client, trained):
    (ml_service.MODEL_DIR / "active" / "pipeline.joblib").write_bytes(b"not a model")
    ml_service.clear_cache()
    r = client.get("/api/model")
    assert r.status_code == 500 and "damaged" in r.json()["detail"]
    assert client.get("/predict").status_code == 200


def test_missing_model(client):
    r = client.post("/api/predict", json=STUDENT)
    assert r.status_code == 404 and "Train a model" in r.json()["detail"]
    page = client.get("/predict")
    assert page.status_code == 200 and "No trained model yet" in page.text
    assert client.get("/api/evaluation").status_code == 404


# ---------------------------------------------------------------- prediction


def test_single_prediction(client, trained):
    r = client.post("/api/predict", json=STUDENT)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["chance_of_admit"] <= 1
    assert body["percentage"] == round(body["chance_of_admit"] * 100, 1)
    pipeline = ml_service.load_model()["pipeline"]
    row = pd.DataFrame([{"GRE Score": 320, "TOEFL Score": 110, "University Rating": 3, "SOP": 3.5,
                         "LOR": 3.5, "CGPA": 8.6, "Research": 1}])
    assert body["chance_of_admit"] == pytest.approx(pipeline.predict(row)[0])


def test_single_prediction_validation(client, trained):
    r = client.post("/api/predict", json={**STUDENT, "gre_score": 400})
    assert r.status_code == 422 and "gre score" in r.json()["detail"]
    missing = {k: v for k, v in STUDENT.items() if k != "cgpa"}
    assert client.post("/api/predict", json=missing).status_code == 422


def test_bulk_predictions(client, trained):
    df = make_dataset(12, seed=5).drop(columns=["Chance of Admit "])
    df["Student"] = [f"S{i}" for i in range(12)]
    df["GRE Score"] = df["GRE Score"].astype(object)
    df.loc[3, "GRE Score"] = "abc"
    df.loc[7, "CGPA"] = np.nan
    df["Research"] = np.where(df["Research"] == 1, "Yes", "No")
    data = df.to_csv(index=False).encode()
    r = client.post("/api/predict-batch", files={"file": ("batch.csv", data, "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 12 and body["successful"] == 10 and body["invalid_count"] == 2
    assert [i["row"] for i in body["invalid"]] == [4, 8]
    assert [row["Student"] for row in body["rows"]] == [f"S{i}" for i in range(12)]
    assert body["rows"][3]["Predicted_Chance_of_Admit"] is None

    csv = client.get(f"/api/predict-batch/{body['result_id']}/download?format=csv")
    assert csv.status_code == 200
    out = pd.read_csv(io.BytesIO(csv.content))
    assert list(out.columns) == list(df.columns) + ["Predicted_Chance_of_Admit"]
    assert out["Student"].tolist() == df["Student"].tolist()

    xlsx = client.get(f"/api/predict-batch/{body['result_id']}/download?format=xlsx")
    assert xlsx.status_code == 200 and xlsx.content[:2] == b"PK"


def test_bulk_missing_columns(client, trained):
    data = pd.DataFrame({"GRE Score": [320]}).to_csv(index=False).encode()
    r = client.post("/api/predict-batch", files={"file": ("b.csv", data, "text/csv")})
    assert r.status_code == 400 and "missing required column" in r.json()["detail"]


# ---------------------------------------------------------------- evaluation


def test_evaluation_metrics(client, trained):
    r = client.get("/api/evaluation")
    assert r.status_code == 200
    ev = r.json()
    assert ev["comparison"]["test"]["r2"] == pytest.approx(r2_score(ev["scatter"]["actual"], ev["scatter"]["predicted"]))
    assert ev["comparison"]["gap"]["r2"] == pytest.approx(ev["comparison"]["train"]["r2"] - ev["comparison"]["test"]["r2"])
    assert {i["feature"] for i in ev["feature_importance"]} == set(ml_service.FEATURES)
    top = ev["errors"]["top10"]
    assert len(top) == 10
    assert [abs(t["error"]) for t in top] == sorted((abs(t["error"]) for t in top), reverse=True)
    assert ev["notes"]


def test_k_analysis_uses_cross_validation(client, trained):
    k = client.get("/api/k-analysis").json()
    assert k["k_values"] == list(range(1, 21))
    assert k["best_k"] == k["k_values"][int(np.argmin(k["cv_mae_mean"]))]
    assert k["folds"] == 5 and k["current_k"] == 5


def test_sensitivity(client, trained):
    same = client.post("/api/sensitivity", json={"profile": STUDENT, "feature": "CGPA", "new_value": 8.6}).json()
    assert same["change"] == 0 and same["counterintuitive"] is False

    r = client.post("/api/sensitivity", json={"profile": STUDENT, "feature": "GRE Score", "new_value": 330})
    assert r.status_code == 200
    body = r.json()
    modified = client.post("/api/predict", json={**STUDENT, "gre_score": 330}).json()["chance_of_admit"]
    assert body["modified_prediction"] == pytest.approx(modified)
    assert body["academic_violation"] == (body["change"] < -1e-9)
    assert 330.0 in body["sweep"]["values"] and 320.0 in body["sweep"]["values"]

    bad = client.post("/api/sensitivity", json={"profile": STUDENT, "feature": "CGPA", "new_value": 12})
    assert bad.status_code == 400


# ---------------------------------------------------------------- pages & real data


def test_pages_render(client, trained):
    assert client.get("/", follow_redirects=False).headers["location"] == "/train"
    for path in ("/train", "/predict", "/evaluation"):
        page = client.get(path)
        assert page.status_code == 200 and "Model Evaluation" in page.text
    assert client.get("/docs").status_code == 200
    assert client.get("/static/js/app.js").status_code == 200


@pytest.mark.skipif(not REAL_DATASET.exists(), reason="original dataset not available")
def test_real_admissions_dataset(client):
    r = client.post("/api/upload", files={"file": ("Admission_Predict_Ver1.1.csv", REAL_DATASET.read_bytes(), "text/csv")})
    assert r.status_code == 200 and r.json()["validation"]["valid"]
    t = client.post("/api/train", json={"upload_id": r.json()["upload_id"]}).json()
    assert t["model"]["n_train"] == 350 and t["model"]["n_test"] == 150
    assert t["model"]["metrics"]["test"]["r2"] > 0.7
