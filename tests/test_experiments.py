"""Checks for the debate experiments on the Model Evaluation page."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

import ml_service
from conftest import STUDENT


def _held_out():
    bundle = ml_service.load_model()
    return bundle, ml_service._held_out_data(bundle)


def test_held_out_rows_reproduce_saved_predictions(trained):
    bundle, d = _held_out()
    stored = [r["predicted"] for r in bundle["evaluation"]["test_rows"]]
    np.testing.assert_allclose(bundle["pipeline"].predict(d["X_test"]), stored)
    assert set(d["train_ids"]).isdisjoint(d["test_ids"])
    assert len(d["test_ids"]) == trained["model"]["n_test"]


def test_held_out_rows_for_models_saved_without_test_index(trained):
    bundle = ml_service.load_model()
    expected = ml_service._held_out_data(bundle)["test_ids"]
    legacy = {**bundle, "evaluation": {k: v for k, v in bundle["evaluation"].items() if k != "test_index"}}
    assert ml_service._held_out_data(legacy)["test_ids"] == expected


def test_error_counts(client, trained):
    errors = client.get("/api/evaluation").json()["errors"]
    abs_err = np.abs(np.array(trained["scatter"]["predicted"]) - np.array(trained["scatter"]["actual"]))
    assert errors["over_5_points"] == int(np.sum(abs_err > 0.05 + 1e-9))
    assert errors["over_10_points"] == int(np.sum(abs_err > 0.10 + 1e-9))
    ranked = errors["ranked"]
    assert len(ranked) == trained["model"]["n_test"]
    assert [r["abs_error"] for r in ranked] == sorted((r["abs_error"] for r in ranked), reverse=True)


def test_subjective_comparison_uses_same_rows_and_keeps_active_model(client, trained):
    pipeline_file = ml_service.MODEL_DIR / "active" / "pipeline.joblib"
    before = pipeline_file.read_bytes()
    r = client.get("/api/experiments/subjective")
    assert r.status_code == 200
    body = r.json()
    assert pipeline_file.read_bytes() == before
    assert client.get("/api/model").json()["model_id"] == trained["model"]["model_id"]

    _, d = _held_out()
    cfg = trained["model"]["config"]
    academic = ["GRE Score", "TOEFL Score", "CGPA", "Research"]
    pipe_b = ml_service.build_pipeline(cfg["k"], cfg["metric"], cfg["weights"]).fit(d["X_train"][academic], d["y_train"])
    pred_b = pipe_b.predict(d["X_test"][academic])
    pred_a = np.array(trained["scatter"]["predicted"])
    assert body["models"]["academic"]["features"] == academic
    assert body["models"]["academic"]["mae"] == pytest.approx(mean_absolute_error(d["y_test"], pred_b))
    assert body["models"]["all"]["mae"] == pytest.approx(trained["model"]["metrics"]["test"]["mae"])
    assert body["matches_active_model"] is True
    assert body["mean_abs_change"] == pytest.approx(np.abs(pred_b - pred_a).mean())
    assert body["over_5_points"] == int(np.sum(np.abs(pred_b - pred_a) > 0.05 + 1e-9))
    assert body["max_abs_change"] == pytest.approx(np.abs(pred_b - pred_a).max())


def test_subjective_comparison_requires_all_features(client, uploaded):
    client.post("/api/train", json={"upload_id": uploaded["upload_id"], "features": ["GRE Score", "CGPA"]})
    r = client.get("/api/experiments/subjective")
    assert r.status_code == 400 and "all seven features" in r.json()["detail"]


def test_consistency_test(client, trained):
    r = client.get("/api/experiments/consistency")
    assert r.status_code == 200
    body = r.json()
    features = {f["feature"]: f for f in body["features"]}
    totals = body["totals"]
    assert totals["tests"] == sum(f["tests"] for f in body["features"])
    assert totals["decreases"] == sum(f["decreases"] for f in body["features"])
    assert totals["decreases_over_5"] <= totals["decreases_over_2"] <= totals["decreases"]
    assert totals["unique_students"] <= totals["decreases"]
    assert all(e["group"] == "academic" for e in body["academic_examples"])
    assert len(body["academic_examples"]) == min(5, totals["academic_decreases"])

    _, d = _held_out()
    X = d["X_test"]
    assert features["Research"]["tests"] == int((X["Research"] == 0).sum())
    assert features["GRE Score"]["tests"] == int((X["GRE Score"] + 1 <= 340).sum())
    assert features["University Rating"]["tests"] == int((X["University Rating"] + 1 <= 5).sum())

    # Each reported example must reproduce with the active model through the public predict API.
    names = {"GRE Score": "gre_score", "TOEFL Score": "toefl_score", "University Rating": "university_rating",
             "SOP": "sop", "LOR": "lor", "CGPA": "cgpa", "Research": "research"}
    for example in body["examples"]:
        row = X.iloc[d["test_ids"].index(example["student"])]
        student = {names[f]: float(row[f]) for f in names}
        student[names[example["feature"]]] = example["improved_value"]
        predicted = client.post("/api/predict", json=student).json()["chance_of_admit"]
        assert predicted == pytest.approx(example["modified_prediction"])
        assert example["modified_prediction"] < example["original_prediction"]


def test_atypical_analysis(client, trained):
    r = client.get("/api/experiments/atypical")
    assert r.status_code == 200
    body = r.json()
    n = trained["model"]["n_test"]
    assert body["atypical"]["count"] == round(0.2 * n)
    assert body["typical"]["count"] + body["atypical"]["count"] == n
    assert body["error_ratio"] == pytest.approx(body["atypical"]["mean_abs_error"] / body["typical"]["mean_abs_error"])
    students = body["students"]
    atypical_dist = [s["avg_distance"] for s in students if s["group"] == "atypical"]
    typical_dist = [s["avg_distance"] for s in students if s["group"] == "typical"]
    assert min(atypical_dist) >= max(typical_dist)
    expected = spearmanr([s["avg_distance"] for s in students], [s["abs_error"] for s in students])
    assert body["spearman_rho"] == pytest.approx(expected.statistic)


def test_nearest_neighbors_match_saved_pipeline(client, trained):
    student = client.get("/api/experiments/atypical").json()["students"][0]["student"]
    r = client.get(f"/api/experiments/neighbors/{student}")
    assert r.status_code == 200
    body = r.json()
    bundle, d = _held_out()
    assert len(body["neighbors"]) == trained["model"]["config"]["k"]
    # Uniform weights: the prediction is the plain mean of the neighbours' recorded chances.
    assert body["predicted"] == pytest.approx(np.mean([n["chance_of_admit"] for n in body["neighbors"]]))
    scaler = bundle["pipeline"].named_steps["scaler"]
    x = scaler.transform(pd.DataFrame([body["features"]]))
    for n in body["neighbors"]:
        train_row = d["X_train"].iloc[d["train_ids"].index(n["student"])]
        manual = np.linalg.norm(scaler.transform(train_row.to_frame().T) - x)
        assert n["distance"] == pytest.approx(manual)
    assert client.get(f"/api/experiments/neighbors/{d['train_ids'][0]}").status_code == 404


def test_cutoff_comparison(client, trained):
    r = client.get("/api/experiments/cutoff?cutoff=0.7")
    assert r.status_code == 200
    body = r.json()
    actual = np.array(trained["scatter"]["actual"]) >= 0.7
    predicted = np.array(trained["scatter"]["predicted"]) >= 0.7
    assert body["disagreements"] == int(np.sum(actual != predicted))
    assert body["recorded_above_predicted_below"] + body["recorded_below_predicted_above"] == body["disagreements"]
    assert client.get("/api/experiments/cutoff?cutoff=1.5").status_code == 422


def test_experiments_need_a_model(client):
    for path in ("subjective", "consistency", "atypical", "neighbors/1"):
        assert client.get(f"/api/experiments/{path}").status_code == 404


def test_experiments_do_not_change_predictions(client, trained):
    before = client.post("/api/predict", json=STUDENT).json()["chance_of_admit"]
    for path in ("subjective", "consistency", "atypical"):
        client.get(f"/api/experiments/{path}")
    assert client.post("/api/predict", json=STUDENT).json()["chance_of_admit"] == before
