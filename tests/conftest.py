import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import file_service  # noqa: E402
import ml_service  # noqa: E402
from main import app  # noqa: E402


def make_dataset(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic admissions data shaped like Admission_Predict_Ver1.1.csv."""
    rng = np.random.default_rng(seed)
    cgpa = rng.uniform(6.8, 9.9, n).round(2)
    gre = np.clip(290 + (cgpa - 6.8) * 14 + rng.normal(0, 5, n), 290, 340).round()
    toefl = np.clip(92 + (cgpa - 6.8) * 8 + rng.normal(0, 3, n), 92, 120).round()
    rating = rng.integers(1, 6, n)
    sop = (rng.integers(2, 11, n) / 2).clip(1, 5)
    lor = (rng.integers(2, 11, n) / 2).clip(1, 5)
    research = rng.integers(0, 2, n)
    chance = (0.12 * (cgpa - 6.8) + 0.002 * (gre - 290) + 0.002 * (toefl - 92)
              + 0.01 * rating + 0.03 * research + 0.35 + rng.normal(0, 0.03, n)).clip(0.34, 0.97).round(2)
    return pd.DataFrame({
        "Serial No.": np.arange(1, n + 1), "GRE Score": gre, "TOEFL Score": toefl,
        "University Rating": rating, "SOP": sop, "LOR ": lor, "CGPA": cgpa,
        "Research": research, "Chance of Admit ": chance,
    })


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Every test gets its own empty model and upload folders."""
    monkeypatch.setattr(ml_service, "MODEL_DIR", tmp_path / "saved_models")
    monkeypatch.setattr(file_service, "UPLOAD_DIR", tmp_path / "uploads")
    ml_service.clear_cache()
    yield
    ml_service.clear_cache()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def dataset() -> pd.DataFrame:
    return make_dataset()


@pytest.fixture
def csv_bytes(dataset) -> bytes:
    return dataset.to_csv(index=False).encode()


@pytest.fixture
def uploaded(client, csv_bytes) -> dict:
    r = client.post("/api/upload", files={"file": ("admissions.csv", csv_bytes, "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture
def trained(client, uploaded) -> dict:
    r = client.post("/api/train", json={"upload_id": uploaded["upload_id"]})
    assert r.status_code == 200, r.text
    return r.json()


STUDENT = {"gre_score": 320, "toefl_score": 110, "university_rating": 3, "sop": 3.5,
           "lor": 3.5, "cgpa": 8.6, "research": 1}


def xlsx_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()
