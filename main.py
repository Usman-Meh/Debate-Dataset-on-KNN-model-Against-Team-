"""FastAPI application: three server-rendered pages plus a small JSON API."""

import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

import file_service as fs
import ml_service as ml

BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger("admission-knn")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ml.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fs.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ml.recover_interrupted_save()
    yield


app = FastAPI(
    title="KNN Admission Predictor",
    description="Predict graduate admission chances with K-Nearest Neighbors regression.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# ---------------------------------------------------------------- schemas


class StudentInput(BaseModel):
    gre_score: float = Field(..., ge=260, le=340, title="GRE Score", examples=[320])
    toefl_score: float = Field(..., ge=0, le=120, title="TOEFL Score", examples=[110])
    university_rating: int = Field(..., ge=1, le=5, title="University Rating", examples=[3])
    sop: float = Field(..., ge=1, le=5, title="SOP", examples=[3.5])
    lor: float = Field(..., ge=1, le=5, title="LOR", examples=[3.5])
    cgpa: float = Field(..., ge=0, le=10, title="CGPA", examples=[8.6])
    research: int = Field(..., ge=0, le=1, title="Research (1 = yes, 0 = no)", examples=[1])

    def to_features(self) -> dict:
        return {
            "GRE Score": self.gre_score, "TOEFL Score": self.toefl_score,
            "University Rating": self.university_rating, "SOP": self.sop, "LOR": self.lor,
            "CGPA": self.cgpa, "Research": self.research,
        }


class TrainRequest(BaseModel):
    source: Literal["upload", "active_model"] = Field(
        "upload", description="'upload' trains on an uploaded file; 'active_model' retrains on the active model's dataset.")
    upload_id: Optional[str] = Field(None, pattern=r"^[a-f0-9]{32}$")
    sheet: Optional[str] = None
    k: int = Field(5, ge=1, le=500)
    test_percent: float = Field(30, ge=10, le=50)
    metric: Literal["euclidean", "manhattan", "chebyshev"] = "euclidean"
    weights: Literal["uniform", "distance"] = "uniform"
    features: list[str] = Field(default_factory=lambda: list(fs.FEATURES))

    @field_validator("features")
    @classmethod
    def check_features(cls, value: list[str]) -> list[str]:
        unknown = [f for f in value if f not in fs.FEATURES]
        if unknown:
            raise ValueError(f"unknown feature(s): {', '.join(unknown)}")
        if not value:
            raise ValueError("select at least one feature")
        return value


class SheetRequest(BaseModel):
    upload_id: str = Field(..., pattern=r"^[a-f0-9]{32}$")
    sheet: str


class SensitivityRequest(BaseModel):
    profile: StudentInput
    feature: Literal["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"]
    new_value: float


# ---------------------------------------------------------------- errors


@app.exception_handler(fs.FileError)
@app.exception_handler(ml.ModelError)
async def app_error_handler(_request: Request, exc):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, exc: RequestValidationError):
    messages = []
    for err in exc.errors():
        field = " → ".join(str(p) for p in err["loc"] if p not in ("body", "query", "profile"))
        messages.append(f"{field.replace('_', ' ')}: {err['msg']}" if field else err["msg"])
    return JSONResponse(status_code=422, content={"detail": "; ".join(messages) or "Invalid request."})


@app.exception_handler(Exception)
async def unexpected_error_handler(_request: Request, exc: Exception):
    logger.exception("Unexpected server error", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "An unexpected server error occurred. Please try again."})


# ---------------------------------------------------------------- pages


def _model_status() -> dict:
    try:
        return {"model": ml.load_model()["metadata"], "model_error": None}
    except ml.ModelError as exc:
        return {"model": None, "model_error": exc.message}


def _page(request: Request, name: str, active: str):
    context = {"active": active, "features": fs.FEATURES, "ranges": ml.FEATURE_RANGES,
               "academic": ml.ACADEMIC_FEATURES, **_model_status()}
    return templates.TemplateResponse(request, name, context)


@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/train")


@app.get("/train", include_in_schema=False)
def train_page(request: Request):
    return _page(request, "train.html", "train")


@app.get("/predict", include_in_schema=False)
def predict_page(request: Request):
    return _page(request, "predict.html", "predict")


@app.get("/evaluation", include_in_schema=False)
def evaluation_page(request: Request):
    return _page(request, "evaluation.html", "evaluation")


# ---------------------------------------------------------------- API


def _read_uploaded_file(file: UploadFile) -> tuple[bytes, str]:
    ext = fs.get_extension(file.filename)
    data = fs.read_upload_bytes(file.file)
    fs.check_content(data, ext)
    return data, ext


def _dataset_summary(upload_id: str, data: bytes, ext: str, filename: str, sheet: Optional[str]) -> dict:
    sheets = fs.list_sheets(data, ext)
    selected = sheet or (sheets[0] if sheets else None)
    df = fs.read_table(data, ext, selected)
    return {
        "upload_id": upload_id,
        "filename": filename,
        "sheets": sheets,
        "sheet": selected,
        **fs.summarize(df),
        "validation": ml.dataset_validation(df),
    }


@app.post("/api/upload", tags=["training"])
def upload_dataset(file: UploadFile = File(...), sheet: Optional[str] = Form(None)):
    """Upload a CSV/Excel training dataset and get a preview plus validation report."""
    data, ext = _read_uploaded_file(file)
    filename = Path(file.filename).name
    summary = _dataset_summary("", data, ext, filename, sheet)  # parse before storing anything
    summary["upload_id"] = fs.store_upload(data, ext, filename)
    return summary


@app.post("/api/upload/sheet", tags=["training"])
def select_sheet(req: SheetRequest):
    """Re-read a previously uploaded Excel workbook using a different sheet."""
    data, ext, filename = fs.load_upload(req.upload_id)
    return _dataset_summary(req.upload_id, data, ext, filename, req.sheet)


@app.post("/api/train", tags=["training"])
def train(req: TrainRequest):
    """Train a StandardScaler + KNeighborsRegressor pipeline and make it the active model."""
    if req.source == "active_model":
        df = ml.load_saved_dataset()
        filename = ml.load_model()["metadata"]["source_filename"]
    else:
        if not req.upload_id:
            raise fs.FileError("Upload a dataset before training.")
        data, ext, filename = fs.load_upload(req.upload_id)
        df = fs.read_table(data, ext, req.sheet)
    metadata = ml.train_model(df, k=req.k, test_percent=req.test_percent, metric=req.metric,
                              weights=req.weights, features=req.features, source_filename=filename)
    ev = ml.get_evaluation()
    return {"model": metadata, "scatter": ev["scatter"], "train_scatter": ev["train_scatter"]}


@app.get("/api/model", tags=["model"])
def get_model():
    """Metadata of the active trained model."""
    return ml.load_model()["metadata"]


@app.post("/api/predict", tags=["prediction"])
def predict(student: StudentInput):
    """Predict the Chance of Admit for one student using the saved model."""
    chance = ml.predict_one(student.to_features())
    return {"chance_of_admit": chance, "percentage": round(chance * 100, 1),
            "features_used": ml.load_model()["metadata"]["features"]}


@app.post("/api/predict-batch", tags=["prediction"])
def predict_batch(file: UploadFile = File(...), sheet: Optional[str] = Form(None)):
    """Predict for every row of an uploaded CSV/Excel file. Invalid rows are reported, not dropped."""
    data, ext = _read_uploaded_file(file)
    sheets = fs.list_sheets(data, ext)
    df = fs.read_table(data, ext, sheet)
    original_columns = df.attrs.get("original_columns", list(df.columns))
    out = ml.predict_batch(df)
    out["result"].columns = original_columns + ["Predicted_Chance_of_Admit"]  # keep the user's headers
    result_id = fs.store_prediction_result(out["result"], Path(file.filename).name)
    return {
        "result_id": result_id,
        "filename": Path(file.filename).name,
        "sheets": sheets,
        "sheet": sheet or (sheets[0] if sheets else None),
        "total": out["total"],
        "successful": out["successful"],
        "invalid_count": len(out["invalid"]),
        "invalid": out["invalid"][:500],
        "columns": [str(c) for c in out["result"].columns],
        "rows": fs.records(out["result"].head(1000)),
        "truncated": out["total"] > 1000,
    }


@app.get("/api/predict-batch/{result_id}/download", tags=["prediction"])
def download_predictions(result_id: str, format: Literal["csv", "xlsx"] = Query("csv")):
    """Download bulk prediction results as CSV or XLSX."""
    df, stem = fs.load_prediction_result(result_id)
    name = f"{stem}_predictions.{format}"
    if format == "xlsx":
        body = fs.to_xlsx_bytes(df)
        media = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        body = df.to_csv(index=False).encode("utf-8")
        media = "text/csv"
    return StreamingResponse(io.BytesIO(body), media_type=media,
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/evaluation", tags=["evaluation"])
def evaluation():
    """Performance comparison, permutation importance and prediction errors of the active model."""
    return ml.get_evaluation()


@app.get("/api/k-analysis", tags=["evaluation"])
def k_analysis():
    """Cross-validated MAE for K = 1..20 computed on the training data only."""
    return ml.get_k_analysis()


@app.post("/api/sensitivity", tags=["evaluation"])
def sensitivity(req: SensitivityRequest):
    """Change one feature of a student profile and compare the two predictions."""
    return ml.sensitivity(req.profile.to_features(), req.feature, req.new_value)
