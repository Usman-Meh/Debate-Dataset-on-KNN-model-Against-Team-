"""File handling: upload validation, reading CSV/Excel, column normalization
and short-lived local storage of uploads and batch prediction results."""

import io
import json
import re
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB is far more than any admissions sheet needs
MAX_ROWS = 50_000
UPLOAD_TTL_SECONDS = 2 * 60 * 60  # uploaded applicant data is deleted after 2 hours
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")

# Canonical column names used everywhere in the application.
FEATURES = ["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"]
TARGET = "Chance of Admit"

_ALIASES = {
    "gre score": "GRE Score", "gre": "GRE Score",
    "toefl score": "TOEFL Score", "toefl": "TOEFL Score",
    "university rating": "University Rating", "univ rating": "University Rating",
    "sop": "SOP", "lor": "LOR", "cgpa": "CGPA", "research": "Research",
    "chance of admit": "Chance of Admit", "chance of admission": "Chance of Admit",
}


class FileError(Exception):
    """A user-facing problem with an uploaded file."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# ---------------------------------------------------------------- reading


def read_upload_bytes(fileobj) -> bytes:
    """Read an uploaded file stream, refusing anything over MAX_FILE_SIZE."""
    data = fileobj.read(MAX_FILE_SIZE + 1)
    if len(data) > MAX_FILE_SIZE:
        raise FileError(f"File is too large. The maximum size is {MAX_FILE_SIZE // (1024 * 1024)} MB.", 413)
    if not data:
        raise FileError("The uploaded file is empty.")
    return data


def get_extension(filename: str) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise FileError("Unsupported file format. Please upload a .csv, .xlsx or .xls file.", 415)
    return ext


def check_content(data: bytes, ext: str) -> None:
    """Verify the file content matches its extension (not just the name)."""
    if ext == ".xlsx" and not data.startswith(b"PK\x03\x04"):
        raise FileError("This file is not a valid .xlsx workbook.")
    if ext == ".xls" and not data.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        raise FileError("This file is not a valid .xls workbook.")
    if ext == ".csv" and b"\x00" in data[:4096]:
        raise FileError("This file does not look like a text CSV file.")


def _decode_csv(data: bytes) -> str:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise FileError("Could not read the CSV text encoding. Please save it as UTF-8.")


def list_sheets(data: bytes, ext: str) -> list[str]:
    if ext == ".csv":
        return []
    try:
        engine = "openpyxl" if ext == ".xlsx" else "xlrd"
        return [str(s) for s in pd.ExcelFile(io.BytesIO(data), engine=engine).sheet_names]
    except Exception:
        raise FileError("The Excel file could not be opened. It may be corrupted or password protected.")


def read_table(data: bytes, ext: str, sheet: str | None = None) -> pd.DataFrame:
    """Parse CSV/Excel bytes into a DataFrame with normalized column names."""
    try:
        if ext == ".csv":
            df = pd.read_csv(io.StringIO(_decode_csv(data)), sep=None, engine="python")
        else:
            sheets = list_sheets(data, ext)
            if sheet is not None and sheet not in sheets:
                raise FileError(f"Sheet '{sheet}' was not found in this workbook.")
            engine = "openpyxl" if ext == ".xlsx" else "xlrd"
            df = pd.read_excel(io.BytesIO(data), sheet_name=sheet or sheets[0], engine=engine)
    except FileError:
        raise
    except pd.errors.EmptyDataError:
        raise FileError("The file does not contain any data.")
    except Exception:
        raise FileError("The file could not be parsed. Check that it is a valid table with a header row.")

    df = df.dropna(how="all")
    if df.empty or len(df.columns) == 0:
        raise FileError("The dataset is empty. It needs a header row and at least one data row.")
    if len(df) > MAX_ROWS:
        raise FileError(f"The dataset has too many rows (maximum {MAX_ROWS:,}).")
    original_columns = [str(c) for c in df.columns]
    df = normalize_columns(df).reset_index(drop=True)
    df.attrs["original_columns"] = original_columns
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map variants like 'LOR ' or 'chance_of_admit' to canonical names."""
    renamed = []
    for col in df.columns:
        key = re.sub(r"\s+", " ", str(col).replace("_", " ")).strip().lower()
        renamed.append(_ALIASES.get(key, str(col).strip()))
    duplicates = sorted({c for c in renamed if renamed.count(c) > 1})
    if duplicates:
        raise FileError(f"Duplicate columns after normalizing names: {', '.join(duplicates)}.")
    df = df.copy()
    df.columns = renamed
    return df


# ---------------------------------------------------------------- summaries


def to_json_safe(value):
    """Convert numpy/pandas scalars (including NaN) into JSON-friendly values."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value if isinstance(value, (int, str, bool)) else str(value)


def records(df: pd.DataFrame) -> list[dict]:
    return [{str(k): to_json_safe(v) for k, v in row.items()} for row in df.to_dict("records")]


def summarize(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": [str(c) for c in df.columns],
        "preview": records(df.head(10)),
        "missing_values": {str(c): int(n) for c, n in df.isna().sum().items()},
    }


# ---------------------------------------------------------------- storage


def _cleanup_old_files() -> None:
    cutoff = time.time() - UPLOAD_TTL_SECONDS
    for path in UPLOAD_DIR.glob("*"):
        try:
            if path.is_file() and path.name != ".gitkeep" and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


def _check_id(file_id: str) -> str:
    if not isinstance(file_id, str) or not ID_PATTERN.match(file_id):
        raise FileError("Invalid file reference.")
    return file_id


def store_upload(data: bytes, ext: str, original_name: str) -> str:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_old_files()
    upload_id = uuid.uuid4().hex
    (UPLOAD_DIR / f"{upload_id}{ext}").write_bytes(data)
    meta = {"filename": Path(original_name).name, "ext": ext}
    (UPLOAD_DIR / f"{upload_id}.json").write_text(json.dumps(meta), encoding="utf-8")
    return upload_id


def load_upload(upload_id: str) -> tuple[bytes, str, str]:
    """Return (bytes, extension, original filename) for a stored upload."""
    _check_id(upload_id)
    meta_path = UPLOAD_DIR / f"{upload_id}.json"
    if not meta_path.exists():
        raise FileError("The uploaded file has expired or was not found. Please upload it again.", 404)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    data_path = UPLOAD_DIR / f"{upload_id}{meta['ext']}"
    if not data_path.exists():
        raise FileError("The uploaded file has expired or was not found. Please upload it again.", 404)
    return data_path.read_bytes(), meta["ext"], meta["filename"]


def store_prediction_result(df: pd.DataFrame, source_name: str) -> str:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_old_files()
    result_id = uuid.uuid4().hex
    df.to_csv(UPLOAD_DIR / f"pred-{result_id}.csv", index=False)
    meta = {"filename": Path(source_name).stem}
    (UPLOAD_DIR / f"pred-{result_id}.json").write_text(json.dumps(meta), encoding="utf-8")
    return result_id


def load_prediction_result(result_id: str) -> tuple[pd.DataFrame, str]:
    _check_id(result_id)
    path = UPLOAD_DIR / f"pred-{result_id}.csv"
    if not path.exists():
        raise FileError("These prediction results have expired. Please run the bulk prediction again.", 404)
    meta = json.loads((UPLOAD_DIR / f"pred-{result_id}.json").read_text(encoding="utf-8"))
    return pd.read_csv(path), meta["filename"]


def to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, sheet_name="Predictions", engine="openpyxl")
    return buffer.getvalue()
