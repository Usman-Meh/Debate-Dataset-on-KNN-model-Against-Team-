"""KNN regression service: data validation, training, persistence,
prediction, evaluation, K analysis and sensitivity experiments."""

import hashlib
import json
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from file_service import FEATURES, TARGET

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"

RANDOM_STATE = 42
METRICS = ("euclidean", "manhattan", "chebyshev")
WEIGHTS = ("uniform", "distance")
MIN_ROWS = 20
MAX_K_ANALYSIS = 20

ACADEMIC_FEATURES = ["GRE Score", "TOEFL Score", "CGPA", "Research"]
REVIEWER_FEATURES = ["SOP", "LOR", "University Rating"]

# Valid input ranges: (min, max, must_be_integer)
FEATURE_RANGES = {
    "GRE Score": (260, 340, False),
    "TOEFL Score": (0, 120, False),
    "University Rating": (1, 5, True),
    "SOP": (1, 5, False),
    "LOR": (1, 5, False),
    "CGPA": (0, 10, False),
    "Research": (0, 1, True),
}
TARGET_RANGE = (0, 1)

_YES_NO = {"yes": 1, "y": 1, "true": 1, "no": 0, "n": 0, "false": 0}

_lock = threading.RLock()
_cache: dict = {"key": None, "bundle": None}


class ModelError(Exception):
    """A user-facing ML problem (bad parameters, missing model, ...)."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def feature_group(feature: str) -> str:
    return "academic" if feature in ACADEMIC_FEATURES else "reviewer"


# ---------------------------------------------------------------- validation


def _coerce_column(series: pd.Series, column: str) -> pd.Series:
    """Convert a column to numbers; Research also accepts Yes/No."""
    if column == "Research" and not pd.api.types.is_numeric_dtype(series):
        series = series.map(lambda v: _YES_NO.get(str(v).strip().lower(), v) if isinstance(v, str) else v)
    return pd.to_numeric(series, errors="coerce")


def validate_rows(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict[int, list[str]]]:
    """Return numeric values for `columns` and per-row error messages.

    Error keys are positional row indexes into `df`."""
    errors: dict[int, list[str]] = {}
    numeric = pd.DataFrame(index=df.index)
    for column in columns:
        raw = df[column]
        values = _coerce_column(raw, column)
        numeric[column] = values
        lo, hi, integer = FEATURE_RANGES.get(column, (*TARGET_RANGE, False))
        for pos, (orig, val) in enumerate(zip(raw.tolist(), values.tolist())):
            if pd.isna(orig) or (isinstance(orig, str) and not orig.strip()):
                msg = f"{column} is missing"
            elif pd.isna(val):
                msg = f"{column} must be a number (got '{orig}')"
            elif not (lo <= val <= hi):
                msg = f"{column} must be between {lo} and {hi} (got {val:g})"
            elif integer and float(val) != int(val):
                msg = f"{column} must be a whole number (got {val:g})"
            else:
                continue
            errors.setdefault(pos, []).append(msg)
    return numeric, errors


def check_columns(df: pd.DataFrame, features: list[str], need_target: bool) -> list[str]:
    required = list(features) + ([TARGET] if need_target else [])
    return [c for c in required if c not in df.columns]


def dataset_validation(df: pd.DataFrame) -> dict:
    """Summary shown after upload: are the admission columns present and valid?"""
    missing = check_columns(df, FEATURES, need_target=True)
    result = {
        "required_columns": FEATURES + [TARGET],
        "missing_columns": missing,
        "extra_columns": [c for c in df.columns if c not in FEATURES + [TARGET]],
        "valid": False,
        "valid_rows": 0,
        "invalid_rows": 0,
        "issues": [],
    }
    if missing:
        result["issues"].append(
            "Missing required column(s): " + ", ".join(missing)
            + ". Column names must match the admissions dataset (e.g. 'GRE Score', 'Chance of Admit')."
        )
        return result
    _, errors = validate_rows(df, FEATURES + [TARGET])
    result["invalid_rows"] = len(errors)
    result["valid_rows"] = len(df) - len(errors)
    for pos in sorted(errors)[:10]:
        result["issues"].append(f"Row {pos + 1}: " + "; ".join(errors[pos]))
    if errors:
        result["issues"].append(f"{len(errors)} row(s) with invalid or missing values will be skipped during training.")
    if result["valid_rows"] < MIN_ROWS:
        result["issues"].append(f"At least {MIN_ROWS} valid rows are needed to train (found {result['valid_rows']}).")
    else:
        result["valid"] = True
    return result


def clean_training_data(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, int]:
    missing = check_columns(df, features, need_target=True)
    if missing:
        raise ModelError("The dataset is missing required column(s): " + ", ".join(missing) + ".")
    numeric, errors = validate_rows(df, features + [TARGET])
    clean = numeric.iloc[[i for i in range(len(numeric)) if i not in errors]].reset_index(drop=True)
    if len(clean) < MIN_ROWS:
        raise ModelError(f"Only {len(clean)} valid row(s) found; at least {MIN_ROWS} are needed to train.")
    return clean, len(errors)


# ---------------------------------------------------------------- training


def _metrics(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def build_pipeline(k: int, metric: str, weights: str) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=k, metric=metric, weights=weights)),
    ])


def k_analysis(X_train: pd.DataFrame, y_train: pd.Series, metric: str, weights: str, current_k: int) -> dict:
    """Cross-validated MAE for a range of K, using only the training data."""
    folds = min(5, len(X_train))
    min_fold_train = len(X_train) - int(np.ceil(len(X_train) / folds))
    max_k = max(1, min(MAX_K_ANALYSIS, min_fold_train))
    cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    k_values, means, stds = [], [], []
    for k in range(1, max_k + 1):
        scores = cross_validate(build_pipeline(k, metric, weights), X_train, y_train,
                                cv=cv, scoring="neg_mean_absolute_error")
        mae = -scores["test_score"]
        k_values.append(k)
        means.append(float(mae.mean()))
        stds.append(float(mae.std()))
    best = int(k_values[int(np.argmin(means))])  # argmin picks the smallest K on ties
    return {
        "k_values": k_values,
        "cv_mae_mean": means,
        "cv_mae_std": stds,
        "best_k": best,
        "best_cv_mae": float(min(means)),
        "current_k": current_k,
        "current_k_cv_mae": means[current_k - 1] if current_k <= max_k else None,
        "folds": folds,
    }


def train_model(df: pd.DataFrame, *, k: int = 5, test_percent: float = 30, metric: str = "euclidean",
                weights: str = "uniform", features: list[str] | None = None,
                source_filename: str = "dataset") -> dict:
    """Train, evaluate and atomically save a new active model. Returns its metadata."""
    features = [f for f in FEATURES if f in (features or FEATURES)]
    if not features:
        raise ModelError("Select at least one input feature.")
    if metric not in METRICS:
        raise ModelError(f"Distance metric must be one of: {', '.join(METRICS)}.")
    if weights not in WEIGHTS:
        raise ModelError(f"Weights must be one of: {', '.join(WEIGHTS)}.")
    if not (10 <= test_percent <= 50):
        raise ModelError("Testing percentage must be between 10% and 50%.")
    if int(k) != k or k < 1:
        raise ModelError("K must be a whole number of at least 1.")
    k = int(k)

    clean, dropped = clean_training_data(df, features)
    X, y = clean[features], clean[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_percent / 100, random_state=RANDOM_STATE)
    if k > len(X_train):
        raise ModelError(f"K = {k} is larger than the number of training samples ({len(X_train)}). "
                         f"Choose K between 1 and {len(X_train)}.")
    if len(X_test) < 2:
        raise ModelError("The test set is too small. Increase the testing percentage or upload more rows.")

    # The scaler lives inside the pipeline, so it is fitted on training rows only.
    pipeline = build_pipeline(k, metric, weights)
    pipeline.fit(X_train, y_train)
    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

    importance = permutation_importance(pipeline, X_test, y_test, scoring="neg_mean_absolute_error",
                                        n_repeats=10, random_state=RANDOM_STATE)
    importance_rows = sorted(
        ({"feature": f, "importance": float(m), "std": float(s), "group": feature_group(f)}
         for f, m, s in zip(features, importance.importances_mean, importance.importances_std)),
        key=lambda r: r["importance"], reverse=True)

    test_rows = X_test.copy()
    test_rows["actual"] = y_test.values
    test_rows["predicted"] = test_pred
    test_rows["error"] = test_pred - y_test.values
    test_rows.insert(0, "row", X_test.index + 1)

    metadata = {
        "model_id": uuid.uuid4().hex,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "features": features,
        "target": TARGET,
        "config": {
            "k": k,
            "test_percent": float(test_percent),
            "train_percent": float(100 - test_percent),
            "metric": metric,
            "weights": weights,
            "random_state": RANDOM_STATE,
            "scaling": "StandardScaler",
        },
        "source_filename": source_filename,
        "n_total": int(len(clean)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "dropped_rows": int(dropped),
        "metrics": {"train": _metrics(y_train, train_pred), "test": _metrics(y_test, test_pred)},
        "feature_stats": {
            f: {"min": float(X_train[f].min()), "max": float(X_train[f].max()),
                "median": float(X_train[f].median()), "mean": float(X_train[f].mean())}
            for f in features
        },
        "sklearn_version": sklearn.__version__,
    }
    evaluation = {
        "train_index": [int(i) for i in X_train.index],
        "train_actual": [float(v) for v in y_train],
        "train_predicted": [float(v) for v in train_pred],
        "test_rows": test_rows.to_dict("records"),
        "permutation_importance": importance_rows,
        "k_analysis": k_analysis(X_train, y_train, metric, weights, k),
    }
    save_model(pipeline, metadata, evaluation, clean)
    return metadata


# ---------------------------------------------------------------- persistence


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _active_dir() -> Path:
    return MODEL_DIR / "active"


def save_model(pipeline: Pipeline, metadata: dict, evaluation: dict, dataset: pd.DataFrame) -> None:
    """Write all artifacts to a temp folder, verify them, then swap it in as active.

    The previous active model stays untouched until the new one is complete."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_DIR / f".tmp-{uuid.uuid4().hex}"
    old = MODEL_DIR / f".old-{uuid.uuid4().hex}"
    tmp.mkdir()
    try:
        joblib.dump(pipeline, tmp / "pipeline.joblib")
        metadata = {**metadata, "pipeline_sha256": _sha256(tmp / "pipeline.joblib")}
        (tmp / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
        dataset.to_csv(tmp / "dataset.csv", index=False)
        (tmp / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        _load_dir(tmp)  # verify the new files load before making them active
        with _lock:
            active = _active_dir()
            if active.exists():
                active.rename(old)
            try:
                tmp.rename(active)
            except OSError:
                if old.exists():
                    old.rename(active)
                raise
            _cache.update(key=None, bundle=None)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(old, ignore_errors=True)


def recover_interrupted_save() -> None:
    """On startup: restore a previous model if a swap was interrupted; drop temp folders."""
    if not MODEL_DIR.exists():
        return
    olds = sorted(MODEL_DIR.glob(".old-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not _active_dir().exists() and olds:
        olds[0].rename(_active_dir())
    for path in list(MODEL_DIR.glob(".tmp-*")) + list(MODEL_DIR.glob(".old-*")):
        shutil.rmtree(path, ignore_errors=True)


def _load_dir(folder: Path) -> dict:
    corrupted = ModelError("The saved model files are damaged or incomplete. Please train a new model.", 500)
    try:
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        pipeline_path = folder / "pipeline.joblib"
        # Only load a pipeline whose hash matches the one this app recorded when saving it.
        if _sha256(pipeline_path) != metadata.get("pipeline_sha256"):
            raise corrupted
        pipeline = joblib.load(pipeline_path)
        evaluation = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
    except ModelError:
        raise
    except Exception:
        raise corrupted
    if not isinstance(pipeline, Pipeline) or list(pipeline.named_steps) != ["scaler", "knn"]:
        raise corrupted
    return {"pipeline": pipeline, "metadata": metadata, "evaluation": evaluation, "folder": folder}


def model_exists() -> bool:
    return (_active_dir() / "metadata.json").exists()


def load_model() -> dict:
    """Load the active model (cached in memory until the files change)."""
    active = _active_dir()
    meta_path = active / "metadata.json"
    if not meta_path.exists():
        raise ModelError("No trained model found. Train a model on the Train Model page first.", 404)
    with _lock:
        key = meta_path.stat().st_mtime_ns
        if _cache["key"] != key:
            _cache.update(key=key, bundle=_load_dir(active))
        return _cache["bundle"]


def clear_cache() -> None:
    with _lock:
        _cache.update(key=None, bundle=None)


def load_saved_dataset() -> pd.DataFrame:
    load_model()
    try:
        return pd.read_csv(_active_dir() / "dataset.csv")
    except Exception:
        raise ModelError("The saved training dataset is missing. Please upload the dataset again.", 500)


# ---------------------------------------------------------------- prediction


def predict_values(rows: list[dict]) -> np.ndarray:
    bundle = load_model()
    features = bundle["metadata"]["features"]
    frame = pd.DataFrame([{f: float(r[f]) for f in features} for r in rows], columns=features)
    return bundle["pipeline"].predict(frame)


def predict_one(values: dict) -> float:
    return float(predict_values([values])[0])


def predict_batch(df: pd.DataFrame) -> dict:
    """Predict every valid row; invalid rows get an error and an empty prediction."""
    bundle = load_model()
    features = bundle["metadata"]["features"]
    missing = check_columns(df, features, need_target=False)
    if missing:
        raise ModelError("The file is missing required column(s): " + ", ".join(missing) + ".")
    numeric, errors = validate_rows(df, features)
    valid_pos = [i for i in range(len(df)) if i not in errors]
    predictions = np.full(len(df), np.nan)
    if valid_pos:
        predictions[valid_pos] = bundle["pipeline"].predict(numeric.iloc[valid_pos][features])
    result = df.copy()
    result["Predicted_Chance_of_Admit"] = np.round(predictions, 4)
    return {
        "result": result,
        "total": len(df),
        "successful": len(valid_pos),
        "invalid": [{"row": pos + 1, "errors": msgs} for pos, msgs in sorted(errors.items())],
    }


# ---------------------------------------------------------------- evaluation


def _performance_notes(meta: dict, k_info: dict) -> list[str]:
    train, test = meta["metrics"]["train"], meta["metrics"]["test"]
    cfg = meta["config"]
    gap = train["r2"] - test["r2"]
    notes = []
    if cfg["weights"] == "distance" or cfg["k"] == 1:
        notes.append("With K = 1 or distance weighting, each training student is its own nearest neighbour, "
                     "so training scores look almost perfect. Judge this model by its testing and "
                     "cross-validation scores instead.")
    elif gap > 0.10:
        notes.append(f"Training R² is {gap:.2f} higher than testing R². That can be a sign of overfitting "
                     "(the model memorizes nearby training students); a larger K often smooths this out.")
    elif test["r2"] < 0.5 and train["r2"] < 0.5:
        notes.append("Both training and testing R² are modest. The model may be underfitting — K could be "
                     "too large, or the selected features may not carry enough information.")
    else:
        notes.append("Training and testing scores are fairly close, which suggests the model generalizes "
                     "reasonably on this particular split.")
    if k_info.get("current_k_cv_mae") is not None:
        notes.append(f"Cross-validated MAE at K = {cfg['k']} is {k_info['current_k_cv_mae']:.3f} versus a test "
                     f"MAE of {test['mae']:.3f}; agreement between the two makes the result more trustworthy.")
    notes.append("These are heuristics, not proof: a single train/test split can be lucky or unlucky, and no "
                 "fixed threshold definitively identifies overfitting or underfitting.")
    return notes


def get_evaluation() -> dict:
    bundle = load_model()
    meta, ev = bundle["metadata"], bundle["evaluation"]
    train, test = meta["metrics"]["train"], meta["metrics"]["test"]
    rows = ev["test_rows"]
    errors = np.array([r["error"] for r in rows])
    top = sorted(rows, key=lambda r: abs(r["error"]), reverse=True)[:10]
    return {
        "model": meta,
        "comparison": {
            "train": train,
            "test": test,
            "gap": {"r2": train["r2"] - test["r2"], "mae": test["mae"] - train["mae"],
                    "rmse": test["rmse"] - train["rmse"]},
        },
        "notes": _performance_notes(meta, ev["k_analysis"]),
        "feature_importance": ev["permutation_importance"],
        "feature_groups": {"academic": [f for f in ACADEMIC_FEATURES if f in meta["features"]],
                           "reviewer": [f for f in REVIEWER_FEATURES if f in meta["features"]]},
        "scatter": {"actual": [r["actual"] for r in rows], "predicted": [r["predicted"] for r in rows]},
        "train_scatter": {"actual": ev["train_actual"], "predicted": ev["train_predicted"]},
        "errors": {
            "mean_absolute_error": float(np.abs(errors).mean()),
            "mean_signed_error": float(errors.mean()),
            "median_absolute_error": float(np.median(np.abs(errors))),
            "max_absolute_error": float(np.abs(errors).max()),
            "top10": top,
        },
    }


def get_k_analysis() -> dict:
    return load_model()["evaluation"]["k_analysis"]


def _sweep_values(feature: str, lo: float, hi: float) -> list[float]:
    if feature == "Research":
        return [0.0, 1.0]
    if feature == "University Rating":
        return [1.0, 2.0, 3.0, 4.0, 5.0]
    if feature in ("SOP", "LOR"):
        return [float(v) for v in np.arange(1.0, 5.01, 0.5)]
    decimals = 2 if feature == "CGPA" else 0
    return sorted({float(round(v, decimals)) for v in np.linspace(lo, hi, 25)})


def sensitivity(profile: dict, feature: str, new_value: float) -> dict:
    """Change one feature of a profile, keep the rest fixed, compare predictions."""
    bundle = load_model()
    meta = bundle["metadata"]
    if feature not in meta["features"]:
        raise ModelError(f"'{feature}' is not used by the trained model. Choose one of: {', '.join(meta['features'])}.")
    lo, hi, integer = FEATURE_RANGES[feature]
    if not (lo <= new_value <= hi) or (integer and float(new_value) != int(new_value)):
        raise ModelError(f"The new {feature} value must be {'a whole number ' if integer else ''}between {lo} and {hi}.")

    original_value = float(profile[feature])
    modified = {**profile, feature: float(new_value)}
    original_pred, modified_pred = predict_values([profile, modified])
    change = float(modified_pred - original_pred)

    stats = meta["feature_stats"][feature]
    sweep_x = sorted(set(_sweep_values(feature, min(stats["min"], original_value, new_value),
                                       max(stats["max"], original_value, new_value)))
                     | {original_value, float(new_value)})
    sweep_y = predict_values([{**profile, feature: v} for v in sweep_x])
    drops = int(np.sum(np.diff(sweep_y) < -1e-9))

    increased = new_value > original_value
    counterintuitive = increased and change < -1e-9
    academic = feature in ACADEMIC_FEATURES
    if counterintuitive and academic:
        message = (f"Increasing {feature} from {original_value:g} to {new_value:g} LOWERED the predicted chance "
                   f"by {abs(change) * 100:.1f} percentage points. A better academic qualification should not "
                   "reduce the estimate — KNN can do this because the new value changes which neighbours are picked.")
    elif counterintuitive:
        message = (f"Increasing {feature} lowered the predicted chance by {abs(change) * 100:.1f} percentage points.")
    elif abs(change) < 1e-9:
        message = "The prediction did not change: the same neighbours were selected for both profiles."
    else:
        message = f"The predicted chance changed by {change * 100:+.1f} percentage points."
    return {
        "feature": feature,
        "group": feature_group(feature),
        "original_value": original_value,
        "modified_value": float(new_value),
        "original_prediction": float(original_pred),
        "modified_prediction": float(modified_pred),
        "change": change,
        "counterintuitive": counterintuitive,
        "academic_violation": counterintuitive and academic,
        "message": message,
        "sweep": {"values": sweep_x, "predictions": [float(v) for v in sweep_y], "drops": drops},
    }
