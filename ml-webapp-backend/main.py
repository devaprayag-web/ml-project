import io
import math
import os
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.schemas import UploadResponse, ColumnInfo, TrainRequest, TrainResponse
from app.session_store import session_store
from app.model_store import model_store
from app.preprocessing import drop_all_null_columns
from app.ml import train_and_evaluate

app = FastAPI(title="No-Code ML Trainer API")

# Allowed frontend origins. Defaults to the local Vite dev server; set the
# ALLOWED_ORIGINS env var (comma-separated) to your deployed frontend URL(s)
# in production, e.g. ALLOWED_ORIGINS=https://your-app.vercel.app
_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
allowed_origins = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", _default_origins).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _clean_for_json(value):
    """
    Make a pandas cell value safe to return as JSON.

    Two separate problems handled here:
    1. NaN isn't valid JSON - convert to None.
    2. numpy scalar types (numpy.int64, numpy.float64, numpy.bool_) aren't
       recognized by Pydantic's serializer and raise a 500 error. This bites
       specifically when a DataFrame's columns are all the same dtype (e.g.
       an all-integer CSV) - pandas' iterrows() returns raw numpy scalars in
       that case, but happens to return native Python types when columns are
       mixed dtypes. Rather than rely on that inconsistency, explicitly
       convert any numpy scalar to its native Python equivalent via .item().
    """
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


@app.post("/api/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    try:
        raw_bytes = await file.read()
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV has no rows.")

    # Drop fully-null columns up front so the preview and target-column picker
    # in the frontend don't offer a column the model can never use.
    df, dropped_columns = drop_all_null_columns(df)

    if df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="All columns in this CSV are entirely null.")

    session_id = session_store.create(df)

    columns = [
        ColumnInfo(
            name=col,
            dtype=str(df[col].dtype),
            null_count=int(df[col].isnull().sum()),
            unique_count=int(df[col].nunique()),
        )
        for col in df.columns
    ]

    preview_df = df.head(5)
    preview_rows = [
        {col: _clean_for_json(row[col]) for col in df.columns}
        for _, row in preview_df.iterrows()
    ]

    return UploadResponse(
        session_id=session_id,
        row_count=len(df),
        columns=columns,
        preview_rows=preview_rows,
        dropped_columns=dropped_columns,
    )


@app.post("/api/train", response_model=TrainResponse)
async def train(request: TrainRequest):
    try:
        df = session_store.get(request.session_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. Please re-upload your CSV."
        )

    try:
        problem_type, metrics, extras, sample_predictions, model_artifact = train_and_evaluate(
            df, request.algorithm.value, request.target_column, request.k, request.auto_k
        )
    except ValueError as e:
        # Expected/handled edge cases (bad target column, non-numeric target for
        # regression, all-null target, etc.) -> 400, not a 500 crash.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

    model_id = model_store.create(model_artifact)

    return TrainResponse(
        problem_type=problem_type,
        algorithm=request.algorithm,
        metrics=metrics,
        extras=extras,
        sample_predictions=sample_predictions,
        model_id=model_id,
    )


@app.get("/api/export/{model_id}")
async def export_model(model_id: str):
    """
    Downloads the trained model (plus its fitted Preprocessor, scaler, and
    KNN feature selector, if any) as a single joblib file, so it can be
    loaded and reused outside this app:

        import joblib
        bundle = joblib.load("model.joblib")
        # bundle["preprocessor"].transform(new_raw_df) -> apply bundle["scaler"]
        # and bundle["knn_selector"] if present -> bundle["model"].predict(...)

    joblib is used rather than pickle directly because it's the format
    scikit-learn's own docs recommend for estimators - it's more efficient
    than pickle for the numpy arrays a fitted model/encoder holds internally.
    """
    try:
        bundle = model_store.get(model_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail="Model not found or expired. Train a model first, then export it.",
        )

    buffer = io.BytesIO()
    joblib.dump(bundle, buffer)
    buffer.seek(0)

    filename = f"model_{bundle['algorithm']}_{model_id[:8]}.joblib"
    return StreamingResponse(
        buffer,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}
