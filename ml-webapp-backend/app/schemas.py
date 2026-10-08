from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from enum import Enum


class Algorithm(str, Enum):
    linear_regression = "linear_regression"
    logistic_regression = "logistic_regression"
    knn = "knn"
    decision_tree = "decision_tree"
    random_forest = "random_forest"


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    null_count: int
    unique_count: int


class UploadResponse(BaseModel):
    session_id: str
    row_count: int
    columns: List[ColumnInfo]
    preview_rows: List[Dict[str, Any]]
    dropped_columns: List[str]  # columns dropped because they were 100% null


class TrainRequest(BaseModel):
    session_id: str
    algorithm: Algorithm
    target_column: str
    k: Optional[int] = 5  # only used for KNN, defaults to 5. Ignored if auto_k=True.
    auto_k: Optional[bool] = False  # only used for KNN: run elbow-method CV search to pick k automatically


class TrainResponse(BaseModel):
    problem_type: str  # "regression" or "classification"
    algorithm: Algorithm
    metrics: Dict[str, Any]
    extras: Dict[str, Any]  # algorithm-specific: coefficients, feature_importances, plot data, etc.
    sample_predictions: List[Dict[str, Any]]
    model_id: str  # pass to GET /api/export/{model_id} to download this trained model as a .joblib file
