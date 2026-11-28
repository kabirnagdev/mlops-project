"""FastAPI service for the MP033 canteen demand model."""

import json
import logging
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"
FEATURES = [
    "students_on_campus",
    "menu_price_rs",
    "temp_c",
    "exam_days_left",
    "special_dishes",
    "queue_minutes_yesterday",
    "day_type",
    "block",
]

logger = logging.getLogger("mp033.api")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

app = FastAPI(title="MP033 Canteen Demand API", version="1.0")
_model = None


class PredictRequest(BaseModel):
    """Features known when forecasting canteen lunch demand."""

    model_config = ConfigDict(extra="forbid")

    students_on_campus: int = Field(ge=400)
    menu_price_rs: float = Field(gt=0, le=200)
    temp_c: float
    exam_days_left: int = Field(ge=0)
    special_dishes: int = Field(ge=0)
    queue_minutes_yesterday: float = Field(ge=0)
    day_type: Literal["weekday", "saturday", "holiday"]
    block: Literal["A-Block", "B-Block", "Hostel"]


def get_model():
    """Load the model lazily so health checks and tests do not need an artifact."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {MODEL_PATH}. Train/export the champion first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    features = request.model_dump()
    prediction = float(get_model().predict(pd.DataFrame([features], columns=FEATURES))[0])
    logger.info(json.dumps({
        "event": "prediction",
        "record": features,
        "prediction": prediction,
    }, separators=(",", ":"), sort_keys=True))
    return {"prediction": prediction}
