"""
FastAPI app serving game outcome predictions from the trained models.

Run:
    uvicorn src.api.main:app --reload

Then visit http://localhost:8000/docs for interactive API docs.
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "models"

FEATURE_COLS = [
    "home_roll_points_for", "home_roll_points_against", "home_roll_point_diff",
    "away_roll_points_for", "away_roll_points_against", "away_roll_point_diff",
    "form_diff",
]

app = FastAPI(
    title="NFL Game Outcome Predictor",
    description="Predicts win probability and point margin for NFL games "
                 "based on each team's trailing form.",
)

_win_model = None
_margin_model = None


class GameFeatures(BaseModel):
    home_roll_points_for: float
    home_roll_points_against: float
    away_roll_points_for: float
    away_roll_points_against: float

    class Config:
        json_schema_extra = {
            "example": {
                "home_roll_points_for": 27.5,
                "home_roll_points_against": 19.0,
                "away_roll_points_for": 21.0,
                "away_roll_points_against": 24.5,
            }
        }


class Prediction(BaseModel):
    home_win_probability: float
    predicted_margin: float
    predicted_winner: str


def _load_models():
    global _win_model, _margin_model
    if _win_model is None or _margin_model is None:
        try:
            _win_model = joblib.load(MODEL_DIR / "win_classifier.joblib")
            _margin_model = joblib.load(MODEL_DIR / "margin_regressor.joblib")
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Models not found. Run `python -m src.models.train` first.",
            )
    return _win_model, _margin_model


@app.get("/")
def root():
    return {"message": "NFL predictor API. See /docs for usage."}


@app.post("/predict", response_model=Prediction)
def predict(features: GameFeatures):
    win_model, margin_model = _load_models()

    home_diff = features.home_roll_points_for - features.home_roll_points_against
    away_diff = features.away_roll_points_for - features.away_roll_points_against
    form_diff = home_diff - away_diff

    row = pd.DataFrame([{
        "home_roll_points_for": features.home_roll_points_for,
        "home_roll_points_against": features.home_roll_points_against,
        "home_roll_point_diff": home_diff,
        "away_roll_points_for": features.away_roll_points_for,
        "away_roll_points_against": features.away_roll_points_against,
        "away_roll_point_diff": away_diff,
        "form_diff": form_diff,
    }])[FEATURE_COLS]

    win_prob = float(win_model.predict_proba(row)[0, 1])
    margin = float(margin_model.predict(row)[0])

    return Prediction(
        home_win_probability=round(win_prob, 3),
        predicted_margin=round(margin, 1),
        predicted_winner="home" if win_prob >= 0.5 else "away",
    )
