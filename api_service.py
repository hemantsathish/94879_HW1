from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import json
import joblib
import numpy as np
from pathlib import Path


class PredictRequest(BaseModel):
    features: Dict[str, Any]


def load_artifacts(artifacts_dir: Path = None):
    if artifacts_dir is None:
        artifacts_dir = Path(__file__).parent / "artifacts"

    features_path = artifacts_dir / "features.json"
    model_path = artifacts_dir / "model.pkl"

    if not features_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Required artifacts not found in {artifacts_dir}")

    with open(features_path, "r") as f:
        meta = json.load(f)
    feature_order = meta["feature_order"]
    model = joblib.load(model_path)
    return feature_order, model


# Load model and features on startup
FEATURE_ORDER, MODEL = load_artifacts()

app = FastAPI(title="Air Quality Prediction API", version="1.0.0")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model": "local",
        "features": FEATURE_ORDER[:5],  # Show first 5 features
    }


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    # Validate feature presence
    missing = [f for f in FEATURE_ORDER if f not in req.features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={"error": "missing_features", "missing": missing[:10]},
        )

    try:
        # Convert features to float array, replacing None/nan with 0
        features = []
        for f in FEATURE_ORDER:
            val = req.features.get(f)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                features.append(0.0)
            else:
                features.append(float(val))

        X = np.array(features, dtype=float).reshape(1, -1)
        yhat = float(MODEL.predict(X)[0])

        return {"prediction": yhat}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail={"error": "prediction_failed", "message": str(e)}
        )
