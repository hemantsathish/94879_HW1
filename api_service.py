from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import json
import joblib
import numpy as np
from pathlib import Path
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
import time

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
    feature_order = meta["features"]
    model = joblib.load(model_path)
    return feature_order, model

# Load model and features on startup
FEATURE_ORDER, MODEL = load_artifacts()

# Prometheus metrics - Counters
prediction_counter = Counter(
    'api_predictions_total', 
    'Total number of predictions made'
)

prediction_errors = Counter(
    'api_prediction_errors_total',
    'Total number of prediction errors',
    ['error_type']
)

# Histograms (for distributions)
prediction_latency = Histogram(
    'api_prediction_latency_seconds',
    'Prediction request latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

prediction_value = Histogram(
    'api_prediction_value',
    'Distribution of prediction values',
    buckets=[0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300]
)

# Gauges (for standalone values) ← NEW
latest_prediction = Gauge(
    'api_latest_prediction',
    'Latest prediction value'
)

latest_latency = Gauge(
    'api_latest_latency_seconds',
    'Latest request latency in seconds'
)

missing_features_gauge = Gauge(
    'api_missing_features',
    'Number of missing features in last request'
)

app = FastAPI(title="Air Quality Prediction API", version="1.0.0")

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model": "local",
        "features": FEATURE_ORDER[:5],
    }

@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    start_time = time.time()
    
    # Validate feature presence
    missing = [f for f in FEATURE_ORDER if f not in req.features]
    missing_features_gauge.set(len(missing))
    
    if missing:
        prediction_errors.labels(error_type='missing_features').inc()
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
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Record metrics - Histograms
        prediction_counter.inc()
        prediction_value.observe(yhat)
        prediction_latency.observe(latency)
        
        # Record metrics - Gauges (standalone values) ← NEW
        latest_prediction.set(yhat)
        latest_latency.set(latency)
        
        return {"prediction": yhat}
    except Exception as e:
        prediction_errors.labels(error_type='prediction_failed').inc()
        raise HTTPException(
            status_code=500, detail={"error": "prediction_failed", "message": str(e)}
        )