# # # from fastapi import FastAPI, HTTPException
# # # from pydantic import BaseModel
# # # from typing import Dict, Any
# # # import json
# # # import joblib
# # # import numpy as np
# # # from pathlib import Path


# # # class PredictRequest(BaseModel):
# # #     features: Dict[str, float]


# # # def load_artifacts(artifacts_dir: Path):
# # #     features_path = artifacts_dir / "features.json"
# # #     model_path = artifacts_dir / "model.pkl"

# # #     if not features_path.exists() or not model_path.exists():
# # #         raise FileNotFoundError("Required artifacts not found: features.json or model.pkl")

# # #     with open(features_path, "r") as f:
# # #         meta = json.load(f)
# # #     feature_order = meta["feature_order"]
# # #     model = joblib.load(model_path)
# # #     return feature_order, model


# # # ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
# # # FEATURE_ORDER, MODEL = load_artifacts(ARTIFACTS_DIR)

# # # app = FastAPI(title="Air Quality Prediction API", version="1.0.0")


# # # @app.get("/health")
# # # def health() -> Dict[str, Any]:
# # #     return {"status": "ok"}


# # # @app.post("/predict")
# # # def predict(req: PredictRequest) -> Dict[str, Any]:
# # #     # Validate feature presence
# # #     missing = [f for f in FEATURE_ORDER if f not in req.features]
# # #     if missing:
# # #         raise HTTPException(status_code=400, detail={"error": "missing_features", "missing": missing[:10]})

# # #     try:
# # #         ordered = np.array([float(req.features[f]) for f in FEATURE_ORDER], dtype=float).reshape(1, -1)
# # #     except Exception as e:
# # #         raise HTTPException(status_code=400, detail={"error": "invalid_feature_values", "message": str(e)})

# # #     try:
# # #         yhat = float(MODEL.predict(ordered)[0])
# # #     except Exception as e:
# # #         raise HTTPException(status_code=500, detail={"error": "prediction_failed", "message": str(e)})

# # #     return {"prediction": yhat}


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Union, Optional, List, Tuple
import os, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow


# ---------- Request schema ----------
class PredictRequest(BaseModel):
    features: Dict[str, Union[int, float]]


# ---------- Helpers ----------
def load_feature_order_fallback(artifacts_dir: Path) -> Optional[List[str]]:
    """Optional fallback if the MLflow model signature does not include column names."""
    fpath = artifacts_dir / "artifacts" / "features.json"
    if not fpath.exists():
        fpath = (Path(__file__).parent / "artifacts" / "features.json")
    if fpath.exists():
        with fpath.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        order = meta.get("feature_order")
        if isinstance(order, list) and order:
            return order
    return None


def get_expected_schema_from_signature(model) -> Tuple[Optional[List[str]], Dict[str, str]]:
    """
    Inspect MLflow model signature to derive (feature_order, dtypes).
    dtypes is a mapping: column -> 'int64' or 'float64'
    """
    feature_order = None
    dtypes: Dict[str, str] = {}
    try:
        sig = model.metadata.get_input_schema()
        if sig and getattr(sig, "inputs", None):
            feature_order = []
            for colspec in sig.inputs:
                name = colspec.name
                t_str = str(colspec.type)
                
                if "long" in t_str.lower() or "integer" in t_str.lower():
                    dtypes[name] = "int64"
                else:
                    dtypes[name] = "float64"
                    
                feature_order.append(name)
    except Exception as e:
        print(f"Warning: Failed to parse signature: {e}", file=sys.stderr)
    return feature_order, dtypes


def load_mlflow_model_from_env():
    """
    Requires env:
      MLFLOW_TRACKING_URI=databricks
      DATABRICKS_HOST, DATABRICKS_TOKEN
      MLFLOW_MODEL_URI=runs:/<run_id>/model
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "databricks"))

    uri = os.getenv("MLFLOW_MODEL_URI")
    run_id = os.getenv("MLFLOW_RUN_ID")
    if not uri and run_id:
        uri = f"runs:/{run_id}/model"
    if not uri:
        raise RuntimeError("Set MLFLOW_MODEL_URI or MLFLOW_RUN_ID")

    print("DEBUG MLFLOW_MODEL_URI =", uri, file=sys.stderr)
    model = mlflow.pyfunc.load_model(uri)

    sig_order, sig_dtypes = get_expected_schema_from_signature(model)

    fallback_order = load_feature_order_fallback(Path(__file__).parent)
    feature_order = sig_order or fallback_order
    if not feature_order:
        raise RuntimeError("Could not determine feature order from MLflow signature or artifacts/features.json")

    if not sig_dtypes:
        sig_dtypes = {c: "float64" for c in feature_order}
        for c in ("hour", "day_of_week", "month"):
            if c in sig_dtypes:
                sig_dtypes[c] = "int64"

    # Log a quick preview for sanity
    preview = {k: sig_dtypes[k] for k in feature_order[:min(5, len(feature_order))]}
    print("DEBUG Expected first dtypes:", preview, file=sys.stderr)
    print("DEBUG All dtypes:", sig_dtypes, file=sys.stderr)

    return model, feature_order, sig_dtypes


# ---------- Load model on startup ----------
MODEL, FEATURE_ORDER, EXPECTED_DTYPES = load_mlflow_model_from_env()

app = FastAPI(title="Air Quality Prediction API", version="1.0.0")


# ---------- Routes ----------
@app.get("/health")
def health() -> Dict[str, Any]:
    sample_dtypes = {k: EXPECTED_DTYPES[k] for k in FEATURE_ORDER[: min(5, len(FEATURE_ORDER))]}
    return {
        "status": "ok",
        "model_source": "databricks-mlflow",
        "n_features": len(FEATURE_ORDER),
        "first_features": FEATURE_ORDER[: min(5, len(FEATURE_ORDER))],
        "first_dtypes": sample_dtypes,
    }


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    # Validate presence-first
    missing = [f for f in FEATURE_ORDER if f not in req.features]
    if missing:
        raise HTTPException(status_code=400, detail={"error": "missing_features", "missing": missing[:15]})

    try:
        row_data = {f: req.features[f] for f in FEATURE_ORDER}
        
        X = pd.DataFrame([row_data], columns=FEATURE_ORDER)
        
        for col in FEATURE_ORDER:
            expected_dtype = EXPECTED_DTYPES[col]
            
            if expected_dtype == "int64":
                val = X[col].iloc[0]
                if pd.notna(val) and not np.isclose(val, np.round(val)):
                    raise ValueError(f"Column '{col}' expects integer, got {val}")
                X[col] = X[col].round().astype('int64')
            else:
                X[col] = X[col].astype('float64')

        # Final sanity: log dtypes
        print("DEBUG Input dtypes ->", X.dtypes.to_dict(), file=sys.stderr)
        print("DEBUG Sample values ->", X.iloc[0][['hour', 'day_of_week', 'month']].to_dict() if 'hour' in X.columns else "N/A", file=sys.stderr)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_feature_values", "message": str(e)},
        )

    # Predict
    try:
        yhat = MODEL.predict(X)
        val = float(np.array(yhat).ravel()[0])
        return {"prediction": val}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "prediction_failed", "message": str(e)})
    

@app.get("/debug/schema")
def debug_schema():
    sig = MODEL.metadata.get_input_schema()
    schema_info = []
    if sig and hasattr(sig, 'inputs'):
        for col in sig.inputs:
            schema_info.append({
                "name": col.name,
                "type": str(col.type),
                "type_string": col.type.to_string() if hasattr(col.type, 'to_string') else str(col.type)
            })
    return {"schema": schema_info, "expected_dtypes": EXPECTED_DTYPES}