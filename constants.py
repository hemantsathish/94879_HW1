import os
import json

RAW_FEATURES = []
cur_dir = os.path.dirname(__file__)

with open(os.path.join(cur_dir, "artifacts/raw_columns.json")) as f:
    features = json.load(f)
    RAW_FEATURES = features["raw_features"]
