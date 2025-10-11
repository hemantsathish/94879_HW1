"""
Generate reference dataset for Evidently monitoring.

Loads evaluation dataset, runs model predictions, and saves as reference.csv
for use in production monitoring.
"""

import os
import json
import pandas as pd
import joblib

# Paths
EVAL_DATA_PATH = "dataset/eval_data_engineered.csv"
MODEL_PATH = "artifacts/model.pkl"
FEATURES_JSON_PATH = "artifacts/features.json"
OUTPUT_PATH = "dataset/reference.csv"


def main():
    print("=" * 60)
    print("Generating Reference Dataset for Monitoring")
    print("=" * 60)

    # Verify files exist
    assert os.path.exists(EVAL_DATA_PATH), f"❌ File not found: {EVAL_DATA_PATH}"
    assert os.path.exists(MODEL_PATH), f"❌ File not found: {MODEL_PATH}"
    assert os.path.exists(
        FEATURES_JSON_PATH
    ), f"❌ File not found: {FEATURES_JSON_PATH}"

    # Load evaluation data
    print(f"\n📂 Loading evaluation data from {EVAL_DATA_PATH}...")
    eval_data = pd.read_csv(EVAL_DATA_PATH)
    print(f"   ✅ Loaded {len(eval_data)} rows, {len(eval_data.columns)} columns")

    # Load feature columns from JSON
    print(f"\n📂 Loading feature configuration from {FEATURES_JSON_PATH}...")
    with open(FEATURES_JSON_PATH, "r") as f:
        feature_config = json.load(f)
        feature_cols = feature_config["features"]
        target_col = feature_config["target"]
    print(f"   ✅ Loaded {len(feature_cols)} features")
    print(f"   ✅ Target column: {target_col}")

    # Verify all features exist in eval data
    missing_features = [col for col in feature_cols if col not in eval_data.columns]
    if missing_features:
        print(f"\n❌ Error: Missing features in eval data: {missing_features[:5]}...")
        return

    # Verify target exists
    if target_col not in eval_data.columns:
        print(f"\n❌ Error: Target column '{target_col}' not found in eval data")
        return

    # Extract features for prediction
    X_eval = eval_data[feature_cols]
    y_eval = eval_data[target_col]

    print(f"\n✅ Feature matrix shape: {X_eval.shape}")
    print(f"✅ Target shape: {y_eval.shape}")

    # Load trained model
    print(f"\n📂 Loading trained model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    print("   ✅ Model loaded successfully")
    print(f"   ✅ Model type: {type(model).__name__}")

    # Generate predictions
    print("\n🔮 Generating predictions on evaluation data...")
    predictions = model.predict(X_eval)
    print(f"   ✅ Generated {len(predictions)} predictions")

    # Calculate evaluation metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    mae = mean_absolute_error(y_eval, predictions)
    rmse = np.sqrt(mean_squared_error(y_eval, predictions))
    r2 = r2_score(y_eval, predictions)

    print("\n📊 Evaluation Metrics:")
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²:   {r2:.4f}")

    # Add predictions to eval data
    print("\n💾 Adding predictions to dataset...")
    eval_data["prediction"] = predictions

    # Verify final dataset structure
    print("\n✅ Final dataset structure:")
    print(f"   Total columns: {len(eval_data.columns)}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Target: {target_col}")
    print("   Prediction: prediction")

    # Check for any NaN values
    nan_count = eval_data.isna().sum().sum()
    if nan_count > 0:
        print(f"\n⚠️  Warning: Dataset contains {nan_count} NaN values")
        print("   NaN counts by column:")
        nan_cols = eval_data.isna().sum()
        for col, count in nan_cols[nan_cols > 0].items():
            print(f"      {col}: {count}")
    else:
        print("\n✅ No NaN values detected")

    # Save reference dataset
    print(f"\n💾 Saving reference dataset to {OUTPUT_PATH}...")
    eval_data.to_csv(OUTPUT_PATH, index=False)
    print("   ✅ Saved successfully!")

    # Summary
    print("\n" + "=" * 60)
    print("Reference Dataset Generation Complete!")
    print("=" * 60)
    print(f"📁 Output file: {OUTPUT_PATH}")
    print(f"📊 Rows: {len(eval_data)}")
    print(f"📊 Columns: {len(eval_data.columns)}")
    print("🎯 Ready for use in MonitoringService!")
    print("=" * 60)

    # Display sample
    print("\n📋 Sample of reference dataset (first 3 rows):")
    print(eval_data[["prediction", target_col] + feature_cols[:5]].head(3))


if __name__ == "__main__":
    main()
