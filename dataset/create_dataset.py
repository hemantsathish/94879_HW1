import os
import json
import pandas as pd
import numpy as np

DATA_PATH = "AirQualityUCI.csv"
assert os.path.exists(DATA_PATH), f"❌ File not found: {DATA_PATH}"

# ============================================================
# LOAD + CLEAN
# ============================================================
df = pd.read_csv(DATA_PATH, sep=";", decimal=",", na_values=[-200, -200.0])
df = df.dropna(axis=1, how="all")
df["DateTime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"], format="%d/%m/%Y %H.%M.%S", errors="coerce"
)
df = (
    df.drop(columns=["Date", "Time"])
    .dropna(subset=["DateTime"])
    .sort_values("DateTime")
)
df = df.set_index("DateTime")
for c in df.select_dtypes(np.number).columns:
    df[c] = df[c].interpolate(method="time", limit_direction="both")
df = df.reset_index().dropna()

# ============================================================
# TEMPORAL SPLIT (70/15/15) - BEFORE FEATURE ENGINEERING
# ============================================================
target_col = "CO(GT)"
n = len(df)
t1, t2 = int(0.7 * n), int(0.85 * n)

# Save raw test data for streaming
raw_test = df.iloc[t2:].copy()
raw_test.to_csv("test_data_raw.csv", index=False)
raw_cols = [col for col in raw_test.columns if col != target_col]
print(f"Saved test_data_raw.csv: {len(raw_test)} rows")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
df = df.sort_values("DateTime").set_index("DateTime")

# Lag features
lag_periods = [1, 2, 3, 6, 12, 24, 48, 72]
for lag in lag_periods:
    df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

# Rolling statistics
windows = [3, 6, 12, 24, 48, 168]
for w in windows:
    roll = df[target_col].shift(1).rolling(window=w, min_periods=1)
    df[f"{target_col}_rolling_mean_{w}"] = roll.mean()
    df[f"{target_col}_rolling_std_{w}"] = roll.std()
    df[f"{target_col}_rolling_min_{w}"] = roll.min()
    df[f"{target_col}_rolling_max_{w}"] = roll.max()

# Rate of change features
pollutants = [
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "NOx(GT)",
    "NO2(GT)",
]
for col in pollutants:
    if col in df.columns:
        df[f"{col}_diff_1h"] = df[col].diff(1)
        df[f"{col}_diff_3h"] = df[col].diff(3)
        df[f"{col}_diff_24h"] = df[col].diff(24)

# Pollutant interaction features
for i, p1 in enumerate(pollutants):
    if p1 in df.columns:
        for p2 in pollutants[i + 1 :]:
            if p2 in df.columns:
                df[f"{p1}_x_{p2}"] = df[p1] * df[p2]
                df[f"{p1}_ratio_{p2}"] = df[p1] / (df[p2] + 1e-8)

# Environmental interactions
if "T" in df.columns and "AH" in df.columns:
    df["temp_humidity"] = df["T"] * df["AH"]
    df["temp_sq"] = df["T"] ** 2
    df["humidity_sq"] = df["AH"] ** 2

if "T" in df.columns and "RH" in df.columns:
    df["temp_rh"] = df["T"] * df["RH"]

df = df.reset_index()

# Time-based features
df["hour"] = df["DateTime"].dt.hour
df["day_of_week"] = df["DateTime"].dt.dayofweek
df["month"] = df["DateTime"].dt.month
df["day_of_month"] = df["DateTime"].dt.day
df["week_of_year"] = df["DateTime"].dt.isocalendar().week

# Cyclical encoding
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

# Categorical time features
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
df["is_night"] = df["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
df["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)
df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)

# Drop rows with NaN from feature engineering
df = df.dropna()

# ============================================================
# PREPARE ENGINEERED DATASETS
# ============================================================
exclude = ["DateTime", target_col]
X = df.drop(columns=exclude)
y = df[target_col]
X = X.select_dtypes(include=[np.number]).copy()
feature_cols = X.columns.tolist()

print(f"Total engineered features: {len(feature_cols)}")

# Split engineered data using temporal boundaries
datetime_index = df["DateTime"]
t1_idx = datetime_index.searchsorted(df.iloc[t1]["DateTime"])
t2_idx = datetime_index.searchsorted(df.iloc[t2]["DateTime"])

# Extract validation set (for Evidently reference)
X_eval = X.iloc[t1_idx:t2_idx]
y_eval = y.iloc[t1_idx:t2_idx]

# Save evaluation dataset with all engineered features + target
eval_data = X_eval.copy()
eval_data["CO(GT)"] = y_eval.values
eval_data.to_csv("eval_data_engineered.csv", index=False)
print(f"✅ Saved eval_data_engineered.csv: {len(eval_data)} rows (features + target)")

# Save feature columns as JSON
with open("../artifacts/features.json", "w") as f:
    json.dump({"features": feature_cols, "target": target_col}, f, indent=2)
print(f"✅ Saved features.json: {len(feature_cols)} features")

# Save raw columns as JSON (what should be streamed)
with open("../artifacts/raw_columns.json", "w") as f:
    json.dump({"raw_features": raw_cols, "target": target_col}, f, indent=2)
print(f"✅ Saved raw_columns.json: {len(raw_cols)} raw features")

print("\n" + "=" * 60)
print("Dataset Generation Complete!")
print("=" * 60)
print(f"Evaluation dataset: {len(eval_data)} rows")
print(f"Test dataset (raw): {len(raw_test)} rows")
print(f"Total features: {len(feature_cols)}")
