# phase_3/train.py
import json, argparse, joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from common import add_time_cols
from statsmodels.tsa.statespace.sarimax import SARIMAX

REQUIRED_BASES = [
    "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)",
    "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)",
    "T", "RH", "AH",
]
TIME_COLS = ["Hour", "Day", "Month", "Hour_sin", "Hour_cos", "Season"]

def _is_lag_or_roll(col, target):
    return col.startswith(f"{target}_lag") or col.startswith(f"{target}_rollmean") or col.startswith(f"{target}_rollstd")

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';', decimal=',', engine='python')

    remove = [c for c in df.columns if str(c).startswith("Unnamed")]
    if remove:
        df = df.drop(columns=remove)

    time_norm = df["Time"].astype(str).str.replace(".", ":", regex=False)
    dt = pd.to_datetime(df["Date"] + " " + time_norm, dayfirst=True, errors="coerce")
    df = df[~dt.isna()].copy()
    df["DateTime"] = dt
    df = df.set_index("DateTime").sort_index()

    # Coerce numerics & convert -200 sentinel to NaN
    for c in df.columns:
        if c not in ("Date", "Time"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] == -200, c] = np.nan
    return df

# Add Hour/Day/Month + cyclical
def build_features(df: pd.DataFrame, target_col="CO(GT)"):
    if target_col not in df.columns:
        raise KeyError(f"Target '{target_col}' not found. Available columns include: {list(df.columns)[:12]}...")

    df = df.copy()
    df = add_time_cols(df)

    # Lags / rolling stats
    for L in (1, 2, 3, 6, 12, 24):
        df[f"{target_col}_lag{L}"] = df[target_col].shift(L)
    for W in (3, 6, 12, 24):
        df[f"{target_col}_rollmean{W}"] = df[target_col].rolling(W).mean()
        df[f"{target_col}_rollstd{W}"]  = df[target_col].rolling(W).std()

    req = [target_col] \
        + [f"{target_col}_lag{L}" for L in (1,2,3,6,12,24)] \
        + [f"{target_col}_rollmean{W}" for W in (3,6,12,24)] \
        + [f"{target_col}_rollstd{W}"  for W in (3,6,12,24)]
    df = df.dropna(subset=req).copy()

    # Keep numeric columns
    numeric = df.select_dtypes(include=[np.number]).copy()
    idx_used = numeric.index

    # Build y
    y = numeric[target_col].astype(float).values

    # Base sensors we know Phase 1 outputs + engineered time + our lags/rolls
    allowed = set(REQUIRED_BASES) | set(TIME_COLS) | {
        c for c in numeric.columns if _is_lag_or_roll(c, target_col)
    }
    # Drop the target from features
    allowed.discard(target_col)

    # Filter out "Unnamed: *" and any columns Phase 1 doesn’t send
    keep_cols = [c for c in numeric.columns if c in allowed]
    X = numeric[keep_cols].copy()

    feature_cols = list(X.columns)
    return X.values, y, feature_cols, idx_used

def chrono_split(X, y, frac=0.85):
    n = len(y)
    cut = int(n * frac)
    return (X[:cut], y[:cut], X[cut:], y[cut:], cut)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",type=Path,default=Path("phase_1_streaming_infrastructure/data/splits/AirQualityUCI_stream.csv"), help="Path to AirQualityUCI_stream.csv")
    ap.add_argument("--outdir", default="artifacts", help="Where to save artifacts")
    ap.add_argument("--train_frac", type=float, default=0.85)
    ap.add_argument("--trees", type=int, default=400)
    ap.add_argument("--target", default="CO(GT)", help="Target column (e.g., 'CO(GT)')")
    args = ap.parse_args()

    df = load_data(args.csv)
    X, y, feats, idx_used = build_features(df, target_col=args.target)
    Xtr, ytr, Xte, yte, cut = chrono_split(X, y, frac=args.train_frac)

    # Naive baseline (prev value)
    # Predict y[t-1] for each test point; first test point uses last train value
    y_true = y[cut:]
    naive_pred = np.empty_like(y_true)
    naive_pred[0] = y[cut-1]
    if len(y_true) > 1:
        naive_pred[1:] = y_true[:-1]

    naive_mae = float(mean_absolute_error(y_true, naive_pred))
    naive_rmse = float(np.sqrt(mean_squared_error(y_true, naive_pred)))
    print(f"[NaivePrev] Holdout MAE={naive_mae:.3f}  RMSE={naive_rmse:.3f}")

    # Random Forest 
    rf = RandomForestRegressor(
        n_estimators=args.trees,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2
    )
    rf.fit(Xtr, ytr)
    rf_pred = rf.predict(Xte)

    rf_mae = float(mean_absolute_error(yte, rf_pred))
    rf_rmse = float(np.sqrt(mean_squared_error(yte, rf_pred)))
    print(f"[RF] Holdout MAE={rf_mae:.3f}  RMSE={rf_rmse:.3f}  n_test={len(yte)}")

    metrics = {
        "RandomForest": {
            "n_estimators": args.trees,
            "train_frac": args.train_frac,
            "MAE": rf_mae,
            "RMSE": rf_rmse,
            "n_test": int(len(yte)),
        },
        "NaivePrev": {
            "MAE": naive_mae,
            "RMSE": naive_rmse,
            "n_test": int(len(y_true)),
        },
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save RF model
    joblib.dump(rf, outdir / "model.pkl")

    # Save feature medians for inference-time imputation
    feat_medians = {}
    for i, col in enumerate(feats):
        col_vals = Xtr[:, i]
        m = float(np.nanmedian(col_vals)) if np.isfinite(col_vals).any() else 0.0
        feat_medians[col] = m

    with open(outdir / "features.json", "w") as f:
        json.dump(
            {
                "feature_order": feats,
                "target": args.target,
                "medians": feat_medians
            },
            f,
            indent=2
        )

    # SARIMA   
    # Fit SARIMA on the same aligned series and evaluate on holdout
    try:
        series = pd.Series(y, index=idx_used).sort_index().asfreq("H")
        series = series.ffill().bfill()

        train = series.iloc[:cut]
        test  = series.iloc[cut:]

        mdl = SARIMAX(
            train,
            order=(2,1,2),
            seasonal_order=(1,0,1,24),
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = mdl.fit(disp=False)

        sar_pred = res.get_forecast(steps=len(test)).predicted_mean
        sar_pred.index = test.index

        sar_mae = float(mean_absolute_error(test, sar_pred))
        sar_rmse = float(np.sqrt(mean_squared_error(test, sar_pred)))
        print(f"[SARIMA] Holdout MAE={sar_mae:.3f}  RMSE={sar_rmse:.3f}")

        joblib.dump(res, outdir / "sarima.pkl")
        metrics["SARIMA"] = {"order":[2,1,2],"seasonal_order":[1,0,1,24],
                     "MAE": sar_mae, "RMSE": sar_rmse, "n_test": int(len(test))}
        
    except Exception as e:
        print(f"[INFO] SARIMA not saved/failed ({e})")

    # Save metrics
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()