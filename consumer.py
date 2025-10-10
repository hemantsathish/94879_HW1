"""
Phase 1 Streaming: consume raw UCI air-quality records, clean/sanitize, make predictions,
and republish results to output topics.

Design notes:
* Clean and validate input data, converting UCI sentinel -200 to missing values
* Build features including lags for prediction
* Call FastAPI inference endpoint
* Output predictions with true values for monitoring
* Use batching for efficiency
"""

# Model Imports
import argparse, json, logging, os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from confluent_kafka import Consumer, Producer

# Add the predictive analytics path to import common
sys.path.append(
    str(Path(__file__).parent.parent.parent / "phase_3_predictive_analytics")
)
from common import RollingState, parse_date_time, add_time_cols

# BASE COLUMNS
REQUIRED_BASES = [
    "CO(GT)",
    "PT08.S1(CO)",
    "NMHC(GT)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]

# Range sanity checks
# Photometric sensor channels use nonnegative unbounded ranges
RANGES = {
    "CO(GT)": (0.0, 50.0),
    "C6H6(GT)": (0.0, 200.0),
    "NOx(GT)": (0.0, 1000.0),
    "NO2(GT)": (0.0, 400.0),
    "T": (-40.0, 60.0),
    "RH": (0.0, 100.0),
    "AH": (0.0, 50.0),
    "PT08.S1(CO)": (0.0, float("inf")),
    "PT08.S2(NMHC)": (0.0, float("inf")),
    "PT08.S3(NOx)": (0.0, float("inf")),
    "PT08.S4(NO2)": (0.0, float("inf")),
    "PT08.S5(O3)": (0.0, float("inf")),
}


# Convert raw field to float, mapping known sentinels and NaNs to None.
def _coerce_float(v):
    try:
        if v is None:
            return None
        if isinstance(v, float) and pd.isna(v):
            return None
        # Treat UCI sentinel
        if v == -200 or v == "-200":
            return None
        return float(v)
    except Exception:
        return None


# Return True if value falls within the range bounds for this field
def _within_range(name, value):
    if value is None:
        return True
    lo, hi = RANGES.get(name, (-float("inf"), float("inf")))
    return lo <= value <= hi


# Parse timestamp into UTC ISO-8601 'Z' form.
def _to_iso_ts(v):
    try:
        ts = pd.to_datetime(v, errors="coerce", utc=True)
        if ts is pd.NaT:
            return None
        return ts.isoformat().replace("+00:00", "Z")
    except Exception:
        return None


# Normalize and clean raw events
def clean_record(rec):
    ts_raw = rec.get("ts") or rec.get("event_time") or rec.get("DateTime")
    out = {
        "ts": _to_iso_ts(ts_raw),
        "site_id": rec.get("site_id", "station_1"),
    }
    missing = 0
    oor = 0

    for col in REQUIRED_BASES:
        v = _coerce_float(rec.get(col))
        if v is None:
            missing += 1
        elif not _within_range(col, v):
            v = None
            oor += 1
        out[col] = v

    # QA fields
    out["qa_missing_fields"] = missing
    out["qa_out_of_range_fields"] = oor
    return out


# Append cleaned data to daily parquet files partitioned by UTC date.
def write_parquet(df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)

    dated = df.dropna(subset=["ts"]).copy()
    undated = df[df["ts"].isna()].copy()

    if not dated.empty:
        dated["date"] = dated["ts"].dt.date
        for day, g in dated.groupby("date"):
            path = out_dir / f"air_quality_clean_{day}.parquet"
            if path.exists():
                prev = pd.read_parquet(path)
                pd.concat(
                    [prev, g.drop(columns=["date"])], ignore_index=True
                ).to_parquet(path, index=False)
            else:
                g.drop(columns=["date"]).to_parquet(path, index=False)
            logging.info("Parquet write: %d rows -> %s", len(g), os.fspath(path))

    if not undated.empty:
        path = out_dir / "air_quality_clean_undated.parquet"
        if path.exists():
            prev = pd.read_parquet(path)
            pd.concat([prev, undated], ignore_index=True).to_parquet(path, index=False)
        else:
            undated.to_parquet(path, index=False)
        logging.warning(
            "Parquet write (undated): %d rows -> %s", len(undated), os.fspath(path)
        )


# Build out the consumer
def make_consumer(bootstrap: str, group_id: str, topic: str) -> Consumer:
    c = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": group_id,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "session.timeout.ms": 10000,
        }
    )
    c.subscribe([topic])
    return c


# Build out the producer
def make_producer(bootstrap):
    return Producer(
        {
            "bootstrap.servers": bootstrap,
            "compression.type": "gzip",
            "enable.idempotence": False,
            "acks": "all",
            "retries": 5,
            "queue.buffering.max.ms": 50,  # Reduce buffering time
            "request.timeout.ms": 5000,  # Increase timeout
            "message.send.max.retries": 3,  # Set max retries
        }
    )


def build_row_features(record: dict, rolling_state: RollingState) -> dict:
    """Build features for a single row including time features and lags."""
    # Parse timestamp
    dt = parse_date_time(record)
    if pd.isna(dt):
        return None

    # Get target value
    try:
        co_gt = float(record.get("CO(GT)"))
        if pd.isna(co_gt) or co_gt == -200:
            return None
    except (TypeError, ValueError):
        return None

    # Update rolling state and get lag features
    site_id = record.get("site_id", "station_1")
    rolling_state.push(site_id, co_gt)

    # Build feature dict matching API's expected format
    features = {
        # Sensor readings
        "PT08.S1(CO)": None,
        "NMHC(GT)": None,
        "C6H6(GT)": None,
        "PT08.S2(NMHC)": None,
        "NOx(GT)": None,
        "PT08.S3(NOx)": None,
        "NO2(GT)": None,
        "PT08.S4(NO2)": None,
        "PT08.S5(O3)": None,
        "T": None,
        "RH": None,
        "AH": None,
    }

    # Fill in sensor readings
    for col in features.keys():
        try:
            val = float(record.get(col, np.nan))
            if val == -200 or pd.isna(val):
                val = None
            features[col] = val
        except (TypeError, ValueError):
            pass  # Keep as None

    # Add time-based features
    features.update(
        {
            "hour": dt.hour,
            "day_of_week": dt.weekday(),
            "month": dt.month,
            "hour_sin": float(np.sin(2 * np.pi * dt.hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * dt.hour / 24)),
            "dow_sin": float(np.sin(2 * np.pi * dt.weekday() / 7)),
            "dow_cos": float(np.cos(2 * np.pi * dt.weekday() / 7)),
            "month_sin": float(np.sin(2 * np.pi * dt.month / 12)),
            "month_cos": float(np.cos(2 * np.pi * dt.month / 12)),
        }
    )

    # Get lag features
    lag_features = rolling_state.make_features(site_id)

    # Map lag features to expected names and convert to Python types
    lag_mappings = {
        "co_gt_lag_1": "CO(GT)_lag_1",
        "co_gt_lag_3": "CO(GT)_lag_3",
        "co_gt_lag_6": "CO(GT)_lag_6",
        "co_gt_lag_12": "CO(GT)_lag_12",
        "co_gt_lag_24": "CO(GT)_lag_24",
        "co_gt_rolling_mean_3": "CO(GT)_rolling_mean_3",
        "co_gt_rolling_std_3": "CO(GT)_rolling_std_3",
        "co_gt_rolling_min_3": "CO(GT)_rolling_min_3",
        "co_gt_rolling_max_3": "CO(GT)_rolling_max_3",
        "co_gt_rolling_mean_6": "CO(GT)_rolling_mean_6",
        "co_gt_rolling_std_6": "CO(GT)_rolling_std_6",
        "co_gt_rolling_min_6": "CO(GT)_rolling_min_6",
        "co_gt_rolling_max_6": "CO(GT)_rolling_max_6",
        "co_gt_rolling_mean_12": "CO(GT)_rolling_mean_12",
        "co_gt_rolling_std_12": "CO(GT)_rolling_std_12",
        "co_gt_rolling_min_12": "CO(GT)_rolling_min_12",
        "co_gt_rolling_max_12": "CO(GT)_rolling_max_12",
        "co_gt_rolling_mean_24": "CO(GT)_rolling_mean_24",
        "co_gt_rolling_std_24": "CO(GT)_rolling_std_24",
        "co_gt_rolling_min_24": "CO(GT)_rolling_min_24",
        "co_gt_rolling_max_24": "CO(GT)_rolling_max_24",
    }

    # Add lag features with proper names
    for old_name, new_name in lag_mappings.items():
        val = lag_features.get(old_name)
        features[new_name] = float(val) if not pd.isna(val) else None

    # Add metadata (not part of features sent to API but needed for results)
    metadata = {"timestamp": dt.isoformat(), "site_id": site_id, "CO(GT)": co_gt}

    return {"features": features, "metadata": metadata}


def call_inference_api(features: dict, api_url: str) -> float:
    """Make prediction call to FastAPI endpoint."""
    try:
        # Debug print
        logging.info(f"Sending features to API: {json.dumps(features, indent=2)}")

        response = requests.post(
            f"{api_url}/predict", json={"features": features}, timeout=5
        )
        if not response.ok:
            logging.error(f"API Error Response: {response.text}")
        response.raise_for_status()
        result = response.json()
        return result["prediction"]
    except Exception as e:
        logging.error(f"API call failed: {str(e)}")
        return None


def process_batch(
    batch, out_topic, p, out_dir, missing_warn_threshold, rolling_state, api_url
):
    """Process a batch - clean data, make predictions, write results."""
    results = []
    cleaned = []
    missing_sum = 0
    fields = 0

    for m in batch:
        if m.error():
            logging.error("Consume error: %s", m.error())
            continue

        try:
            record = json.loads(m.value().decode("utf-8"))
        except json.JSONDecodeError:
            continue

        # Clean record first
        cr = clean_record(record)
        cleaned.append(cr)
        missing_sum += cr.get("qa_missing_fields", 0)
        fields += 4

        # Build features and get prediction
        result = build_row_features(cr, rolling_state)
        if result is None:
            continue

        features = result["features"]  # Features for API
        metadata = result["metadata"]  # Metadata for results

        prediction = call_inference_api(features, api_url)
        if prediction is None:
            continue

        # Prepare simplified output record
        output = {
            "XGBOOST_Prediction": prediction,
            "Timestamp": metadata["timestamp"],
            "True_Value": metadata["CO(GT)"],
        }

        results.append(output)

    if not cleaned:
        return

    # QA metrics for cleaning
    ratio = (missing_sum / fields) if fields else 0.0
    if ratio > missing_warn_threshold:
        logging.warning("High missing ratio: %.1f%%", 100 * ratio)

    # Write cleaned data to parquet
    df_cleaned = pd.DataFrame(cleaned)
    write_parquet(df_cleaned, out_dir)

    # Publish results with predictions
    for row in results:
        # Get site_id from cleaned record
        key = cleaned[-1]["site_id"].encode()
        p.produce(out_topic, key=key, value=json.dumps(row).encode())
    p.flush()

    # Log prediction metrics
    if results:
        df = pd.DataFrame(results)
        mae = np.mean(np.abs(df["True_Value"] - df["XGBOOST_Prediction"]))
        logging.info(f"Batch MAE: {mae:.3f} ({len(results)} predictions)")


def main():
    ap = argparse.ArgumentParser("Air Quality Consumer")
    ap.add_argument("--in-topic", default="air_quality.raw")
    ap.add_argument("--out-topic", default="air_quality.pred")
    ap.add_argument("--bootstrap", default="127.0.0.1:9092")
    ap.add_argument("--group-id", default="air-quality-consumers")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./phase_1_streaming_infrastructure/data/silver"),
    )
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--missing-warn-threshold", type=float, default=0.20)
    ap.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="FastAPI inference service URL",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    c = make_consumer(args.bootstrap, args.group_id, args.in_topic)
    p = make_producer(args.bootstrap)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rolling_state = RollingState(maxlen=48)  # Keep 48 hours of history

    buf = []
    try:
        while True:
            msg = c.poll(1.0)
            if msg is None:
                if buf:
                    process_batch(
                        buf,
                        args.out_topic,
                        p,
                        args.out_dir,
                        args.missing_warn_threshold,
                        rolling_state,
                        args.api_url,
                    )
                    c.commit()
                    buf = []
                continue
            buf.append(msg)
            if len(buf) >= args.batch_size:
                process_batch(
                    buf,
                    args.out_topic,
                    p,
                    args.out_dir,
                    args.missing_warn_threshold,
                    rolling_state,
                    args.api_url,
                )
                c.commit()
                buf = []
    except KeyboardInterrupt:
        logging.info("Stopping...")
    finally:
        if buf:
            try:
                process_batch(
                    buf,
                    args.out_topic,
                    p,
                    args.out_dir,
                    args.missing_warn_threshold,
                    rolling_state,
                    args.api_url,
                )
                c.commit()
            except Exception as e:
                logging.error(f"Error in final batch: {e}")
        c.close()


if __name__ == "__main__":
    main()
