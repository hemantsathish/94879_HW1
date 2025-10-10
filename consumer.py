import argparse
import json
import logging
import pandas as pd
import requests
from confluent_kafka import Consumer, Producer
from constants import RAW_FEATURES
from feature_buffer import FeatureBuffer


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
    out = {}

    for col in rec.keys():
        if col == "DateTime":
            out["ts"] = _to_iso_ts(rec.get("DateTime"))
        else:
            out[col] = _coerce_float(rec.get(col))

    return out


# Build out the consumer with auto-commit enabled
def make_consumer(bootstrap: str, group_id: str, topic: str) -> Consumer:
    c = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": group_id,
            "enable.auto.commit": True,
            "auto.commit.interval.ms": 5000,
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
            "queue.buffering.max.ms": 50,
            "request.timeout.ms": 5000,
            "message.send.max.retries": 3,
        }
    )


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


def main():
    ap = argparse.ArgumentParser("Air Quality Consumer")
    ap.add_argument("--in-topic", default="air_quality.raw")
    ap.add_argument("--out-topic", default="air_quality.pred")
    ap.add_argument("--bootstrap", default="127.0.0.1:9092")
    ap.add_argument("--group-id", default="air-quality-consumers")
    ap.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="FastAPI inference service URL",
    )
    ap.add_argument(
        "--features-json",
        default="./artifacts/features.json",
        help="Path to features.json file",
    )

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    # Initialize Kafka consumer and producer
    c = make_consumer(args.bootstrap, args.group_id, args.in_topic)
    p = make_producer(args.bootstrap)

    # Initialize feature buffer
    feature_buffer = FeatureBuffer(maxlen=168, features_json_path=args.features_json)
    logging.info("Feature buffer initialized with maxlen=168")

    # Track statistics
    records_processed = 0
    predictions_made = 0

    try:
        logging.info(f"Starting consumer on topic '{args.in_topic}'")
        logging.info("Warming up buffer (need 168 rows before predictions start)...")

        while True:
            msg = c.poll(timeout=1.0)

            if msg is None:
                continue

            if msg.error():
                logging.error(f"Consumer error: {msg.error()}")
                continue

            try:
                # Decode JSON message
                record = json.loads(msg.value().decode("utf-8"))
                records_processed += 1

                # Clean the record
                cleaned = clean_record(record)

                # Extract CO(GT) target value
                co_gt = cleaned.get("CO(GT)")
                if co_gt is None or pd.isna(co_gt):
                    logging.warning(
                        f"Record {records_processed}: Missing CO(GT) value, skipping"
                    )
                    continue

                # Prepare record for buffer
                buffer_record = {
                    "timestamp": cleaned.get("ts"),
                    "CO(GT)": float(co_gt),
                }

                # Add all sensor readings
                sensor_cols = [
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

                for col in sensor_cols:
                    val = cleaned.get(col)
                    buffer_record[col] = float(val) if val is not None else None

                # Push to buffer
                feature_buffer.push(buffer_record)

                buffer_len = feature_buffer.get_buffer_length()
                logging.info(f"Record {records_processed}: Buffer at {buffer_len}/168")

                # Check if we have enough data for prediction
                if not feature_buffer.has_minimum_rows(min_rows=168):
                    continue

                # Compute features
                features = feature_buffer.compute_features()
                if features is None:
                    logging.warning(
                        f"Record {records_processed}: Could not compute features"
                    )
                    continue

                # Call API for prediction
                prediction = call_inference_api(features, args.api_url)
                if prediction is None:
                    logging.error(
                        f"Record {records_processed}: Prediction failed, skipping"
                    )
                    continue

                predictions_made += 1

                # Get current record info
                current = feature_buffer.get_current_record()
                timestamp = current["timestamp"]
                true_value = current["CO(GT)"]

                # Prepare output record
                output = {
                    "XGBOOST_Prediction": prediction,
                    "Timestamp": timestamp,
                    "True_Value": true_value,
                }

                # Produce to output topic
                key = b"predictions"
                p.produce(args.out_topic, key=key, value=json.dumps(output).encode())
                p.flush()

                # Calculate and log error
                error = abs(true_value - prediction)
                logging.info(
                    f"Prediction {predictions_made}: True={true_value:.3f}, "
                    f"Pred={prediction:.3f}, Error={error:.3f}"
                )

            except json.JSONDecodeError:
                logging.error(f"Record {records_processed}: Invalid JSON, skipping")
                continue
            except Exception as e:
                logging.error(
                    f"Record {records_processed}: Unexpected error: {str(e)}",
                    exc_info=True,
                )
                continue

    except KeyboardInterrupt:
        logging.info("Stopping consumer...")
    finally:
        logging.info(
            f"Consumer stopped. Processed {records_processed} records, "
            f"made {predictions_made} predictions"
        )
        c.close()


if __name__ == "__main__":
    main()
