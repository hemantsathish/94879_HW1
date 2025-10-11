# Model Imports
import argparse
import json
import logging
import time
from pathlib import Path
import pandas as pd
from confluent_kafka import Producer

STREAMING_TIMEOUT = 0.5


# Build out the producer
def build_producer(bootstrap: str) -> Producer:
    return Producer(
        {
            "bootstrap.servers": bootstrap,
            "compression.type": "gzip",
            "enable.idempotence": False,
            "acks": "all",
            "retries": 5,
            "queue.buffering.max.ms": 50,
            "message.max.bytes": 1_048_576,
        }
    )


# Stream rows from the raw UCI CSV
def load_rows(csv_path: Path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    dt = pd.to_datetime(df["DateTime"], dayfirst=True, errors="coerce")
    df["event_time"] = dt.dt.tz_localize("UTC")

    for r in df.to_dict(orient="records"):
        yield r


def main():
    ap = argparse.ArgumentParser("Air Quality Producer")
    ap.add_argument(
        "--csv", required=True, type=Path, default=Path("dataset/test_data_raw.csv")
    )
    ap.add_argument("--topic", default="air_quality.raw")
    ap.add_argument("--bootstrap", default="127.0.0.1:9092")
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = build_producer(args.bootstrap)

    sent = 0
    skipped = 0

    try:
        while True:
            for rec in load_rows(args.csv):
                t = rec["event_time"]

                # Skip rows with invalid timestamps
                if pd.isna(t):
                    skipped += 1
                    continue

                payload = dict(rec)
                payload["event_time"] = t.isoformat()

                time.sleep(STREAMING_TIMEOUT)

                key = b"air_quality"
                p.produce(args.topic, key=key, value=json.dumps(payload).encode())
                p.poll(0)
                sent += 1

                logging.info(f"Sent row {sent}")

            if not args.loop:
                break

    finally:
        # Ensure all buffered messages are delivered before exit
        p.flush(10)
        logging.info(
            f"Producer done. Sent {sent} messages, skipped {skipped} rows with invalid timestamps"
        )


if __name__ == "__main__":
    main()
