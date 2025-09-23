"""
Phase 1 Producer: replays historical UCI Air Quality CSV as if it were real-time sensor data, streaming into the raw Kafka topic.

Design notes:
- Raw events are preserved "as-is" (including UCI -200 sentinel values).
- Timing between rows is scaled by `--speedup` so hours/days can be replayed in minutes.
- Messages are keyed by (site_id, event_time) for deterministic partitioning.
"""

# Model Imports
import argparse, json, logging, time
from pathlib import Path
import pandas as pd
from confluent_kafka import Producer

# Build out the producer
def build_producer(bootstrap: str) -> Producer:
    return Producer({
        "bootstrap.servers": bootstrap,
        "compression.type": "gzip",
        "enable.idempotence": False,
        "acks": "all",
        "retries": 5,
        "queue.buffering.max.ms": 50,
        "message.max.bytes": 1_048_576,
    })

# Stream rows from the raw UCI CSV
def load_rows(csv_path: Path):
    df = pd.read_csv(csv_path, sep=";", decimal=",")
    time_norm = df["Time"].astype(str).str.replace(".", ":", regex=False)
    dt = pd.to_datetime(df["Date"] + " " + time_norm, dayfirst=True, errors="coerce")
    df = df[~dt.isna()].copy()
    df["event_time"] = dt.dt.tz_localize("UTC")
    # Drop trailing unnamed columns if present
    junk = [c for c in df.columns if c.startswith("Unnamed:")]
    if junk: df = df.drop(columns=junk)
    # Keep -200 markers
    for r in df.to_dict(orient="records"):
        yield r

def main():
    ap = argparse.ArgumentParser("Air Quality Producer")
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--topic", default="air_quality.raw")
    ap.add_argument("--bootstrap", default="127.0.0.1:9092")
    ap.add_argument("--site-id", default="station_1")
    ap.add_argument("--speedup", type=float, default=120.0, help="historical seconds per 1 real second")
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = build_producer(args.bootstrap)

    prev_ts = None
    sent = 0
    try:
        while True:
            for rec in load_rows(args.csv):
                t = rec["event_time"]
                payload = dict(rec)
                payload["event_time"] = t.isoformat()
                payload["site_id"] = args.site_id

                # Pace replay roughly according to historical deltas, but scale down by --speedup so days of data fit in minutes
                if prev_ts is not None:
                    dt = (t - prev_ts).total_seconds()
                    time.sleep(min(max(dt/args.speedup, 0.0), 2.0))
                prev_ts = t

                key = f"{args.site_id}:{payload['event_time']}".encode()
                p.produce(args.topic, key=key, value=json.dumps(payload).encode())
                p.poll(0)
                sent += 1
                if sent % 500 == 0:
                    logging.info("Queued %d messages", sent)
            if not args.loop:
                break
    finally:
        # Ensure all buffered messages are delivered before exit
        p.flush(10)
        logging.info("Producer done. Sent %d", sent)

if __name__ == "__main__":
    main()