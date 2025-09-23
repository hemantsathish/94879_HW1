"""
Phase 1 Streaming: consume raw UCI air-quality records, clean/sanitize, persist daily parquet silver files, and republish to a cleaned Kafka topic.

Design notes:
* Treat UCI sentinel -200 as missing (None) and enforce light range QA.
* Partition parquet by calendar day of the UTC timestamp for easy downstream reads.
* Use consumer-side batching to amortize IO and commit offsets after durable work.
* Publish cleaned events keyed by site_id to shard evenly across partitions.
"""

# Model Imports
import argparse, json, logging, os
from pathlib import Path
import pandas as pd
from confluent_kafka import Consumer, Producer

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
                pd.concat([prev, g.drop(columns=["date"])], ignore_index=True).to_parquet(path, index=False)
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
        logging.warning("Parquet write (undated): %d rows -> %s", len(undated), os.fspath(path))

# Build out the consumer
def make_consumer(bootstrap: str, group_id: str, topic: str) -> Consumer:
    c = Consumer({
        "bootstrap.servers": bootstrap,
        "group.id": group_id,
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        "session.timeout.ms": 10000,
    })
    c.subscribe([topic])
    return c

# Build out the producer
def make_producer(bootstrap):
    return Producer({
        "bootstrap.servers": bootstrap,
        "compression.type": "gzip",
        "enable.idempotence": False,
        "acks": "all",
        "retries": 5,
    })

# Handle a batch - clean, write and publish cleaned events
def process_batch(batch, out_topic, p, out_dir, missing_warn_threshold):
    cleaned = []
    missing_sum = 0
    fields = 0
    for m in batch:
        if m.error():
            logging.error("Consume error: %s", m.error())
            continue
        rec = json.loads(m.value().decode("utf-8"))
        cr = clean_record(rec)
        cleaned.append(cr)
        missing_sum += cr.get("qa_missing_fields", 0)
        fields += 4

    if not cleaned:
        return

    # QA metric
    ratio = (missing_sum / fields) if fields else 0.0
    if ratio > missing_warn_threshold:
        logging.warning("High missing ratio: %.1f%%", 100 * ratio)

    df = pd.DataFrame(cleaned)
    write_parquet(df, out_dir)

    # publish cleaned
    for row in cleaned:
        key = (row.get('site_id', 'station_1')).encode('utf-8')
        p.produce(out_topic, key=key, value=json.dumps(row).encode("utf-8"))
    p.flush()

def main():
    ap = argparse.ArgumentParser("Air Quality Consumer")
    ap.add_argument("--in-topic", default="air_quality.raw")
    ap.add_argument("--out-topic", default="air_quality.clean")
    ap.add_argument("--bootstrap", default="127.0.0.1:9092")
    ap.add_argument("--group-id", default="air-quality-consumers")
    ap.add_argument("--out-dir", type=Path, default=Path("./phase_1_streaming_infrastructure/data/silver"))
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--missing-warn-threshold", type=float, default=0.20)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    c = make_consumer(args.bootstrap, args.group_id, args.in_topic)
    p = make_producer(args.bootstrap)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    buf = []
    try:
        while True:
            msg = c.poll(1.0)
            if msg is None:
                if buf:
                    process_batch(buf, args.out_topic, p, args.out_dir, args.missing_warn_threshold)
                    c.commit()
                    buf = []
                continue
            buf.append(msg)
            if len(buf) >= args.batch_size:
                process_batch(buf, args.out_topic, p, args.out_dir, args.missing_warn_threshold)
                c.commit()
                buf = []
    except KeyboardInterrupt:
        logging.info("Stopping...")
    finally:
        if buf:
            try:
                process_batch(buf, args.out_topic, p, args.out_dir, args.missing_warn_threshold)
                c.commit()
            except Exception:
                pass
        c.close()

if __name__ == "__main__":
    main()
