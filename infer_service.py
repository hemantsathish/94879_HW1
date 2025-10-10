import json, argparse, joblib, time
import pandas as pd
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from pathlib import Path
from common import parse_date_time, add_time_cols, RollingState

def build_row_features(record, state, feature_order, target_col="CO(GT)", key="global"):
    """
    Build a single-row feature vector matching feature_order.
    Returns (ready: bool, Xrow: np.ndarray, aux_info: dict)
    """
    # Parse datetime index
    dt = parse_date_time(record) 
    if pd.isna(dt):
        return False, None, {"reason": "bad_datetime"}
    df = pd.DataFrame([record])
    df["DateTime"] = dt
    df = df.set_index("DateTime")

    df = add_time_cols(df)

    # Push target into state buffer
    try:
        target_val = float(record.get(target_col, np.nan))
    except Exception:
        target_val = np.nan

    if not np.isnan(target_val):
        state.push(key, target_val)

    # Assemble lag/roll features from buffer
    lag_roll = state.make_features(key)

    # Construct current numeric features
    numerics = df.select_dtypes(include=[np.number]).iloc[0].to_dict()
    numerics.pop(target_col, None)

    feats = {**numerics, **lag_roll}

    missing = [c for c in feature_order if c not in feats]
    if missing:
        # Not enough history yet or field missing
        return False, None, {"reason": "missing_features", "missing": missing[:5]}

    # Order and create row
    Xrow = np.array([feats[c] for c in feature_order], dtype=float)
    # replace inf/nan
    if not np.all(np.isfinite(Xrow)):
        return False, None, {"reason": "non_finite_in_features"}
    return True, Xrow, {"ok": True}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default="127.0.0.1:9092")
    ap.add_argument("--in-topic", default="air_quality.clean")
    ap.add_argument("--out-topic", default="air_quality.pred")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--min_history", type=int, default=24, help="min points before predicting")
    ap.add_argument("--use_sarima", action="store_true", help="emit SARIMA alongside RF")
    ap.add_argument("--group-id", default="phase3-infer", help="Kafka consumer group id")
    args = ap.parse_args()

    # Load artifacts
    feats_meta = json.load(open(Path(args.artifacts)/"features.json"))
    feature_order = feats_meta["feature_order"]
    target_col = feats_meta.get("target", "CO(GT)")

    rf = joblib.load(Path(args.artifacts)/"model.pkl")
    sarima = None
    sarima_res = None
    if args.use_sarima:
        try:
            sarima = joblib.load(Path(args.artifacts)/"sarima.pkl")
            sarima_res = sarima
        except Exception:
            sarima = None

    consumer = KafkaConsumer(
        args.in_topic,
        bootstrap_servers=args.bootstrap,
        group_id=args.group_id,        
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
    )
    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        value_serializer=lambda d: json.dumps(d).encode("utf-8"),
        acks="all"
    )

    state = RollingState(maxlen=max(48, args.min_history))
    key = "global"

    print(f"[infer] Listening on '{args.in_topic}', writing predictions to '{args.out_topic}' ...")
    for msg in consumer:
        record = msg.value

        ready, Xrow, info = build_row_features(
            record, state, feature_order, target_col=target_col, key=key
        )

        # Keep filling buffer if not enough data
        if not state.ready(key, args.min_history) or not ready:
            continue

        # RF prediction
        yhat = float(rf.predict(Xrow.reshape(1, -1))[0])
        out = {
            "RandomForest_Prediction": yhat
        }
        out["Timestamp"] = record.get("ts")


        # SARIMA one-step-ahead forecast
        if sarima_res is not None:
            try:
                sar_fore = float(sarima_res.forecast(steps=1).iloc[0])
                out["SARIMA_Prediction"] = sar_fore
            except Exception:
                pass

        try:
            out["True_Value"] = float(record.get(target_col))
        except Exception:
            pass

        producer.send(args.out_topic, out)

    print("[infer] Consumer loop ended.")

if __name__ == "__main__":
    main()
