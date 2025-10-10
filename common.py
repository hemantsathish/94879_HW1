import pandas as pd
import numpy as np
from datetime import datetime

TIME_FORMATS = [
    "%d/%m/%Y %H.%M.%S",
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
]


def parse_date_time(record):
    ts = (
        record.get("ts")
        or record.get("Timestamp")
        or record.get("dateTime")
        or record.get("datetime")
    )
    if ts is not None:
        dt = pd.to_datetime(ts, errors="coerce", utc=True)
        if not pd.isna(dt):
            return dt.tz_convert(None)

    date = record.get("Date")
    time = record.get("Time")
    if date is not None and time is not None:
        tnorm = str(time).replace(".", ":")
        dt = pd.to_datetime(
            f"{str(date).strip()} {tnorm}", dayfirst=True, errors="coerce"
        )
        if not pd.isna(dt):
            return dt

    return pd.NaT


def add_time_cols(df_idxed):
    df_idxed["Hour"] = df_idxed.index.hour
    df_idxed["Day"] = df_idxed.index.day
    df_idxed["Month"] = df_idxed.index.month
    # cyclical encodings
    df_idxed["Hour_sin"] = np.sin(2 * np.pi * df_idxed["Hour"] / 24)
    df_idxed["Hour_cos"] = np.cos(2 * np.pi * df_idxed["Hour"] / 24)
    df_idxed["Season"] = ((df_idxed["Month"] % 12) // 3).astype(int)

    return df_idxed


# Keeps last N values for lag/rolling features by a key
class RollingState:
    def __init__(self, maxlen=48):
        from collections import deque

        self.maxlen = maxlen
        self.buf = {}

    def push(self, key, value):
        from collections import deque

        if key not in self.buf:
            self.buf[key] = deque(maxlen=self.maxlen)
        self.buf[key].append(value)

    def ready(self, key, need):
        return key in self.buf and len(self.buf[key]) >= need

    # Return dict of lag_*, rollmean_*, rollstd_* for the target series
    def make_features(
        self, key, lags=(1, 2, 3, 6, 12, 24), roll_windows=(3, 6, 12, 24)
    ):
        if key not in self.buf:
            return {}
        arr = np.array(self.buf[key], dtype=float)
        feats = {}
        # Lags
        for L in lags:
            if len(arr) >= L:
                feats[f"CO(GT)_lag{L}"] = arr[-L]
        # Rolling stats
        for W in roll_windows:
            if len(arr) >= W:
                window = arr[-W:]
                feats[f"CO(GT)_rollmean{W}"] = float(np.nanmean(window))
                feats[f"CO(GT)_rollstd{W}"] = float(np.nanstd(window))
        return feats
