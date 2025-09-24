# Script to split the UCI Air Quality dataset into 70% train and 30% stream/test.
# Outputs CSV files that can be reused by training (Phase 3) and streaming (Phase 1 producer).

import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser("Split UCI Air Quality Dataset")
    ap.add_argument("--csv", required=True, type=Path,
                    help="Path to AirQualityUCI.csv")
    ap.add_argument("--train-frac", type=float, default=0.7,
                    help="Fraction of data for training (default 0.7)")
    ap.add_argument("--outdir", type=Path,
                    default=Path("../data/splits"),
                    help="Directory to save split CSVs")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, sep=";", decimal=",", engine="python")

    n = len(df)
    cut = int(n * args.train_frac)

    df_train = df.iloc[:cut].copy()
    df_stream = df.iloc[cut:].copy()

    args.outdir.mkdir(parents=True, exist_ok=True)

    train_path = args.outdir / f"{args.csv.stem}_train.csv"
    stream_path = args.outdir / f"{args.csv.stem}_stream.csv"

    df_train.to_csv(train_path, sep=";", index=False)
    df_stream.to_csv(stream_path, sep=";", index=False)

    print(f"Total rows: {n}")
    print(f"Train ({args.train_frac*100:.0f}%): {len(df_train)} rows -> {train_path}")
    print(f"Stream ({(1-args.train_frac)*100:.0f}%): {len(df_stream)} rows -> {stream_path}")

if __name__ == "__main__":
    main()