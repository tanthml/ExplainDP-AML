"""Feature engineering for IBM AML transactions.

Reads HI-Small_Merged.parquet (or raw CSV) and produces a numpy-ready feature matrix.
Saves:
  data/features.parquet   — full feature DataFrame with labels + pattern_type
  data/feature_names.json — list of feature column names
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, RANDOM_SEED


CATEGORICAL_COLS = ["Payment Format", "Payment Currency", "Receiving Currency"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Temporal features
    df["tx_hour"] = df["Timestamp"].dt.hour
    df["tx_day_of_week"] = df["Timestamp"].dt.dayofweek
    df["tx_day_of_month"] = df["Timestamp"].dt.day

    # Amount features
    eps = 1e-9
    df["log_amount_paid"] = np.log1p(df["Amount Paid"])
    df["log_amount_received"] = np.log1p(df["Amount Received"])
    df["amount_ratio"] = df["Amount Paid"] / (df["Amount Received"] + eps)
    df["log_amount_ratio"] = np.log1p(df["amount_ratio"])
    df["amount_diff"] = np.abs(df["Amount Paid"] - df["Amount Received"])
    df["log_amount_diff"] = np.log1p(df["amount_diff"])

    # Currency flags
    df["is_same_currency"] = (df["Payment Currency"] == df["Receiving Currency"]).astype(int)
    df["is_cross_bank"] = (df["From Bank"] != df["To Bank"]).astype(int)

    # Bank encoding (frequency encoding — robust to high cardinality)
    for col in ["From Bank", "To Bank"]:
        freq = df[col].value_counts(normalize=True)
        df[f"{col.replace(' ', '_').lower()}_freq"] = df[col].map(freq).fillna(0.0)

    # Label-encode low-cardinality categoricals
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[f"{col.replace(' ', '_').lower()}_enc"] = le.fit_transform(df[col].fillna("UNK"))
        encoders[col] = le

    return df, encoders


def get_feature_columns(df: pd.DataFrame) -> list:
    drop = {
        "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
        "Amount Received", "Receiving Currency", "Amount Paid",
        "Payment Currency", "Payment Format",
        "Is Laundering", "pattern_type", "group_id",
        "amount_ratio", "amount_diff",  # keep log versions
    }
    return [c for c in df.columns if c not in drop]


def main():
    parser = argparse.ArgumentParser(description="Feature engineering for IBM AML.")
    parser.add_argument("--input", default=str(DATA_DIR / "HI-Small_Merged.parquet"))
    parser.add_argument("--out-features", default=str(DATA_DIR / "features.parquet"))
    parser.add_argument("--out-names", default=str(DATA_DIR / "feature_names.json"))
    parser.add_argument("--out-scaler", default=str(DATA_DIR / "scaler.pkl"))
    parser.add_argument("--nrows", type=int, default=None)
    args = parser.parse_args()

    inp = Path(args.input)
    if inp.suffix == ".parquet":
        df = pd.read_parquet(inp)
    else:
        df = pd.read_csv(inp, nrows=args.nrows, low_memory=False)
        df.columns = [c.strip() for c in df.columns]

    if args.nrows:
        df = df.head(args.nrows)

    print(f"Loaded {len(df):,} rows")

    df, encoders = engineer_features(df)
    feature_cols = get_feature_columns(df)
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    X = df[feature_cols].values.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Rebuild scaled feature DataFrame
    out_df = pd.DataFrame(X_scaled, columns=feature_cols)
    out_df["Is Laundering"] = df["Is Laundering"].values
    if "pattern_type" in df.columns:
        out_df["pattern_type"] = df["pattern_type"].values
    if "group_id" in df.columns:
        out_df["group_id"] = df["group_id"].values
    if "From Bank" in df.columns:
        out_df["From Bank"] = df["From Bank"].values  # keep for partitioning

    out_df.to_parquet(args.out_features, index=False)
    print(f"Saved features → {args.out_features}")

    with open(args.out_names, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Saved feature names → {args.out_names}")

    joblib.dump(scaler, args.out_scaler)
    print(f"Saved scaler → {args.out_scaler}")


if __name__ == "__main__":
    main()
