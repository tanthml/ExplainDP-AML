"""Partition features.parquet into federated client splits.

Two strategies:
  --strategy bank     (default) Each of the top-N banks becomes one client.
  --strategy dirichlet          Dirichlet(alpha) label-based non-IID split.

Output:
  data/partitions/<strategy>/client_<i>_{train,val,test}.parquet
  data/partitions/<strategy>/global_test.parquet
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, NUM_CLIENTS, RANDOM_SEED, DIRICHLET_ALPHA, TEST_SIZE, VAL_SIZE


def bank_partition(df: pd.DataFrame, num_clients: int) -> list:
    """Each client gets transactions from the top-N most common source banks."""
    top_banks = df["From Bank"].value_counts().head(num_clients).index.tolist()
    partitions = []
    for bank in top_banks:
        partitions.append(df[df["From Bank"] == bank].copy())
    # Remaining transactions go to the last client
    others = df[~df["From Bank"].isin(top_banks)].copy()
    if len(others) > 0 and len(partitions) > 0:
        partitions[-1] = pd.concat([partitions[-1], others], ignore_index=True)
    return partitions


def dirichlet_partition(df: pd.DataFrame, num_clients: int, alpha: float, rng: np.random.Generator) -> list:
    """Label-proportional non-IID split using Dirichlet distribution."""
    labels = df["Is Laundering"].values
    n = len(labels)
    partitions = [[] for _ in range(num_clients)]

    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        proportions = rng.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(idx)).astype(int)
        # Fix rounding
        proportions[-1] += len(idx) - proportions.sum()
        start = 0
        for i, size in enumerate(proportions):
            partitions[i].extend(idx[start:start + size].tolist())
            start += size

    return [df.iloc[sorted(p)].copy() for p in partitions]


def split_client_data(client_df: pd.DataFrame, val_size: float, test_size: float, rng: np.random.Generator):
    """Split into train/val/test preserving class ratio."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(client_df))
    y = client_df["Is Laundering"].values

    try:
        train_idx, test_idx = train_test_split(idx, test_size=test_size, stratify=y, random_state=RANDOM_SEED)
        y_train = y[train_idx]
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=val_size / (1 - test_size),
            stratify=y_train,
            random_state=RANDOM_SEED,
        )
    except ValueError:
        # Fallback when class too rare for stratify
        train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=RANDOM_SEED)
        train_idx, val_idx = train_test_split(train_idx, test_size=val_size / (1 - test_size), random_state=RANDOM_SEED)

    return (
        client_df.iloc[train_idx],
        client_df.iloc[val_idx],
        client_df.iloc[test_idx],
    )


def main():
    parser = argparse.ArgumentParser(description="Partition features into FL client splits.")
    parser.add_argument("--input", default=str(DATA_DIR / "features.parquet"))
    parser.add_argument("--strategy", choices=["bank", "dirichlet"], default="bank")
    parser.add_argument("--num-clients", type=int, default=NUM_CLIENTS)
    parser.add_argument("--alpha", type=float, default=DIRICHLET_ALPHA, help="Dirichlet alpha")
    parser.add_argument("--global-test-size", type=float, default=0.1)
    args = parser.parse_args()

    rng = np.random.default_rng(RANDOM_SEED)
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} rows from {args.input}")

    # Hold out a global test set before partitioning
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(df))
    try:
        client_idx, test_idx = train_test_split(
            idx, test_size=args.global_test_size,
            stratify=df["Is Laundering"].values,
            random_state=RANDOM_SEED,
        )
    except ValueError:
        client_idx, test_idx = train_test_split(idx, test_size=args.global_test_size, random_state=RANDOM_SEED)

    global_test = df.iloc[test_idx]
    client_pool = df.iloc[client_idx].reset_index(drop=True)

    out_dir = DATA_DIR / "partitions" / args.strategy
    out_dir.mkdir(parents=True, exist_ok=True)

    global_test.to_parquet(out_dir / "global_test.parquet", index=False)
    print(f"Global test set: {len(global_test):,} rows")

    if args.strategy == "bank":
        partitions = bank_partition(client_pool, args.num_clients)
    else:
        partitions = dirichlet_partition(client_pool, args.num_clients, args.alpha, rng)

    print(f"\nPartition summary ({args.strategy}, {args.num_clients} clients):")
    for i, part in enumerate(partitions):
        train, val, test = split_client_data(part, VAL_SIZE, TEST_SIZE, rng)
        illicit = part["Is Laundering"].mean()
        print(f"  Client {i}: total={len(part):,}  illicit={illicit:.2%}  "
              f"train={len(train):,}  val={len(val):,}  test={len(test):,}")
        train.to_parquet(out_dir / f"client_{i}_train.parquet", index=False)
        val.to_parquet(out_dir / f"client_{i}_val.parquet", index=False)
        test.to_parquet(out_dir / f"client_{i}_test.parquet", index=False)

    print(f"\nSaved splits → {out_dir}")


if __name__ == "__main__":
    main()
