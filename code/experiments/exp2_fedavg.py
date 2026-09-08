"""Experiment 2 — FedAvg without Differential Privacy.

Usage:
    python experiments/exp2_fedavg.py \\
        --data data/HI-Small_Trans.csv \\
        --patterns data/HI-Small_Patterns.txt \\
        --num-clients 5 --rounds 20 --partition bank
"""

import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_DIR, RESULTS_DIR, RANDOM_SEED, NUM_CLIENTS, NUM_ROUNDS,
    TEST_SIZE, HIDDEN_DIMS, DROPOUT, DIRICHLET_ALPHA, LEARNING_RATE,
)
from preprocessing.utils import load_and_prepare, load_from_features_parquet, grouped_train_test_split
from models.mlp import MLP
from fl.client import FLClient
from fl.server import fedavg_aggregate, evaluate_global


def partition_bank(X, y, banks, num_clients):
    unique, counts = np.unique(banks, return_counts=True)
    top_banks = unique[np.argsort(-counts)][:num_clients]
    parts = []
    for bank in top_banks:
        mask = banks == bank
        parts.append((X[mask], y[mask]))
    others = ~np.isin(banks, top_banks)
    if others.any() and parts:
        Xl, yl = parts[-1]
        parts[-1] = (np.vstack([Xl, X[others]]), np.concatenate([yl, y[others]]))
    return parts


def partition_dirichlet(X, y, num_clients, alpha, rng):
    partitions = [[] for _ in range(num_clients)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        props = rng.dirichlet([alpha] * num_clients)
        sizes = (props * len(idx)).astype(int)
        sizes[-1] += len(idx) - sizes.sum()
        start = 0
        for i, sz in enumerate(sizes):
            partitions[i].extend(idx[start:start + sz].tolist())
            start += sz
    return [(X[sorted(p)], y[sorted(p)]) for p in partitions]


def run_fedavg(clients, model_kwargs, X_test, y_test, num_rounds, device, max_samples=None, lr=LEARNING_RATE):
    global_model = MLP(**model_kwargs).to(device)
    global_weights = global_model.get_weights()
    history = []
    best_f1 = -1.0
    best_weights = None
    best_round = None

    for r in range(1, num_rounds + 1):
        all_w, all_n, all_loss = [], [], []
        for c in clients:
            w, n, loss = c.train(global_weights, max_samples=max_samples, lr=lr)
            all_w.append(w)
            all_n.append(n)
            all_loss.append(loss)

        global_weights = fedavg_aggregate(all_w, all_n)
        global_model.set_weights(global_weights)
        metrics = evaluate_global(global_model, X_test, y_test, device)
        avg_loss = float(np.mean(all_loss))
        history.append({"round": r, "f1_macro": metrics["f1_macro"], "auc": metrics["auc"], "loss": avg_loss})
        print(f"  Round {r:02d}  F1={metrics['f1_macro']:.4f}  AUC={metrics['auc']:.4f}  loss={avg_loss:.4f}")

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_weights = [w.clone() for w in global_weights]
            best_round = r

    global_model.set_weights(best_weights)
    return global_model, history, best_round


def main():
    parser = argparse.ArgumentParser(description="FedAvg experiment (no DP).")
    parser.add_argument("--data", default=str(DATA_DIR / "HI-Small_Trans.csv"))
    parser.add_argument("--patterns", default=str(DATA_DIR / "HI-Small_Patterns.txt"))
    parser.add_argument("--num-clients", type=int, default=NUM_CLIENTS)
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--partition", choices=["bank", "dirichlet"], default="bank")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--fast", action="store_true",
                        help="Load from pre-computed features.parquet instead of raw CSV.")
    parser.add_argument("--max-samples", type=int, default=100000,
                        help="Max samples per client per round (0=unlimited).")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Learning rate (default: config.LEARNING_RATE, which is "
                             "tuned for DP-FL, not FedAvg -- override for a fair, "
                             "separately-tuned FedAvg comparison).")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rng = np.random.default_rng(RANDOM_SEED)
    device = torch.device(args.device)

    features_parquet = DATA_DIR / "features.parquet"
    if args.fast and features_parquet.exists():
        print(f"Fast load from {features_parquet}...")
        X, y, pt, banks, feature_cols, scaler, groups = load_from_features_parquet(features_parquet)
    else:
        X, y, pt, banks, feature_cols, scaler, groups = load_and_prepare(
            Path(args.data), Path(args.patterns), nrows=args.nrows
        )
    print(f"Dataset: {len(X):,} samples  illicit={y.mean():.2%}")

    if banks is not None:
        X_client, X_test, y_client, y_test, banks_client, banks_test = grouped_train_test_split(
            X, y, banks, groups=groups, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )
    else:
        X_client, X_test, y_client, y_test = grouped_train_test_split(
            X, y, groups=groups, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )
        banks_client = None

    if args.partition == "bank" and banks_client is not None:
        client_data = partition_bank(X_client, y_client, banks_client, args.num_clients)
    else:
        client_data = partition_dirichlet(X_client, y_client, args.num_clients, DIRICHLET_ALPHA, rng)

    model_kwargs = {"input_dim": X.shape[1], "hidden_dims": HIDDEN_DIMS, "dropout": DROPOUT}
    clients = [FLClient(i, Xc, yc, model_kwargs, device) for i, (Xc, yc) in enumerate(client_data)]
    for i, c in enumerate(clients):
        print(f"  Client {i}: {c.n_samples:,} samples  illicit={c.y.float().mean():.2%}")

    max_samples = args.max_samples if args.max_samples > 0 else None
    print(f"\n=== FedAvg | {args.num_clients} clients | {args.rounds} rounds | {args.partition} | "
          f"max_samples={max_samples} | lr={args.lr} ===")
    global_model, history, best_round = run_fedavg(clients, model_kwargs, X_test, y_test, args.rounds, device,
                                                    max_samples=max_samples, lr=args.lr)

    best = max(history, key=lambda h: h["f1_macro"])
    assert best["round"] == best_round
    print(f"\nBest: Round {best['round']}  F1={best['f1_macro']:.4f}  AUC={best['auc']:.4f}")
    print(f"Saving best-round ({best_round}) checkpoint (not final round {args.rounds})")

    torch.save(global_model.state_dict(), RESULTS_DIR / "models" / "fedavg.pt")

    out = RESULTS_DIR / "tables" / "exp2_fedavg.json"
    with open(out, "w") as f:
        json.dump({"experiment": "exp2_fedavg", "max_samples": max_samples, "lr": args.lr,
                   "history": history, "best": best}, f, indent=2)
    print(f"Results → {out}")


if __name__ == "__main__":
    main()
