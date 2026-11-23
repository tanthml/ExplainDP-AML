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
    TEST_SIZE, VAL_SIZE, HIDDEN_DIMS, DROPOUT, DIRICHLET_ALPHA, LEARNING_RATE,
)
from preprocessing.utils import load_and_prepare, load_from_features_parquet, grouped_train_test_split
from models.mlp import MLP
from fl.client import FLClient
from fl.server import fedavg_aggregate, evaluate_global
from fl.dp_client import DP_MAX_SAMPLES


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


def run_fedavg(clients, model_kwargs, X_val, y_val, num_rounds, device, max_samples=None, lr=LEARNING_RATE,
               mu=0.0, loss_type="weighted"):
    """Train FedAvg for num_rounds, selecting the reported round AND decision
    threshold from (X_val, y_val) -- never from the test set, which the caller
    evaluates separately once training is done. mu > 0 trains FedProx instead."""
    global_model = MLP(**model_kwargs).to(device)
    global_weights = global_model.get_weights()
    history = []
    best_f1 = -1.0
    best_weights = None
    best_round = None
    best_threshold = None

    for r in range(1, num_rounds + 1):
        all_w, all_n, all_loss = [], [], []
        for c in clients:
            w, n, loss = c.train(global_weights, max_samples=max_samples, lr=lr, mu=mu, loss_type=loss_type)
            all_w.append(w)
            all_n.append(n)
            all_loss.append(loss)

        global_weights = fedavg_aggregate(all_w, all_n)
        global_model.set_weights(global_weights)
        metrics = evaluate_global(global_model, X_val, y_val, device)
        avg_loss = float(np.mean(all_loss))
        history.append({"round": r, "val_f1_macro": metrics["f1_macro"], "val_auc": metrics["auc"], "loss": avg_loss})
        print(f"  Round {r:02d}  val F1={metrics['f1_macro']:.4f}  val AUC={metrics['auc']:.4f}  loss={avg_loss:.4f}")

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_weights = [w.clone() for w in global_weights]
            best_round = r
            best_threshold = metrics["threshold"]

    global_model.set_weights(best_weights)
    return global_model, history, best_round, best_threshold


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
    parser.add_argument("--max-samples", type=int, default=DP_MAX_SAMPLES,
                        help="Max samples per client per round (0=unlimited). Defaults to "
                             "DP_MAX_SAMPLES so the non-private baseline and DP-FL (exp3) see "
                             "the same per-client cap Kmax -- otherwise the comparison between "
                             "them is confounded by unequal data access, not just DP noise.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Learning rate (default: config.LEARNING_RATE, which is "
                             "tuned for DP-FL, not FedAvg -- override for a fair, "
                             "separately-tuned FedAvg comparison).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[18, 19, 20],
                        help="Random seeds for multi-seed averaging (default: 3 seeds, "
                             "chosen to match the IEEE-RIVF 2026 presentation dates "
                             "18-20 Dec 2026 -- any 3 distinct seeds are statistically "
                             "equivalent). The train/val/test split is fixed "
                             "(config.RANDOM_SEED) across seeds; only client partitioning "
                             "and training vary.")
    parser.add_argument("--mu", type=float, default=0.0,
                        help="FedProx proximal term coefficient (0 = plain FedAvg). "
                             "mu > 0 trains FedProx: adds (mu/2)*||theta_local-theta_global||^2 "
                             "to each client's loss, penalising local drift.")
    parser.add_argument("--sampler", choices=["positive_preserving", "uniform"],
                        default="positive_preserving",
                        help="Per-client subsampling strategy to Kmax (reviewer point 6 "
                             "ablation). positive_preserving (default): keep all positives, "
                             "uniformly subsample negatives. uniform: subsample uniformly "
                             "regardless of label.")
    parser.add_argument("--loss-type", choices=["weighted", "ipw", "focal"], default="weighted",
                        help="Loss function (reviewer point 6 ablation). weighted (default): "
                             "fixed class-prior weights. ipw: inverse-probability weighting by "
                             "each row's actual per-round subsampling inclusion probability. "
                             "focal: focal loss (Lin et al. 2017) combined with class weights.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

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
        X_tv, X_test, y_tv, y_test, banks_tv, banks_test, groups_tv, groups_test = grouped_train_test_split(
            X, y, banks, groups, groups=groups, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )
    else:
        X_tv, X_test, y_tv, y_test, groups_tv, groups_test = grouped_train_test_split(
            X, y, groups, groups=groups, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )
        banks_tv = None

    if banks_tv is not None:
        X_client, X_val, y_client, y_val, banks_client, banks_val = grouped_train_test_split(
            X_tv, y_tv, banks_tv, groups=groups_tv, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=RANDOM_SEED
        )
    else:
        X_client, X_val, y_client, y_val = grouped_train_test_split(
            X_tv, y_tv, groups=groups_tv, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=RANDOM_SEED
        )
        banks_client = None

    model_kwargs = {"input_dim": X.shape[1], "hidden_dims": HIDDEN_DIMS, "dropout": DROPOUT}
    max_samples = args.max_samples if args.max_samples > 0 else None
    # Tag encodes every axis that differs from the paper's main config
    # (positive_preserving sampler, weighted loss, mu=0), so the sampling/
    # calibration ablation grid (reviewer point 6) and FedProx (point 5)
    # never collide on output filenames.
    tag = "fedprox" if args.mu > 0 else "fedavg"
    if args.sampler != "positive_preserving":
        tag += "_unif"
    if args.loss_type != "weighted":
        tag += f"_{args.loss_type}"
    print(f"\n=== {'FedProx (mu='+str(args.mu)+')' if args.mu > 0 else 'FedAvg'} | "
          f"sampler={args.sampler} | loss={args.loss_type} | "
          f"{args.num_clients} clients | {args.rounds} rounds | {args.partition} | "
          f"max_samples={max_samples} | lr={args.lr} | seeds={args.seeds} === [{tag}]")

    seed_results = []
    for si, seed in enumerate(args.seeds):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        if args.partition == "bank" and banks_client is not None:
            client_data = partition_bank(X_client, y_client, banks_client, args.num_clients)
        else:
            client_data = partition_dirichlet(X_client, y_client, args.num_clients, DIRICHLET_ALPHA, rng)

        clients = [FLClient(i, Xc, yc, model_kwargs, device, sampler=args.sampler)
                   for i, (Xc, yc) in enumerate(client_data)]
        if si == 0:
            for i, c in enumerate(clients):
                print(f"  Client {i}: {c.n_samples:,} samples  illicit={c.y.float().mean():.2%}")

        global_model, history, best_round, best_threshold = run_fedavg(
            clients, model_kwargs, X_val, y_val, args.rounds, device, max_samples=max_samples, lr=args.lr,
            mu=args.mu, loss_type=args.loss_type,
        )
        test_metrics = evaluate_global(global_model, X_test, y_test, device, threshold=best_threshold)
        print(f"[seed {seed}] best round {best_round} (val-selected)  "
              f"test F1={test_metrics['f1_macro']:.4f}  test AUC={test_metrics['auc']:.4f}  "
              f"threshold={best_threshold:.4f}")

        seed_results.append({
            "seed": seed, "best_round": best_round, "threshold": best_threshold,
            "val_history": history,
            "test_f1_macro": test_metrics["f1_macro"], "test_auc": test_metrics["auc"],
        })

        print(f"Saving seed-{seed} best-round ({best_round}) checkpoint (not final round {args.rounds})")
        torch.save(global_model.state_dict(), RESULTS_DIR / "models" / f"{tag}_seed{seed}.pt")
        if si == 0:
            # Also save under the un-suffixed name for backward compatibility
            # with any script still reading the single-seed checkpoint.
            torch.save(global_model.state_dict(), RESULTS_DIR / "models" / f"{tag}.pt")

    f1s = [s["test_f1_macro"] for s in seed_results]
    aucs = [s["test_auc"] for s in seed_results]
    summary = {
        "f1_macro_mean": float(np.mean(f1s)), "f1_macro_std": float(np.std(f1s)),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
    }
    print(f"\n=== Aggregate over {len(args.seeds)} seeds ===")
    print(f"Test F1  = {summary['f1_macro_mean']:.4f} ± {summary['f1_macro_std']:.4f}")
    print(f"Test AUC = {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")

    out = RESULTS_DIR / "tables" / f"exp2_{tag}.json"
    with open(out, "w") as f:
        json.dump({"experiment": f"exp2_{tag}", "max_samples": max_samples, "lr": args.lr, "mu": args.mu,
                   "sampler": args.sampler, "loss_type": args.loss_type,
                   "seeds": args.seeds, "per_seed": seed_results, "summary": summary}, f, indent=2)
    print(f"Results → {out}")


if __name__ == "__main__":
    main()
