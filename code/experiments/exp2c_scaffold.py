"""Experiment 2c — SCAFFOLD (non-DP only; see fl/scaffold.py docstring for why
DP-SCAFFOLD is out of scope for this camera-ready).

Usage:
    python experiments/exp2c_scaffold.py --fast --rounds 20
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
from fl.server import evaluate_global
from fl.dp_client import DP_MAX_SAMPLES
from fl.scaffold import SCAFFOLDClient, run_scaffold
from experiments.exp2_fedavg import partition_bank, partition_dirichlet


def main():
    parser = argparse.ArgumentParser(description="SCAFFOLD experiment (non-DP).")
    parser.add_argument("--data", default=str(DATA_DIR / "HI-Small_Trans.csv"))
    parser.add_argument("--patterns", default=str(DATA_DIR / "HI-Small_Patterns.txt"))
    parser.add_argument("--num-clients", type=int, default=NUM_CLIENTS)
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--partition", choices=["bank", "dirichlet"], default="bank")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--max-samples", type=int, default=DP_MAX_SAMPLES)
    parser.add_argument("--lr", type=float, default=0.01,
                        help="SCAFFOLD uses vanilla SGD (fl/scaffold.py), so the Adam-tuned "
                             "LEARNING_RATE elsewhere in this codebase (3e-3) is not necessarily "
                             "appropriate; default here is a separate, SGD-appropriate value.")
    parser.add_argument("--sampler", choices=["positive_preserving", "uniform"], default="positive_preserving")
    parser.add_argument("--loss-type", choices=["weighted", "focal"], default="weighted")
    parser.add_argument("--seeds", type=int, nargs="+", default=[18, 19, 20])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    max_samples = args.max_samples if args.max_samples > 0 else None
    tag = "scaffold"
    if args.sampler != "positive_preserving":
        tag += "_unif"
    if args.loss_type != "weighted":
        tag += f"_{args.loss_type}"

    features_parquet = DATA_DIR / "features.parquet"
    if args.fast and features_parquet.exists():
        print(f"Fast load from {features_parquet}...")
        X, y, pt, banks, feature_cols, scaler, groups = load_from_features_parquet(features_parquet)
    else:
        X, y, pt, banks, feature_cols, scaler, groups = load_and_prepare(
            Path(args.data), Path(args.patterns), nrows=args.nrows
        )

    X_tv, X_test, y_tv, y_test, banks_tv, banks_test, groups_tv, groups_test = grouped_train_test_split(
        X, y, banks, groups, groups=groups, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    X_client, X_val, y_client, y_val, banks_client, banks_val = grouped_train_test_split(
        X_tv, y_tv, banks_tv, groups=groups_tv, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=RANDOM_SEED
    )

    model_kwargs = {"input_dim": X.shape[1], "hidden_dims": HIDDEN_DIMS, "dropout": DROPOUT}
    print(f"\n=== SCAFFOLD | sampler={args.sampler} | loss={args.loss_type} | lr={args.lr} | "
          f"{args.num_clients} clients | {args.rounds} rounds | seeds={args.seeds} === [{tag}]")

    seed_results = []
    for si, seed in enumerate(args.seeds):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        if args.partition == "bank" and banks_client is not None:
            client_data = partition_bank(X_client, y_client, banks_client, args.num_clients)
        else:
            client_data = partition_dirichlet(X_client, y_client, args.num_clients, DIRICHLET_ALPHA, rng)

        clients = [SCAFFOLDClient(i, Xc, yc, model_kwargs, device, sampler=args.sampler)
                   for i, (Xc, yc) in enumerate(client_data)]
        if si == 0:
            for i, c in enumerate(clients):
                print(f"  Client {i}: {c.n_samples:,} samples  illicit={c.y.float().mean():.2%}")

        global_model, history, best_round, best_threshold = run_scaffold(
            clients, model_kwargs, X_val, y_val, args.rounds, device,
            max_samples=max_samples, lr=args.lr, loss_type=args.loss_type,
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

        torch.save(global_model.state_dict(), RESULTS_DIR / "models" / f"{tag}_seed{seed}.pt")
        if si == 0:
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
        json.dump({"experiment": f"exp2_{tag}", "max_samples": max_samples, "lr": args.lr,
                   "sampler": args.sampler, "loss_type": args.loss_type,
                   "seeds": args.seeds, "per_seed": seed_results, "summary": summary}, f, indent=2)
    print(f"Results → {out}")


if __name__ == "__main__":
    main()
