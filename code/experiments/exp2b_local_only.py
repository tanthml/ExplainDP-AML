"""Experiment 2b — Local-only (no-collaboration) baseline.

Reviewer point 5: "A valid [FedAvg] comparison requires ... include at
least: (1) Local-only models." Each bank trains its own MLP using ONLY
its own partition, with no FedAvg aggregation across clients at all —
this quantifies how much FedAvg/DP-FL actually buy over a bank acting
alone, which the paper did not previously report.

Same per-client Kmax subsampling cap as FedAvg/DP-FL (fairness), same
R*E total local gradient steps as the federated runs (R=20 rounds *
E=5 epochs = 100 "effective epochs" over the client's own data, since
there is no cross-client averaging to do per round), evaluated on the
SAME held-out global test set used everywhere else in the paper (not
each bank's own slice) so the numbers are directly comparable to
Table 1/5.

Usage:
    python experiments/exp2b_local_only.py --fast --rounds 20
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
    TEST_SIZE, VAL_SIZE, HIDDEN_DIMS, DROPOUT, LEARNING_RATE, LOCAL_EPOCHS,
)
from preprocessing.utils import load_and_prepare, load_from_features_parquet, grouped_train_test_split
from models.mlp import MLP
from fl.client import FLClient
from fl.server import evaluate_global
from fl.dp_client import DP_MAX_SAMPLES
from experiments.exp2_fedavg import partition_bank


def train_local_only(client, model_kwargs, num_rounds, epochs_per_round, device,
                      X_val, y_val, max_samples, lr):
    """Train one client's model for num_rounds*epochs_per_round total local
    epochs, with no aggregation step -- the client's weights at round r+1
    continue from round r's weights (this client's own), never averaged
    with any other client. Round/threshold selection still uses the
    client's own view of (X_val, y_val) exactly like FedAvg/DP-FL, so the
    comparison isolates "no collaboration" rather than "no model selection
    discipline"."""
    model = MLP(**model_kwargs).to(device)
    weights = model.get_weights()
    best_f1, best_weights, best_round, best_threshold = -1.0, None, None, None

    for r in range(1, num_rounds + 1):
        weights, _, loss = client.train(weights, epochs=epochs_per_round, lr=lr, max_samples=max_samples)
        model.set_weights(weights)
        metrics = evaluate_global(model, X_val, y_val, device)
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_weights = [w.clone() for w in weights]
            best_round = r
            best_threshold = metrics["threshold"]

    model.set_weights(best_weights)
    return model, best_round, best_threshold


def main():
    parser = argparse.ArgumentParser(description="Local-only (no-collaboration) baseline.")
    parser.add_argument("--data", default=str(DATA_DIR / "HI-Small_Trans.csv"))
    parser.add_argument("--patterns", default=str(DATA_DIR / "HI-Small_Patterns.txt"))
    parser.add_argument("--num-clients", type=int, default=NUM_CLIENTS)
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--max-samples", type=int, default=DP_MAX_SAMPLES)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[18, 19, 20])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    max_samples = args.max_samples if args.max_samples > 0 else None

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
    print(f"=== Local-only (no collaboration) | {args.num_clients} clients | "
          f"max_samples={max_samples} | seeds={args.seeds} ===")

    seed_results = []
    for si, seed in enumerate(args.seeds):
        torch.manual_seed(seed)
        client_data = partition_bank(X_client, y_client, banks_client, args.num_clients)
        clients = [FLClient(i, Xc, yc, model_kwargs, device) for i, (Xc, yc) in enumerate(client_data)]

        per_client = []
        for i, client in enumerate(clients):
            model, best_round, best_threshold = train_local_only(
                client, model_kwargs, args.rounds, LOCAL_EPOCHS, device,
                X_val, y_val, max_samples, LEARNING_RATE,
            )
            test_metrics = evaluate_global(model, X_test, y_test, device, threshold=best_threshold)
            print(f"[seed {seed}] client {i} ({client.n_samples:,} rows, "
                  f"illicit={client.y.float().mean():.3%}): best round {best_round}  "
                  f"test F1={test_metrics['f1_macro']:.4f}  test AUC={test_metrics['auc']:.4f}")
            per_client.append({
                "client_id": i, "n_samples": client.n_samples,
                "best_round": best_round, "threshold": best_threshold,
                "test_f1_macro": test_metrics["f1_macro"], "test_auc": test_metrics["auc"],
            })
            if si == 0:
                torch.save(model.state_dict(), RESULTS_DIR / "models" / f"local_only_client{i}.pt")

        seed_results.append({"seed": seed, "per_client": per_client})

    f1s = [c["test_f1_macro"] for s in seed_results for c in s["per_client"]]
    aucs = [c["test_auc"] for s in seed_results for c in s["per_client"]]
    summary = {
        "f1_macro_mean": float(np.mean(f1s)), "f1_macro_std": float(np.std(f1s)),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "n_clients_x_seeds": len(f1s),
    }
    print(f"\n=== Aggregate over {len(args.seeds)} seeds x {args.num_clients} clients ===")
    print(f"Test F1  = {summary['f1_macro_mean']:.4f} ± {summary['f1_macro_std']:.4f}")
    print(f"Test AUC = {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    print("(Compare to FedAvg/DP-FL in Table 1 -- this isolates the value of "
          "collaboration itself, holding Kmax, architecture, and round budget fixed.)")

    out = RESULTS_DIR / "tables" / "exp2b_local_only.json"
    with open(out, "w") as f:
        json.dump({"experiment": "exp2b_local_only", "max_samples": max_samples,
                   "seeds": args.seeds, "per_seed": seed_results, "summary": summary}, f, indent=2)
    print(f"Results → {out}")


if __name__ == "__main__":
    main()
