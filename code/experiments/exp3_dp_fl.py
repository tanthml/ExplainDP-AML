"""Experiment 3 — DP-FL with Opacus (ε ∈ {1, 5, 10, 50}).

Usage:
    for eps in 1 5 10 50; do
        python experiments/exp3_dp_fl.py \\
            --data data/HI-Small_Trans.csv \\
            --patterns data/HI-Small_Patterns.txt \\
            --num-clients 5 --rounds 20 \\
            --epsilon $eps --delta 1e-5 --max-grad-norm 1.0
    done
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
    TEST_SIZE, DELTA, MAX_GRAD_NORM, HIDDEN_DIMS, DROPOUT, DIRICHLET_ALPHA,
)
from preprocessing.utils import load_and_prepare, load_from_features_parquet, grouped_train_test_split
from models.mlp import MLP
from fl.dp_client import DPFLClient, DP_MAX_SAMPLES
from fl.server import fedavg_aggregate, evaluate_global
from experiments.exp2_fedavg import partition_bank, partition_dirichlet


def run_dp_fedavg(clients, model_kwargs, X_test, y_test, num_rounds,
                  target_epsilon, delta, max_grad_norm, device):
    global_model = MLP(**model_kwargs).to(device)
    global_weights = global_model.get_weights()
    history = []
    best_f1 = -1.0
    best_weights = None
    best_round = None

    for r in range(1, num_rounds + 1):
        all_w, all_n, all_eps, all_loss = [], [], [], []
        for c in clients:
            w, n, eps_spent, loss = c.train(
                global_weights,
                target_epsilon=target_epsilon,
                delta=delta,
                max_grad_norm=max_grad_norm,
            )
            all_w.append(w)
            all_n.append(n)
            all_eps.append(eps_spent)
            all_loss.append(loss)

        global_weights = fedavg_aggregate(all_w, all_n)
        global_model.set_weights(global_weights)
        metrics = evaluate_global(global_model, X_test, y_test, device)
        avg_eps = float(np.mean(all_eps))
        avg_loss = float(np.mean(all_loss))
        history.append({
            "round": r,
            "f1_macro": metrics["f1_macro"],
            "auc": metrics["auc"],
            "epsilon_spent": avg_eps,
            "loss": avg_loss,
        })
        print(f"  Round {r:02d}  F1={metrics['f1_macro']:.4f}  AUC={metrics['auc']:.4f}"
              f"  ε_spent={avg_eps:.4f}  loss={avg_loss:.4f}")

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_weights = [w.clone() for w in global_weights]
            best_round = r

    global_model.set_weights(best_weights)
    return global_model, history, best_round


def main():
    parser = argparse.ArgumentParser(description="DP-FL experiment.")
    parser.add_argument("--data", default=str(DATA_DIR / "HI-Small_Trans.csv"))
    parser.add_argument("--patterns", default=str(DATA_DIR / "HI-Small_Patterns.txt"))
    parser.add_argument("--num-clients", type=int, default=NUM_CLIENTS)
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--epsilon", type=float, default=5.0)
    parser.add_argument("--delta", type=float, default=DELTA)
    parser.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM)
    parser.add_argument("--max-samples", type=int, default=DP_MAX_SAMPLES,
                        help="Max samples per client per round for DP training (0=unlimited).")
    parser.add_argument("--partition", choices=["bank", "dirichlet"], default="bank")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--fast", action="store_true",
                        help="Load from pre-computed features.parquet instead of raw CSV.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rng = np.random.default_rng(RANDOM_SEED)
    device = torch.device(args.device)
    max_samples = args.max_samples if args.max_samples > 0 else None
    print(f"=== DP-FL | ε={args.epsilon} | δ={args.delta} | C={args.max_grad_norm} | max_samples={max_samples} ===")

    features_parquet = DATA_DIR / "features.parquet"
    if args.fast and features_parquet.exists():
        print(f"Fast load from {features_parquet}...")
        X, y, pt, banks, feature_cols, scaler, groups = load_from_features_parquet(features_parquet)
    else:
        X, y, pt, banks, feature_cols, scaler, groups = load_and_prepare(
            Path(args.data), Path(args.patterns), nrows=args.nrows
        )

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
    clients = [DPFLClient(i, Xc, yc, model_kwargs, device, max_samples=max_samples)
               for i, (Xc, yc) in enumerate(client_data)]
    for i, c in enumerate(client_data):
        Xc, yc = c
        n_pos = int(yc.sum())
        n_sub = min(len(yc), max_samples) if max_samples else len(yc)
        print(f"  Client {i}: {len(yc):,} samples (illicit={n_pos}) → subsampled to ~{n_sub:,}")

    global_model, history, best_round = run_dp_fedavg(
        clients, model_kwargs, X_test, y_test,
        args.rounds, args.epsilon, args.delta, args.max_grad_norm, device,
    )

    best = max(history, key=lambda h: h["f1_macro"])
    assert best["round"] == best_round
    print(f"\nBest: Round {best['round']}  F1={best['f1_macro']:.4f}  "
          f"AUC={best['auc']:.4f}  ε_spent={best['epsilon_spent']:.4f}")
    print(f"Saving best-round ({best_round}) checkpoint (not final round {args.rounds})")

    eps_tag = str(args.epsilon).replace(".", "_")
    torch.save(global_model.state_dict(), RESULTS_DIR / "models" / f"dp_fl_eps{eps_tag}.pt")

    out = RESULTS_DIR / "tables" / f"exp3_dp_fl_eps{eps_tag}.json"
    with open(out, "w") as f:
        json.dump({
            "experiment": "exp3_dp_fl",
            "epsilon": args.epsilon,
            "delta": args.delta,
            "history": history,
            "best": best,
        }, f, indent=2)
    print(f"Results → {out}")


if __name__ == "__main__":
    main()
