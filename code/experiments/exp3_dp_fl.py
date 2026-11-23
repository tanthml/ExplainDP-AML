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
    TEST_SIZE, VAL_SIZE, DELTA, MAX_GRAD_NORM, HIDDEN_DIMS, DROPOUT, DIRICHLET_ALPHA,
)
from preprocessing.utils import load_and_prepare, load_from_features_parquet, grouped_train_test_split
from models.mlp import MLP
from fl.dp_client import DPFLClient, DP_MAX_SAMPLES
from fl.server import fedavg_aggregate, evaluate_global
from experiments.exp2_fedavg import partition_bank, partition_dirichlet


def run_dp_fedavg(clients, model_kwargs, X_val, y_val, num_rounds,
                  target_epsilon, delta, max_grad_norm, device, mu=0.0, loss_type="weighted"):
    """Train DP-FedAvg for num_rounds, selecting the reported round AND decision
    threshold from (X_val, y_val) -- never from the test set, which the caller
    evaluates separately once training is done. mu > 0 trains DP-FedProx."""
    global_model = MLP(**model_kwargs).to(device)
    global_weights = global_model.get_weights()
    history = []
    best_f1 = -1.0
    best_weights = None
    best_round = None
    best_threshold = None
    round1_accounting = None

    for r in range(1, num_rounds + 1):
        all_w, all_n, all_eps, all_loss, all_acct = [], [], [], [], []
        for c in clients:
            w, n, eps_spent, loss, acct = c.train(
                global_weights,
                target_epsilon=target_epsilon,
                delta=delta,
                max_grad_norm=max_grad_norm,
                mu=mu,
                loss_type=loss_type,
            )
            all_w.append(w)
            all_n.append(n)
            all_eps.append(eps_spent)
            all_loss.append(loss)
            all_acct.append(acct)

        if r == 1:
            round1_accounting = all_acct

        global_weights = fedavg_aggregate(all_w, all_n)
        global_model.set_weights(global_weights)
        metrics = evaluate_global(global_model, X_val, y_val, device)
        avg_eps = float(np.mean(all_eps))
        avg_loss = float(np.mean(all_loss))
        history.append({
            "round": r,
            "val_f1_macro": metrics["f1_macro"],
            "val_auc": metrics["auc"],
            "epsilon_spent": avg_eps,
            "loss": avg_loss,
        })
        print(f"  Round {r:02d}  val F1={metrics['f1_macro']:.4f}  val AUC={metrics['auc']:.4f}"
              f"  ε_spent={avg_eps:.4f}  loss={avg_loss:.4f}")

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_weights = [w.clone() for w in global_weights]
            best_round = r
            best_threshold = metrics["threshold"]

    global_model.set_weights(best_weights)
    return global_model, history, best_round, best_threshold, round1_accounting


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
    parser.add_argument("--seeds", type=int, nargs="+", default=[18, 19, 20],
                        help="Random seeds for multi-seed averaging (default: 3 seeds, "
                             "chosen to match the IEEE-RIVF 2026 presentation dates "
                             "18-20 Dec 2026 -- any 3 distinct seeds are statistically "
                             "equivalent). The train/val/test split is fixed "
                             "(config.RANDOM_SEED) across seeds; only client partitioning "
                             "and training vary.")
    parser.add_argument("--mu", type=float, default=0.0,
                        help="FedProx proximal term coefficient under DP-SGD (0 = plain DP-FL). "
                             "Injected directly into each parameter's .grad after Opacus's own "
                             "clip-and-noise step (see fl/dp_client.py), not through the loss.")
    parser.add_argument("--sampler", choices=["positive_preserving", "uniform"],
                        default="positive_preserving",
                        help="Per-client subsampling strategy to Kmax (reviewer point 6 ablation).")
    parser.add_argument("--loss-type", choices=["weighted", "ipw", "focal"], default="weighted",
                        help="Loss function (reviewer point 6 ablation): weighted (default), "
                             "ipw (inverse-probability weighting), or focal.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    max_samples = args.max_samples if args.max_samples > 0 else None
    tag = "dp_fedprox" if args.mu > 0 else "dp_fl"
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
    print(f"=== DP-FL | ε={args.epsilon} | δ={args.delta} | C={args.max_grad_norm} | "
          f"sampler={args.sampler} | loss={args.loss_type} | mu={args.mu} | "
          f"max_samples={max_samples} | seeds={args.seeds} === [{tag}]")

    seed_results = []
    for si, seed in enumerate(args.seeds):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        if args.partition == "bank" and banks_client is not None:
            client_data = partition_bank(X_client, y_client, banks_client, args.num_clients)
        else:
            client_data = partition_dirichlet(X_client, y_client, args.num_clients, DIRICHLET_ALPHA, rng)

        clients = [DPFLClient(i, Xc, yc, model_kwargs, device, max_samples=max_samples, sampler=args.sampler)
                   for i, (Xc, yc) in enumerate(client_data)]
        if si == 0:
            for i, (Xc, yc) in enumerate(client_data):
                n_pos = int(yc.sum())
                n_sub = min(len(yc), max_samples) if max_samples else len(yc)
                print(f"  Client {i}: {len(yc):,} samples (illicit={n_pos}) → subsampled to ~{n_sub:,}")

        global_model, history, best_round, best_threshold, round1_accounting = run_dp_fedavg(
            clients, model_kwargs, X_val, y_val,
            args.rounds, args.epsilon, args.delta, args.max_grad_norm, device,
            mu=args.mu, loss_type=args.loss_type,
        )
        test_metrics = evaluate_global(global_model, X_test, y_test, device, threshold=best_threshold)
        eps_spent_at_best = history[best_round - 1]["epsilon_spent"]
        print(f"[seed {seed}] best round {best_round} (val-selected)  "
              f"test F1={test_metrics['f1_macro']:.4f}  test AUC={test_metrics['auc']:.4f}  "
              f"ε_spent={eps_spent_at_best:.4f}  threshold={best_threshold:.4f}")

        seed_results.append({
            "seed": seed, "best_round": best_round, "threshold": best_threshold,
            "epsilon_spent": eps_spent_at_best, "val_history": history,
            "test_f1_macro": test_metrics["f1_macro"], "test_auc": test_metrics["auc"],
            "round1_accounting": round1_accounting,
        })

        eps_tag = str(args.epsilon).replace(".", "_")
        print(f"Saving seed-{seed} best-round ({best_round}) checkpoint (not final round {args.rounds})")
        torch.save(global_model.state_dict(), RESULTS_DIR / "models" / f"{tag}_eps{eps_tag}_seed{seed}.pt")
        if si == 0:
            # Also save under the un-suffixed name for backward compatibility
            # with any script still reading the single-seed checkpoint.
            torch.save(global_model.state_dict(), RESULTS_DIR / "models" / f"{tag}_eps{eps_tag}.pt")

    f1s = [s["test_f1_macro"] for s in seed_results]
    aucs = [s["test_auc"] for s in seed_results]
    summary = {
        "f1_macro_mean": float(np.mean(f1s)), "f1_macro_std": float(np.std(f1s)),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
    }
    print(f"\n=== Aggregate over {len(args.seeds)} seeds (ε={args.epsilon}) ===")
    print(f"Test F1  = {summary['f1_macro_mean']:.4f} ± {summary['f1_macro_std']:.4f}")
    print(f"Test AUC = {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")

    eps_tag = str(args.epsilon).replace(".", "_")
    out = RESULTS_DIR / "tables" / f"exp3_{tag}_eps{eps_tag}.json"
    with open(out, "w") as f:
        json.dump({
            "experiment": f"exp3_{tag}",
            "epsilon": args.epsilon,
            "delta": args.delta,
            "max_samples": max_samples,
            "mu": args.mu,
            "sampler": args.sampler,
            "loss_type": args.loss_type,
            "seeds": args.seeds,
            "per_seed": seed_results,
            "summary": summary,
        }, f, indent=2)
    print(f"Results → {out}")


if __name__ == "__main__":
    main()
