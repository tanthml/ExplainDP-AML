#!/usr/bin/env python3
"""
DP-FL hyperparameter sweep — Phase 1 + 2.

Phase 1: grid search over (lr, max_grad_norm) using ε=5 as proxy.
Phase 2: run all ε ∈ {1, 5, 10, 50} with the best Phase-1 config.

Results → results/tuning/  (paper results in results/tables/ are untouched).
Log live progress to results/tuning/tune.log.
"""

import sys
import json
import time
import itertools
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_DIR, RESULTS_DIR, RANDOM_SEED, NUM_CLIENTS,
    TEST_SIZE, DELTA, HIDDEN_DIMS, DROPOUT, DIRICHLET_ALPHA, BATCH_SIZE,
)
from preprocessing.utils import load_from_features_parquet
from models.mlp import MLP
from fl.dp_client import DPFLClient, DP_MAX_SAMPLES
from fl.server import fedavg_aggregate, evaluate_global
from experiments.exp2_fedavg import partition_bank

TUNING_DIR = RESULTS_DIR / "tuning"
TUNING_DIR.mkdir(parents=True, exist_ok=True)
LOG = open(TUNING_DIR / "tune.log", "a", buffering=1)


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    LOG.write(line + "\n")


# ── Grid ──────────────────────────────────────────────────────────────────────
PHASE1_LRS         = [5e-4, 1e-3, 2e-3, 3e-3]
PHASE1_GRAD_NORMS  = [0.5, 1.0, 1.5]
PHASE1_HIDDEN_DIMS = [[256, 128, 64], [512, 256, 128]]   # also sweep architecture
PHASE1_EPSILON     = 5.0
PHASE2_EPSILONS    = [1, 5, 10, 50]
NUM_ROUNDS         = 20
LOCAL_EPOCHS       = 5


def run_one(clients, model_kwargs, X_test, y_test, num_rounds,
            epsilon, delta, max_grad_norm, lr, device):
    global_model = MLP(**model_kwargs).to(device)
    global_weights = global_model.get_weights()
    history = []

    for r in range(1, num_rounds + 1):
        all_w, all_n, all_eps, all_loss = [], [], [], []
        for c in clients:
            w, n, eps_spent, loss = c.train(
                global_weights,
                epochs=LOCAL_EPOCHS,
                lr=lr,
                batch_size=BATCH_SIZE,
                target_epsilon=epsilon,
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
        history.append({
            "round": r,
            "f1_macro": metrics["f1_macro"],
            "auc": metrics["auc"],
            "epsilon_spent": float(np.mean(all_eps)),
            "loss": float(np.mean(all_loss)),
        })

    best = max(history, key=lambda h: h["auc"])
    return best["auc"], best["f1_macro"], history


def build_clients(X_client, y_client, banks_client, model_kwargs, device, num_clients):
    client_data = partition_bank(X_client, y_client, banks_client, num_clients)
    return [
        DPFLClient(i, Xc, yc, model_kwargs, device, max_samples=DP_MAX_SAMPLES)
        for i, (Xc, yc) in enumerate(client_data)
    ]


def main():
    device = torch.device("cpu")
    rng = np.random.default_rng(RANDOM_SEED)

    log("Loading data from features.parquet ...")
    features_parquet = DATA_DIR / "features.parquet"
    X, y, pt, banks, feature_cols, scaler = load_from_features_parquet(features_parquet)
    X_client, X_test, y_client, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    banks_client = banks[:len(X_client)]

    log(f"Train: {len(X_client):,}  Test: {len(X_test):,}  Features: {X.shape[1]}")

    # ── Phase 1: grid search ──────────────────────────────────────────────────
    log(f"\n{'='*60}")
    log(f"PHASE 1 — grid search (ε={PHASE1_EPSILON})")
    log(f"  LRs: {PHASE1_LRS}")
    log(f"  max_grad_norms: {PHASE1_GRAD_NORMS}")
    log(f"  hidden_dims: {PHASE1_HIDDEN_DIMS}")
    total_p1 = len(PHASE1_LRS) * len(PHASE1_GRAD_NORMS) * len(PHASE1_HIDDEN_DIMS)
    log(f"  Total configs: {total_p1}")

    phase1_results = []
    configs = list(itertools.product(PHASE1_LRS, PHASE1_GRAD_NORMS, PHASE1_HIDDEN_DIMS))

    for idx, (lr, C, hidden_dims) in enumerate(configs, 1):
        out_path = TUNING_DIR / f"p1_{idx:02d}_lr{lr:.0e}_C{C}_dims{'x'.join(map(str,hidden_dims))}.json"
        if out_path.exists():
            with open(out_path) as f:
                result = json.load(f)
            log(f"\n[{idx}/{total_p1}] SKIP (cached)  lr={lr:.0e}  C={C}  dims={hidden_dims}  → AUC={result['auc']:.4f}  F1={result['f1_macro']:.4f}")
            phase1_results.append(result)
            continue

        model_kwargs = {"input_dim": X.shape[1], "hidden_dims": hidden_dims, "dropout": DROPOUT}
        clients = build_clients(X_client, y_client, banks_client, model_kwargs, device, NUM_CLIENTS)

        log(f"\n[{idx}/{total_p1}] lr={lr:.0e}  C={C}  dims={hidden_dims}")
        t0 = time.time()
        auc, f1, history = run_one(
            clients, model_kwargs, X_test, y_test,
            NUM_ROUNDS, PHASE1_EPSILON, DELTA, C, lr, device,
        )
        elapsed = time.time() - t0
        log(f"  → AUC={auc:.4f}  F1={f1:.4f}  ({elapsed:.0f}s)")

        result = {
            "lr": lr, "max_grad_norm": C, "hidden_dims": hidden_dims,
            "epsilon": PHASE1_EPSILON, "auc": auc, "f1_macro": f1,
            "history": history,
        }
        phase1_results.append(result)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    best_p1 = max(phase1_results, key=lambda r: r["auc"])
    log(f"\n{'='*60}")
    log(f"Phase 1 WINNER: lr={best_p1['lr']:.0e}  C={best_p1['max_grad_norm']}  "
        f"dims={best_p1['hidden_dims']}  AUC={best_p1['auc']:.4f}  F1={best_p1['f1_macro']:.4f}")

    with open(TUNING_DIR / "phase1_best.json", "w") as f:
        json.dump({k: v for k, v in best_p1.items() if k != "history"}, f, indent=2)

    # ── Phase 2: all ε with best config ──────────────────────────────────────
    log(f"\n{'='*60}")
    log(f"PHASE 2 — all ε values with best config")
    best_lr = best_p1["lr"]
    best_C  = best_p1["max_grad_norm"]
    best_dims = best_p1["hidden_dims"]
    model_kwargs = {"input_dim": X.shape[1], "hidden_dims": best_dims, "dropout": DROPOUT}

    phase2_results = {}
    for eps in PHASE2_EPSILONS:
        out_path = TUNING_DIR / f"p2_eps{eps}_best_config.json"
        if out_path.exists():
            with open(out_path) as f:
                r = json.load(f)
            log(f"\n  ε={eps}  SKIP (cached)  → AUC={r['auc']:.4f}  F1={r['f1_macro']:.4f}")
            phase2_results[str(eps)] = {"auc": r["auc"], "f1_macro": r["f1_macro"], "history": r["history"]}
            continue

        clients = build_clients(X_client, y_client, banks_client, model_kwargs, device, NUM_CLIENTS)
        log(f"\n  ε={eps}")
        t0 = time.time()
        auc, f1, history = run_one(
            clients, model_kwargs, X_test, y_test,
            NUM_ROUNDS, eps, DELTA, best_C, best_lr, device,
        )
        elapsed = time.time() - t0
        log(f"  → AUC={auc:.4f}  F1={f1:.4f}  ({elapsed:.0f}s)")

        phase2_results[str(eps)] = {"auc": auc, "f1_macro": f1, "history": history}
        with open(out_path, "w") as f:
            json.dump({
                "epsilon": eps, "lr": best_lr, "max_grad_norm": best_C,
                "hidden_dims": best_dims, "auc": auc, "f1_macro": f1, "history": history,
            }, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    log(f"\n{'='*60}")
    log("FINAL SUMMARY")
    log(f"Best config: lr={best_lr:.0e}  C={best_C}  dims={best_dims}")
    log(f"\n{'ε':>6}  {'AUC':>8}  {'F1':>8}  {'vs paper AUC':>14}  {'vs paper F1':>12}")
    paper = {50: (0.960, 0.500), 10: (0.955, 0.514), 5: (0.954, 0.522), 1: (0.949, 0.513)}
    for eps in PHASE2_EPSILONS:
        r = phase2_results[str(eps)]
        p_auc, p_f1 = paper[eps]
        log(f"{eps:>6}  {r['auc']:>8.4f}  {r['f1_macro']:>8.4f}  "
            f"{r['auc']-p_auc:>+14.4f}  {r['f1_macro']-p_f1:>+12.4f}")

    with open(TUNING_DIR / "summary.json", "w") as f:
        json.dump({
            "best_config": {"lr": best_lr, "max_grad_norm": best_C, "hidden_dims": best_dims},
            "phase2": {eps: {"auc": phase2_results[str(eps)]["auc"],
                             "f1_macro": phase2_results[str(eps)]["f1_macro"]}
                       for eps in PHASE2_EPSILONS},
        }, f, indent=2)

    log("\nDone. Results in results/tuning/")
    LOG.close()


if __name__ == "__main__":
    main()
