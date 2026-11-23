"""Experiment 5 — Extended metrics for severe class imbalance.

Reviewer point 4: AUC-ROC and macro-F1 "are not suitable for the
imbalance level" (0.10% illicit); a classifier predicting every
transaction legitimate can score macro-F1 near 0.5. This script adds
the metrics the reviewer asked for, computed post-hoc from already-saved
checkpoints and the already-locked validation-selected thresholds (no
retraining): PR-AUC, illicit-class precision/recall/F1 at the locked
threshold, recall at fixed false-positive rates, precision at fixed
alert budgets, false alerts per detected laundering transaction, and
calibration (Brier score + a 10-bin expected calibration error).

Requires: checkpoints + per_seed result JSONs from exp1/exp2/exp2b/exp3.

Usage:
    python experiments/exp5_extended_metrics.py --fast
"""

import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, RESULTS_DIR, RANDOM_SEED, TEST_SIZE, VAL_SIZE, HIDDEN_DIMS, DROPOUT, NUM_CLIENTS
from preprocessing.utils import load_and_prepare, load_from_features_parquet, grouped_train_test_split, select_f1_threshold
from models.mlp import MLP

SEEDS = [18, 19, 20]
FPR_TARGETS = [0.01, 0.05, 0.10]
ALERT_BUDGETS = [100, 500, 1000]


def predict_proba(model, X, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device))
        return torch.softmax(logits, dim=1).cpu().numpy()[:, 1]


def recall_at_fpr(y, scores, fpr_target):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, scores)
    valid = fpr <= fpr_target
    return float(tpr[valid].max()) if valid.any() else 0.0


def precision_at_k(y, scores, k):
    order = np.argsort(-scores)[:k]
    return float(y[order].mean()) if k > 0 else float("nan")


def calibration_error(y, probs, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc = y[mask].mean()
        ece += (mask.sum() / len(y)) * abs(bin_conf - bin_acc)
    return float(ece)


def score_model(model, X_test, y_test, device, threshold):
    from sklearn.metrics import (
        average_precision_score, precision_score, recall_score, f1_score,
        roc_auc_score, brier_score_loss, confusion_matrix,
    )
    scores = predict_proba(model, X_test, device)
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()

    out = {
        "threshold": float(threshold),
        "auc_roc": float(roc_auc_score(y_test, scores)) if len(np.unique(y_test)) == 2 else None,
        "pr_auc": float(average_precision_score(y_test, scores)),
        "precision_illicit": float(precision_score(y_test, preds, pos_label=1, zero_division=0)),
        "recall_illicit": float(recall_score(y_test, preds, pos_label=1, zero_division=0)),
        "f1_illicit": float(f1_score(y_test, preds, pos_label=1, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "false_alerts_per_detected": float(fp / tp) if tp > 0 else float("inf"),
        "brier_score": float(brier_score_loss(y_test, scores)),
        "calibration_error_10bin": calibration_error(y_test, scores),
    }
    for fpr_t in FPR_TARGETS:
        out[f"recall_at_fpr{fpr_t}"] = recall_at_fpr(y_test, scores, fpr_t)
    for k in ALERT_BUDGETS:
        out[f"precision_at_top{k}"] = precision_at_k(y_test, scores, k)
    return out


def load_threshold(results_path: Path, per_client: int = None):
    if not results_path.exists():
        return None
    with open(results_path) as f:
        data = json.load(f)
    if per_client is not None:
        # exp2b_local_only.json: per_seed[0]["per_client"][per_client]["threshold"]
        return data["per_seed"][0]["per_client"][per_client]["threshold"]
    per_seed = data.get("per_seed")
    if per_seed:
        return per_seed[0]["threshold"]
    return data.get("threshold")


def discover_checkpoints(base_name: str) -> dict:
    found = {}
    for seed in SEEDS:
        p = RESULTS_DIR / "models" / f"{base_name}_seed{seed}.pt"
        if p.exists():
            found[seed] = p
    if not found:
        p = RESULTS_DIR / "models" / f"{base_name}.pt"
        if p.exists():
            found[None] = p
    return found


def main():
    parser = argparse.ArgumentParser(description="Extended imbalance-aware metrics.")
    parser.add_argument("--data", default=str(DATA_DIR / "HI-Small_Trans.csv"))
    parser.add_argument("--patterns", default=str(DATA_DIR / "HI-Small_Patterns.txt"))
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
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
    X_tv, X_test, y_tv, y_test, groups_tv, groups_test = grouped_train_test_split(
        X, y, groups, groups=groups, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    # Same val split exp1_centralized.py uses -- needed because exp1's result
    # JSON doesn't persist its val-selected MLP threshold, only f1/auc.
    _, X_val_c, _, y_val_c = grouped_train_test_split(
        X_tv, y_tv, groups=groups_tv, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=RANDOM_SEED,
    )
    print(f"Test set: {len(y_test):,} rows, illicit={y_test.mean():.4%}  "
          f"({int(y_test.sum())} illicit)")

    model_kwargs = {"input_dim": X.shape[1], "hidden_dims": HIDDEN_DIMS, "dropout": DROPOUT}
    results = {}

    # Centralized MLP
    ckpt = RESULTS_DIR / "models" / "centralized_mlp.pt"
    if ckpt.exists():
        model = MLP(**model_kwargs).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        # exp1_centralized.json doesn't persist its val-selected threshold
        # (only f1_macro/auc), so recompute it here on the same val split
        # exp1 used -- no retraining, just a forward pass + PR-curve scan.
        val_scores = predict_proba(model, X_val_c, device)
        thr = select_f1_threshold(y_val_c, val_scores)
        results["Centralized MLP"] = [score_model(model, X_test, y_test, device, thr)]

    # FedAvg (multi-seed)
    seed_paths = discover_checkpoints("fedavg")
    if seed_paths:
        res = RESULTS_DIR / "tables" / "exp2_fedavg.json"
        thr = load_threshold(res)
        runs = []
        for seed, path in seed_paths.items():
            model = MLP(**model_kwargs).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            runs.append(score_model(model, X_test, y_test, device, thr))
        results["FedAvg"] = runs

    # DP-FL per epsilon (multi-seed)
    for eps in [50, 10, 5, 1]:
        tag = str(float(eps)).replace(".", "_")
        seed_paths = discover_checkpoints(f"dp_fl_eps{tag}")
        if not seed_paths:
            continue
        res = RESULTS_DIR / "tables" / f"exp3_dp_fl_eps{tag}.json"
        thr = load_threshold(res)
        runs = []
        for seed, path in seed_paths.items():
            model = MLP(**model_kwargs).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            runs.append(score_model(model, X_test, y_test, device, thr))
        results[f"DP-FL eps={eps}"] = runs

    # Local-only (point 5a): one model per client, seed 0 only (checkpoints
    # are only saved for the first seed; cross-seed variance is already in
    # exp2b_local_only.json's per_client f1/auc, not re-derived here).
    res2b = RESULTS_DIR / "tables" / "exp2b_local_only.json"
    if res2b.exists():
        runs = []
        for i in range(NUM_CLIENTS):
            ckpt = RESULTS_DIR / "models" / f"local_only_client{i}.pt"
            if not ckpt.exists():
                continue
            thr = load_threshold(res2b, per_client=i)
            model = MLP(**model_kwargs).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            runs.append(score_model(model, X_test, y_test, device, thr))
        if runs:
            results["Local-only (mean over 5 banks)"] = runs

    # Aggregate mean/std across runs (seeds, or clients for local-only) per family.
    summary = {}
    keys = ["auc_roc", "pr_auc", "precision_illicit", "recall_illicit", "f1_illicit",
            "false_alerts_per_detected", "brier_score", "calibration_error_10bin"] \
        + [f"recall_at_fpr{t}" for t in FPR_TARGETS] + [f"precision_at_top{k}" for k in ALERT_BUDGETS]
    for family, runs in results.items():
        summary[family] = {}
        for k in keys:
            vals = [r[k] for r in runs if r.get(k) is not None and np.isfinite(r.get(k, np.nan))]
            if vals:
                summary[family][f"{k}_mean"] = float(np.mean(vals))
                summary[family][f"{k}_std"] = float(np.std(vals))
        print(f"\n{family} (n={len(runs)}):")
        print(f"  PR-AUC        = {summary[family].get('pr_auc_mean', float('nan')):.4f} ± {summary[family].get('pr_auc_std', 0):.4f}")
        print(f"  Illicit P/R/F1= {summary[family].get('precision_illicit_mean', float('nan')):.4f} / "
              f"{summary[family].get('recall_illicit_mean', float('nan')):.4f} / "
              f"{summary[family].get('f1_illicit_mean', float('nan')):.4f}")
        print(f"  FP per TP     = {summary[family].get('false_alerts_per_detected_mean', float('nan')):.2f}")
        print(f"  Recall@FPR1%  = {summary[family].get('recall_at_fpr0.01_mean', float('nan')):.4f}")
        print(f"  Brier / ECE   = {summary[family].get('brier_score_mean', float('nan')):.4f} / "
              f"{summary[family].get('calibration_error_10bin_mean', float('nan')):.4f}")

    out = RESULTS_DIR / "tables" / "exp5_extended_metrics.json"
    with open(out, "w") as f:
        json.dump({"experiment": "exp5_extended_metrics", "per_run": results, "summary": summary}, f, indent=2)
    print(f"\nResults → {out}")


if __name__ == "__main__":
    main()
