"""Experiment 1 — Centralized Baseline (MLP + XGBoost).

Usage:
    python experiments/exp1_centralized.py \\
        --data data/HI-Small_Trans.csv \\
        --patterns data/HI-Small_Patterns.txt \\
        --model mlp       # or xgboost or both
"""

import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, RESULTS_DIR, RANDOM_SEED, TEST_SIZE, VAL_SIZE, HIDDEN_DIMS, DROPOUT
from preprocessing.utils import load_and_prepare, load_from_features_parquet, grouped_train_test_split
from models.baseline import train_mlp, evaluate_model, train_xgboost, evaluate_xgboost


def main():
    parser = argparse.ArgumentParser(description="Centralized baseline experiment.")
    parser.add_argument("--data", default=str(DATA_DIR / "HI-Small_Trans.csv"))
    parser.add_argument("--patterns", default=str(DATA_DIR / "HI-Small_Patterns.txt"))
    parser.add_argument("--model", choices=["mlp", "xgboost", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--fast", action="store_true",
                        help="Load from pre-computed features.parquet instead of raw CSV.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate for MLP (default: from config).")
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)

    features_parquet = DATA_DIR / "features.parquet"
    if args.fast and features_parquet.exists():
        print(f"Fast load from {features_parquet}...")
        X, y, pt, banks, feature_cols, scaler, groups = load_from_features_parquet(features_parquet)
    else:
        X, y, pt, banks, feature_cols, scaler, groups = load_and_prepare(
            Path(args.data), Path(args.patterns), nrows=args.nrows
        )
    print(f"Dataset: {X.shape[0]:,} samples  {X.shape[1]} features  illicit={y.mean():.2%}")

    X_tv, X_test, y_tv, y_test, pt_tv, pt_test, groups_tv, groups_test = grouped_train_test_split(
        X, y, pt, groups, groups=groups, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = grouped_train_test_split(
        X_tv, y_tv,
        groups=groups_tv,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_SEED,
    )

    out = RESULTS_DIR / "tables" / "exp1_centralized.json"
    if out.exists():
        with open(out) as f:
            results = json.load(f)
    else:
        results = {"experiment": "exp1_centralized", "feature_cols": feature_cols}

    if args.model in ("mlp", "both"):
        print("\n--- Centralized MLP ---")
        model, val_f1 = train_mlp(X_train, y_train, X_val, y_val, epochs=args.epochs, device_str=args.device, lr=args.lr)
        metrics = evaluate_model(model, X_test, y_test, device_str=args.device)
        print(f"  F1={metrics['f1_macro']:.4f}  AUC={metrics['auc']:.4f}")
        results["mlp"] = {"f1_macro": metrics["f1_macro"], "auc": metrics["auc"], "epsilon": "inf"}

        pattern_f1 = {}
        for p in np.unique(pt_test):
            if p == "none":
                continue
            mask = pt_test == p
            if mask.sum() < 2:
                continue
            pattern_f1[p] = float(f1_score(y_test[mask], metrics["preds"][mask], average="macro", zero_division=0))
        results["mlp"]["pattern_f1"] = pattern_f1
        print(f"  Per-pattern F1: {pattern_f1}")

        torch.save(model.state_dict(), RESULTS_DIR / "models" / "centralized_mlp.pt")

    if args.model in ("xgboost", "both"):
        print("\n--- Centralized XGBoost ---")
        clf = train_xgboost(X_train, y_train)
        metrics_xgb = evaluate_xgboost(clf, X_test, y_test)
        print(f"  F1={metrics_xgb['f1_macro']:.4f}  AUC={metrics_xgb['auc']:.4f}")
        results["xgboost"] = {"f1_macro": metrics_xgb["f1_macro"], "auc": metrics_xgb["auc"]}

        pattern_f1_xgb = {}
        for p in np.unique(pt_test):
            if p == "none":
                continue
            mask = pt_test == p
            if mask.sum() < 2:
                continue
            pattern_f1_xgb[p] = float(f1_score(y_test[mask], metrics_xgb["preds"][mask], average="macro", zero_division=0))
        results["xgboost"]["pattern_f1"] = pattern_f1_xgb

    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {out}")


if __name__ == "__main__":
    main()
