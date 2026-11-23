"""Experiment 5 — Cross-dataset Validation on LI-Small.

Evaluates all saved models on LI-Small (~1% illicit ratio vs HI-Small's ~5%).

Usage:
    python experiments/exp5_li_small.py \\
        --train data/HI-Small_Trans.csv \\
        --test data/LI-Small_Trans.csv \\
        --patterns-test data/LI-Small_Patterns.txt \\
        --epsilon 5
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, RESULTS_DIR, RANDOM_SEED, EPSILON_LIST, HIDDEN_DIMS, DROPOUT
from preprocessing.utils import (
    parse_patterns_txt, assign_pattern_labels, engineer_features, get_feature_columns
)
from models.mlp import MLP
from sklearn.preprocessing import StandardScaler


def prepare_test_dataset(trans_path: Path, patterns_path: Path, scaler, feature_cols: list, nrows=None):
    df = pd.read_csv(trans_path, nrows=nrows, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = assign_pattern_labels(df, parse_patterns_txt(patterns_path))
    df, _ = engineer_features(df)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    X = df[feature_cols].values.astype(np.float32)
    y = df["Is Laundering"].values.astype(int)
    pt = df["pattern_type"].values if "pattern_type" in df.columns else np.full(len(y), "none")
    X = scaler.transform(X)
    return X, y, pt


def evaluate(model, X, y, pt, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    f1 = f1_score(y, preds, average="macro", zero_division=0)
    auc = roc_auc_score(y, probs[:, 1]) if len(np.unique(y)) == 2 else 0.0

    pattern_f1 = {}
    for p in np.unique(pt):
        if p in ("none", "unknown"):
            continue
        mask = pt == p
        if mask.sum() < 2:
            continue
        pattern_f1[p] = float(f1_score(y[mask], preds[mask], average="macro", zero_division=0))

    return {"f1_macro": float(f1), "auc": float(auc), "pattern_f1": pattern_f1}


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset validation on LI-Small.")
    parser.add_argument("--train", default=str(DATA_DIR / "HI-Small_Trans.csv"))
    parser.add_argument("--patterns-train", default=str(DATA_DIR / "HI-Small_Patterns.txt"))
    parser.add_argument("--test", default=str(DATA_DIR / "LI-Small_Trans.csv"))
    parser.add_argument("--patterns-test", default=str(DATA_DIR / "LI-Small_Patterns.txt"))
    parser.add_argument("--epsilon", type=float, nargs="+", default=EPSILON_LIST)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Fitting scaler on HI-Small...")
    df_train = pd.read_csv(args.train, nrows=args.nrows, low_memory=False)
    df_train.columns = [c.strip() for c in df_train.columns]
    df_train["Timestamp"] = pd.to_datetime(df_train["Timestamp"])
    df_train = assign_pattern_labels(df_train, parse_patterns_txt(Path(args.patterns_train)))
    df_train, _ = engineer_features(df_train)
    feature_cols = get_feature_columns(df_train)
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].values.astype(np.float32))
    del df_train

    print("Preparing LI-Small test set...")
    X_test, y_test, pt_test = prepare_test_dataset(
        Path(args.test), Path(args.patterns_test),
        scaler, feature_cols, nrows=args.nrows,
    )
    print(f"LI-Small: {len(X_test):,} samples  illicit={y_test.mean():.2%}")

    model_kwargs = {"input_dim": len(feature_cols), "hidden_dims": HIDDEN_DIMS, "dropout": DROPOUT}
    model_files = {
        "Centralized MLP (ε=∞)": RESULTS_DIR / "models" / "centralized_mlp.pt",
        "FedAvg (ε=∞)": RESULTS_DIR / "models" / "fedavg.pt",
    }
    for eps in args.epsilon:
        tag = str(eps).replace(".", "_")
        model_files[f"DP-FL (ε={eps})"] = RESULTS_DIR / "models" / f"dp_fl_eps{tag}.pt"

    all_results = {}
    for name, path in model_files.items():
        if not path.exists():
            print(f"  [skip] {name}")
            continue
        model = MLP(**model_kwargs).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        m = evaluate(model, X_test, y_test, pt_test, device)
        print(f"  {name:30s}  F1={m['f1_macro']:.4f}  AUC={m['auc']:.4f}")
        all_results[name] = m

    out = RESULTS_DIR / "tables" / "exp5_li_small.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults → {out}")


if __name__ == "__main__":
    main()
