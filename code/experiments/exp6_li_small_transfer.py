"""Experiment 6 — Zero-shot Cross-dataset Transfer Validation on LI-Small.

Evaluates all HI-Small-trained models on LI-Small without retraining.
Loads saved scaler.pkl and feature_names.json — no HI-Small re-read needed.

Usage:
    python experiments/exp6_li_small_transfer.py [--device cpu|cuda]
"""

import sys
import json
import time
import joblib
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, RESULTS_DIR, HIDDEN_DIMS, DROPOUT, EPSILON_LIST
from preprocessing.utils import parse_patterns_txt, engineer_features
from models.mlp import MLP


def log(msg, pct=None):
    tag = f"  [{pct:3d}%]" if pct is not None else ""
    print(f"[{time.strftime('%H:%M:%S')}]{tag} {msg}", flush=True)


def prepare_li_small(trans_path, patterns_path, scaler, feature_cols):
    log("Loading LI-Small_Trans.csv ...", pct=5)
    df = pd.read_csv(trans_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    log(f"Loaded {len(df):,} rows  illicit={df['Is Laundering'].mean():.4%}", pct=25)

    log("Parsing LI-Small_Patterns.txt ...", pct=30)
    account_to_pattern = parse_patterns_txt(patterns_path)
    log(f"{len(account_to_pattern):,} accounts mapped to pattern types", pct=35)

    # Vectorized pattern assignment (avoids slow df.apply on 6.9M rows)
    log("Assigning pattern labels (vectorized) ...", pct=38)
    illicit_mask = df["Is Laundering"] == 1
    pattern_col = pd.Series("none", index=df.index, dtype=object)
    from_map = df.loc[illicit_mask, "Account"].astype(str).map(account_to_pattern)
    to_map   = df.loc[illicit_mask, "Account.1"].astype(str).map(account_to_pattern)
    pattern_col[illicit_mask] = from_map.combine_first(to_map).fillna("unknown")
    df["pattern_type"] = pattern_col
    log(f"Pattern breakdown: {df.loc[illicit_mask, 'pattern_type'].value_counts().to_dict()}", pct=42)

    log("Engineering features ...", pct=45)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df, _ = engineer_features(df)
    log("Feature engineering complete", pct=68)

    log("Scaling with saved HI-Small scaler ...", pct=70)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    X = df[feature_cols].values.astype(np.float32)
    y = df["Is Laundering"].values.astype(int)
    pt = df["pattern_type"].values
    X = scaler.transform(X)
    log(f"Ready: X={X.shape}  illicit={y.sum():,}/{len(y):,} ({y.mean():.4%})", pct=78)
    return X, y, pt


def threshold_opt_f1(y_true, probs_pos):
    """Scan PR curve to find threshold maximising macro-F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, probs_pos)
    denom = precision + recall
    f1s = np.where(denom == 0, 0.0, 2 * precision * recall / (denom + 1e-9))
    best_idx = int(np.argmax(f1s[:-1]))  # last entry has no threshold
    best_thr = float(thresholds[best_idx])
    preds = (probs_pos >= best_thr).astype(int)
    return float(f1_score(y_true, preds, average="macro", zero_division=0)), best_thr


def evaluate_model(model, X, y, pt, device, name, step_pct):
    log(f"Evaluating: {name}", pct=step_pct)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    auc = float(roc_auc_score(y, probs[:, 1])) if len(np.unique(y)) == 2 else 0.0
    f1, thr = threshold_opt_f1(y, probs[:, 1])

    preds = (probs[:, 1] >= thr).astype(int)
    pattern_f1 = {}
    for p in np.unique(pt):
        if p in ("none", "unknown"):
            continue
        mask = pt == p
        if mask.sum() < 2:
            continue
        pattern_f1[p] = float(f1_score(y[mask], preds[mask], average="macro", zero_division=0))

    log(f"  AUC={auc:.4f}  F1={f1:.4f}  thr={thr:.4f}  patterns={len(pattern_f1)}", pct=step_pct)
    return {"auc": auc, "f1_macro": f1, "threshold": thr, "pattern_f1": pattern_f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    log(f"=== Exp6: LI-Small Zero-shot Transfer  device={device} ===", pct=0)

    log("Loading scaler.pkl ...", pct=1)
    scaler = joblib.load(DATA_DIR / "scaler.pkl")
    with open(DATA_DIR / "feature_names.json") as f:
        feature_cols = json.load(f)
    log(f"{len(feature_cols)} features: {feature_cols}", pct=3)

    X, y, pt = prepare_li_small(
        DATA_DIR / "LI-Small_Trans.csv",
        DATA_DIR / "LI-Small_Patterns.txt",
        scaler, feature_cols,
    )

    # Model registry — same naming convention as exp3
    model_files = {"Centralized MLP": RESULTS_DIR / "models" / "centralized_mlp.pt",
                   "FedAvg (no DP)":  RESULTS_DIR / "models" / "fedavg.pt"}
    for eps in sorted(EPSILON_LIST, reverse=True):
        tag = str(float(eps)).replace(".", "_")
        model_files[f"DP-FL (ε={eps})"] = RESULTS_DIR / "models" / f"dp_fl_eps{tag}.pt"

    results = {}
    entries = list(model_files.items())
    for i, (name, path) in enumerate(entries):
        pct = 80 + int(18 * i / len(entries))
        if not path.exists():
            log(f"[skip] {name} — {path.name} not found", pct=pct)
            continue
        sd = torch.load(path, map_location=device, weights_only=True)
        inferred_dims = [sd["net.0.weight"].shape[0], sd["net.3.weight"].shape[0], sd["net.6.weight"].shape[0]]
        model = MLP(input_dim=len(feature_cols), hidden_dims=inferred_dims, dropout=DROPOUT).to(device)
        model.load_state_dict(sd)
        results[name] = evaluate_model(model, X, y, pt, device, name, pct)

    out = RESULTS_DIR / "tables" / "exp6_li_small.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved → {out}", pct=100)
    log("=== Done ===", pct=100)

    # Print summary table
    print("\n--- Summary ---")
    print(f"{'Model':<25}  {'AUC':>6}  {'F1':>6}")
    for name, m in results.items():
        print(f"{name:<25}  {m['auc']:.4f}  {m['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
