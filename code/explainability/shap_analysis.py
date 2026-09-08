"""SHAP feature importance stability under DP noise — produces Fig 4.

Computes KernelSHAP for each saved model and plots mean(|SHAP|) per feature.
Also computes a top-k stability score: fraction of top-k SHAP features
shared between Centralized and each DP-FL model.

Usage:
    python explainability/shap_analysis.py \\
        --data data/HI-Small_Trans.csv \\
        --patterns data/HI-Small_Patterns.txt \\
        --epsilon 1 5 10 50 \\
        --n-explain 300
"""

import sys
import math
import json
import argparse
import numpy as np
import torch
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Black/white academic style
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#cccccc",
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "text.color": "black",
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
})

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_DIR, RESULTS_DIR, RANDOM_SEED, TEST_SIZE, EPSILON_LIST,
    HIDDEN_DIMS, DROPOUT,
)
from preprocessing.utils import load_and_prepare, load_from_features_parquet, grouped_train_test_split
from models.mlp import MLP


def predict_proba(model, X: np.ndarray, device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device))
        return torch.softmax(logits, dim=1).cpu().numpy()


def compute_shap(model, X_background: np.ndarray, X_explain: np.ndarray, device, n_bg=100, nsamples=80):
    bg = shap.sample(X_background, n_bg)

    def f(x):
        return predict_proba(model, x, device)[:, 1:2]

    explainer = shap.KernelExplainer(f, bg)
    return explainer.shap_values(X_explain, nsamples=nsamples)


def plot_shap_comparison(shap_dict: dict, feature_names: list, out_path: Path, top_k: int = 12):
    valid = {k: v for k, v in shap_dict.items() if v is not None}
    n = len(valid)
    if n == 0:
        print("[warn] No SHAP values to plot")
        return

    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.2 * nrows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (name, vals) in zip(axes_flat, valid.items()):
        mean_abs = np.abs(_unwrap(vals)).mean(axis=0)
        idx = np.argsort(mean_abs)[-top_k:][::-1]
        ax.barh(range(top_k), mean_abs[idx], color="black", alpha=0.75,
                edgecolor="black", linewidth=0.4)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([feature_names[i] for i in idx], fontsize=8)
        ax.set_xlabel("mean(|SHAP value|)", fontsize=8)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.invert_yaxis()

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("SHAP Feature Importance — Centralized vs FedAvg vs DP-FL", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Fig 4 saved → {out_path}")


def _unwrap(vals):
    """Return a 2-D [n_samples, n_features] SHAP array.

    KernelExplainer returns (n_samples, n_features, 1) for a single-output
    function or a list of such arrays for multi-output. Squeeze last dim.
    """
    if isinstance(vals, list):
        vals = vals[0]
    vals = np.array(vals)
    if vals.ndim == 3 and vals.shape[-1] == 1:
        vals = vals[:, :, 0]
    return vals


def stability_score(shap_a, shap_b, top_k: int = 10) -> float:
    """Jaccard overlap of top-k features by mean |SHAP|."""
    a, b = _unwrap(shap_a), _unwrap(shap_b)
    top_a = set(int(i) for i in np.argsort(np.abs(a).mean(axis=0))[-top_k:])
    top_b = set(int(i) for i in np.argsort(np.abs(b).mean(axis=0))[-top_k:])
    return len(top_a & top_b) / top_k


def main():
    parser = argparse.ArgumentParser(description="SHAP analysis under DP-FL.")
    parser.add_argument("--data", default=str(DATA_DIR / "HI-Small_Trans.csv"))
    parser.add_argument("--patterns", default=str(DATA_DIR / "HI-Small_Patterns.txt"))
    parser.add_argument("--epsilon", type=float, nargs="+", default=EPSILON_LIST)
    parser.add_argument("--nrows", type=int, default=50000)
    parser.add_argument("--n-explain", type=int, default=300)
    parser.add_argument("--fast", action="store_true",
                        help="Load from pre-computed features.parquet instead of raw CSV.")
    parser.add_argument("--device", default="cpu")
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
    _, X_test, _, y_test = grouped_train_test_split(X, y, groups=groups, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    rng = np.random.default_rng(RANDOM_SEED)
    launder_idx = np.where(y_test == 1)[0]
    legit_idx = np.where(y_test == 0)[0]
    n_each = min(args.n_explain // 2, len(launder_idx), len(legit_idx))
    explain_idx = np.concatenate([
        rng.choice(launder_idx, n_each, replace=False),
        rng.choice(legit_idx, n_each, replace=False),
    ])
    X_explain = X_test[explain_idx]

    model_kwargs = {"input_dim": X.shape[1], "hidden_dims": HIDDEN_DIMS, "dropout": DROPOUT}
    model_files = {
        "Centralized": RESULTS_DIR / "models" / "centralized_mlp.pt",
        "FedAvg": RESULTS_DIR / "models" / "fedavg.pt",
    }
    for eps in sorted(args.epsilon, reverse=True):
        # Match exp3_dp_fl.py's float-derived save tag (e.g. 50 -> "50_0")
        # regardless of whether eps arrived as an int (config default) or a
        # CLI-parsed float.
        tag = str(float(eps)).replace(".", "_")
        model_files[f"DP-FL ε={eps}"] = RESULTS_DIR / "models" / f"dp_fl_eps{tag}.pt"

    shap_dict = {}
    ref_shap = None
    stability = {}

    for name, path in model_files.items():
        if not path.exists():
            print(f"  [skip] {name}")
            shap_dict[name] = None
            continue
        print(f"Computing SHAP: {name}  ({len(X_explain)} samples)...")
        model = MLP(**model_kwargs).to(device)
        model.load_state_dict(torch.load(path, map_location=device))

        vals = compute_shap(model, X_test, X_explain, device)
        shap_dict[name] = vals

        if ref_shap is None:
            ref_shap = vals
        else:
            score = stability_score(ref_shap, vals)
            stability[name] = float(score)
            print(f"  Stability vs Centralized: {score:.2%}")

    plot_shap_comparison(shap_dict, feature_cols, RESULTS_DIR / "figures" / "fig4_shap_comparison.png")

    save_np = {k.replace(" ", "_").replace("=", "").replace(".", ""): v
               for k, v in shap_dict.items() if v is not None}
    np.savez(RESULTS_DIR / "tables" / "shap_values.npz",
             feature_names=np.array(feature_cols), **save_np)

    out = RESULTS_DIR / "tables" / "shap_stability.json"
    with open(out, "w") as f:
        json.dump(stability, f, indent=2)
    print(f"Stability scores → {out}")


if __name__ == "__main__":
    main()
