"""SHAP feature importance stability under DP noise — produces Fig 4.

Computes KernelSHAP for each saved model and plots mean(|SHAP|) per feature.
Also computes several stability metrics between the centralised model and
each FedAvg/DP-FL variant, going beyond a single top-10 overlap ratio per
reviewer request (point 8 of the real post-acceptance review): top-k overlap
at k in {3,5,10}, Spearman/Kendall rank correlation over the full feature
ranking, cosine similarity of the mean(|SHAP|) vectors, a per-instance local
consistency score, a bootstrap CI over explained instances, a breakdown by
true-positive / false-negative / legitimate subsets, and variance across the
three seeded checkpoints (18/19/20) saved by exp2/exp3 where available.

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
from scipy.stats import spearmanr, kendalltau
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

SEEDS = [18, 19, 20]
BOOTSTRAP_N = 500
BOOTSTRAP_CI = 0.95


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


def _unwrap(vals):
    """Return a 2-D [n_samples, n_features] SHAP array."""
    if isinstance(vals, list):
        vals = vals[0]
    vals = np.array(vals)
    if vals.ndim == 3 and vals.shape[-1] == 1:
        vals = vals[:, :, 0]
    return vals


def topk_overlap(shap_a, shap_b, top_k: int = 10, idx=None) -> float:
    """Szymkiewicz-Simpson overlap ratio of top-k features by mean |SHAP|
    (NOT Jaccard: both sets have size k by construction, so the natural
    normaliser is k, not |union|). `idx` optionally restricts to a subset
    of instances (for the TP/FN/legit breakdown and bootstrap resampling)."""
    a, b = _unwrap(shap_a), _unwrap(shap_b)
    if idx is not None:
        a, b = a[idx], b[idx]
    top_a = set(int(i) for i in np.argsort(np.abs(a).mean(axis=0))[-top_k:])
    top_b = set(int(i) for i in np.argsort(np.abs(b).mean(axis=0))[-top_k:])
    return len(top_a & top_b) / top_k


def rank_correlations(shap_a, shap_b, idx=None) -> dict:
    """Spearman and Kendall correlation over the FULL feature ranking
    (not just top-k), computed on mean |SHAP| per feature. Unlike top-k
    overlap, this is sensitive to reordering within the top-k set and to
    the ranking of the remaining features."""
    a, b = _unwrap(shap_a), _unwrap(shap_b)
    if idx is not None:
        a, b = a[idx], b[idx]
    mean_a = np.abs(a).mean(axis=0)
    mean_b = np.abs(b).mean(axis=0)
    rho, _ = spearmanr(mean_a, mean_b)
    tau, _ = kendalltau(mean_a, mean_b)
    return {"spearman": float(rho), "kendall": float(tau)}


def cosine_similarity(shap_a, shap_b, idx=None) -> float:
    """Cosine similarity of the mean |SHAP| vectors -- sensitive to
    magnitude, not just ranking, unlike top-k overlap or rank correlation."""
    a, b = _unwrap(shap_a), _unwrap(shap_b)
    if idx is not None:
        a, b = a[idx], b[idx]
    mean_a = np.abs(a).mean(axis=0)
    mean_b = np.abs(b).mean(axis=0)
    denom = np.linalg.norm(mean_a) * np.linalg.norm(mean_b)
    return float(np.dot(mean_a, mean_b) / denom) if denom > 0 else float("nan")


def local_consistency(shap_a, shap_b) -> float:
    """Per-instance local explanation consistency: for each explained
    transaction, the Spearman correlation between its two feature-
    attribution vectors (one per model), averaged over instances. This is
    a per-transaction check, distinct from the aggregate (mean-|SHAP|)
    metrics above, which only compare feature RANKINGS averaged across
    all explained instances."""
    a, b = _unwrap(shap_a), _unwrap(shap_b)
    n = min(len(a), len(b))
    rhos = []
    for i in range(n):
        rho, _ = spearmanr(a[i], b[i])
        if not np.isnan(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else float("nan")


def bootstrap_ci(shap_a, shap_b, metric_fn, n_boot: int = BOOTSTRAP_N, ci: float = BOOTSTRAP_CI, seed: int = RANDOM_SEED):
    """Bootstrap CI for a metric over the explained-instance axis (resample
    instances with replacement, recompute the metric each time). Addresses
    the reviewer's "KernelSHAP on only 200 transactions may have
    considerable sampling variance" point directly, rather than reporting
    a single point estimate."""
    a = _unwrap(shap_a)
    n = len(a)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(metric_fn(shap_a, shap_b, idx=idx))
    lo = float(np.percentile(vals, (1 - ci) / 2 * 100))
    hi = float(np.percentile(vals, (1 + ci) / 2 * 100))
    return {"point": float(metric_fn(shap_a, shap_b)), "ci_lo": lo, "ci_hi": hi, "n_boot": n_boot}


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


def discover_checkpoints(base_name: str) -> dict:
    """Return {seed: path} for every existing seed-suffixed checkpoint
    named `{base_name}_seed{seed}.pt`, falling back to the single
    un-suffixed `{base_name}.pt` (seed=None) if no seeded files exist yet
    (pre-camera-ready result directories)."""
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
    parser = argparse.ArgumentParser(description="SHAP analysis under DP-FL.")
    parser.add_argument("--data", default=str(DATA_DIR / "HI-Small_Trans.csv"))
    parser.add_argument("--patterns", default=str(DATA_DIR / "HI-Small_Patterns.txt"))
    parser.add_argument("--epsilon", type=float, nargs="+", default=EPSILON_LIST)
    parser.add_argument("--nrows", type=int, default=50000)
    parser.add_argument("--n-explain", type=int, default=300)
    parser.add_argument("--fast", action="store_true",
                        help="Load from pre-computed features.parquet instead of raw CSV.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for the TP/FN/legit breakdown "
                             "(a coarse 0.5 default; this breakdown is diagnostic, "
                             "not the paper's reported operating point).")
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
    y_explain = y_test[explain_idx]

    model_kwargs = {"input_dim": X.shape[1], "hidden_dims": HIDDEN_DIMS, "dropout": DROPOUT}

    # {family_name: {seed_or_None: checkpoint_path}}
    model_families = {
        "Centralized": {None: RESULTS_DIR / "models" / "centralized_mlp.pt"},
        "FedAvg": discover_checkpoints("fedavg"),
    }
    for eps in sorted(args.epsilon, reverse=True):
        tag = str(float(eps)).replace(".", "_")
        model_families[f"DP-FL ε={eps}"] = discover_checkpoints(f"dp_fl_eps{tag}")

    # Compute SHAP for every (family, seed) checkpoint that exists.
    shap_by_family = {}   # family -> {seed_or_None: shap_vals}
    pred_by_family = {}   # family -> {seed_or_None: binary predictions on explain set}
    for family, seed_paths in model_families.items():
        if not seed_paths:
            print(f"  [skip] {family} — no checkpoint found")
            continue
        shap_by_family[family] = {}
        pred_by_family[family] = {}
        for seed, path in seed_paths.items():
            tag = f"{family} (seed {seed})" if seed is not None else family
            print(f"Computing SHAP: {tag}  ({len(X_explain)} samples)...")
            model = MLP(**model_kwargs).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            vals = compute_shap(model, X_test, X_explain, device)
            shap_by_family[family][seed] = vals
            probs = predict_proba(model, X_explain, device)[:, 1]
            pred_by_family[family][seed] = (probs >= args.threshold).astype(int)

    if "Centralized" not in shap_by_family:
        print("[error] Centralized SHAP values are required as the reference model; aborting.")
        return
    ref_shap = shap_by_family["Centralized"][None]
    ref_pred = pred_by_family["Centralized"][None]

    # TP / FN / legitimate index groups, defined against the Centralized
    # model's own predictions (a fixed reference grouping applied to every
    # family, so groups are comparable across models).
    tp_idx = np.where((y_explain == 1) & (ref_pred == 1))[0]
    fn_idx = np.where((y_explain == 1) & (ref_pred == 0))[0]
    legit_idx_local = np.where(y_explain == 0)[0]
    print(f"Reference-model groups: TP={len(tp_idx)}  FN={len(fn_idx)}  legit={len(legit_idx_local)}")

    stability = {}
    for family, seed_shaps in shap_by_family.items():
        if family == "Centralized":
            continue
        per_seed_metrics = []
        for seed, vals in seed_shaps.items():
            m = {
                "seed": seed,
                "top3": topk_overlap(ref_shap, vals, top_k=3),
                "top5": topk_overlap(ref_shap, vals, top_k=5),
                "top10": topk_overlap(ref_shap, vals, top_k=10),
                **rank_correlations(ref_shap, vals),
                "cosine": cosine_similarity(ref_shap, vals),
                "local_consistency": local_consistency(ref_shap, vals),
            }
            per_seed_metrics.append(m)
            print(f"  {family} (seed {seed}): top10={m['top10']:.2f} "
                  f"spearman={m['spearman']:.2f} kendall={m['kendall']:.2f} "
                  f"cosine={m['cosine']:.2f} local={m['local_consistency']:.2f}")

        keys = ["top3", "top5", "top10", "spearman", "kendall", "cosine", "local_consistency"]
        agg = {f"{k}_mean": float(np.mean([m[k] for m in per_seed_metrics])) for k in keys}
        agg.update({f"{k}_std": float(np.std([m[k] for m in per_seed_metrics])) for k in keys})

        # Bootstrap CI and TP/FN/legit breakdown on the first available seed only
        # (compute-bounded; the cross-seed std above already covers run-to-run
        # variance, so this adds within-run sampling variance on one run).
        first_seed = next(iter(seed_shaps))
        first_vals = seed_shaps[first_seed]
        agg["top10_bootstrap_ci"] = bootstrap_ci(ref_shap, first_vals, lambda a, b, idx=None: topk_overlap(a, b, top_k=10, idx=idx))
        agg["by_group"] = {
            "true_positive": topk_overlap(ref_shap, first_vals, top_k=10, idx=tp_idx) if len(tp_idx) >= 2 else None,
            "false_negative": topk_overlap(ref_shap, first_vals, top_k=10, idx=fn_idx) if len(fn_idx) >= 2 else None,
            "legitimate": topk_overlap(ref_shap, first_vals, top_k=10, idx=legit_idx_local) if len(legit_idx_local) >= 2 else None,
        }
        agg["per_seed"] = per_seed_metrics
        stability[family] = agg

    # Plot using the first available seed's SHAP per family (consistent
    # with exp4's convention of reporting the first-seed checkpoint as the
    # representative run; cross-seed variance lives in the JSON instead).
    plot_dict = {"Centralized": ref_shap}
    for family, seed_shaps in shap_by_family.items():
        if family == "Centralized":
            continue
        plot_dict[family] = next(iter(seed_shaps.values()))
    plot_shap_comparison(plot_dict, feature_cols, RESULTS_DIR / "figures" / "fig4_shap_comparison.png")

    save_np = {}
    for family, seed_shaps in shap_by_family.items():
        for seed, vals in seed_shaps.items():
            key = family.replace(" ", "_").replace("=", "").replace(".", "")
            if seed is not None:
                key += f"_seed{seed}"
            save_np[key] = vals
    np.savez(RESULTS_DIR / "tables" / "shap_values.npz",
             feature_names=np.array(feature_cols), **save_np)

    out = RESULTS_DIR / "tables" / "shap_stability.json"
    with open(out, "w") as f:
        json.dump(stability, f, indent=2, default=lambda o: None)
    print(f"Stability scores → {out}")


if __name__ == "__main__":
    main()
