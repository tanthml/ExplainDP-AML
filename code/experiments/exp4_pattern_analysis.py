"""Experiment 4 — Pattern-Stratified Analysis (key novelty of the paper).

Evaluates detection recall per laundering pattern type (fan-in, fan-out,
cycle, ...) under each DP noise level ε, using one shared globally-optimised
threshold. Produces Table 2 and Fig 5 from IDEAS.md.

Note: per-pattern subsets contain only illicit transactions (legitimate ones
are labeled "none"), so macro-F1 on these subsets is degenerate -- recall is
the well-defined metric here. See per_pattern_recall() docstring.

Requires: saved models from exp1 (centralized_mlp.pt), exp2 (fedavg.pt),
          and exp3 (dp_fl_eps{1,5,10,50}.pt).

Usage:
    python experiments/exp4_pattern_analysis.py \\
        --data data/HI-Small_Trans.csv \\
        --patterns data/HI-Small_Patterns.txt \\
        --epsilon 1 5 10 50
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path

# Black/white academic style
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "text.color": "black",
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
})

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_DIR, RESULTS_DIR, RANDOM_SEED, TEST_SIZE,
    EPSILON_LIST, PATTERN_TYPES, HIDDEN_DIMS, DROPOUT,
)
from preprocessing.utils import load_and_prepare, load_from_features_parquet, grouped_train_test_split
from models.mlp import MLP


def load_model(model_kwargs, path: Path, device):
    model = MLP(**model_kwargs).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def per_pattern_recall(model, X_test, y_test, pt_test, device):
    """Per-pattern detection recall under one shared, globally-optimised threshold.

    Every per-pattern subset (pt_test == pattern) contains ONLY illicit
    transactions -- legitimate transactions are always labeled "none". Macro-F1
    on a single-true-class subset is degenerate under sklearn's default label
    inference (it collapses to exactly 1.0 whenever recall==1, since only class 1
    is "present", and to roughly 0.5*F1_1 otherwise) -- it does not reflect a real
    precision/recall tradeoff because there are no true negatives to be wrong
    about within the subset. Recall (fraction of that pattern's illicit
    transactions flagged) is the metric that is actually well-defined here.
    """
    from sklearn.metrics import precision_recall_curve
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    scores = probs[:, 1]
    # Use global threshold-optimized predictions (consistent with Table 1 evaluation).
    prec, rec, thresh = precision_recall_curve(y_test, scores)
    f1_cls1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    best_idx = int(np.argmax(f1_cls1))
    best_thresh = float(thresh[best_idx]) if best_idx < len(thresh) else 0.5
    preds = (scores >= best_thresh).astype(int)

    out = {}
    for pt in np.unique(pt_test):
        if pt == "none":
            continue
        mask = pt_test == pt
        if mask.sum() < 2:
            continue
        out[pt] = float(preds[mask].mean())

    # Overall recall on all illicit transactions (matched patterns + unmatched/"unknown")
    mask_all = pt_test != "none"
    if mask_all.sum() > 0:
        out["_overall"] = float(preds[mask_all].mean())
    return out


def plot_heatmap(table: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(max(8, len(table.columns) * 1.5), max(4, len(table) * 0.6 + 1)))
    sns.heatmap(
        table.astype(float), annot=True, fmt=".2f",
        cmap="Greys", vmin=0, vmax=max(0.5, table.values.max()),
        ax=ax, linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Recall"},
        annot_kws={"fontsize": 9, "fontfamily": "serif"},
    )
    ax.set_title("Per-Pattern Detection Recall under DP-FL (IBM HI-Small)", fontsize=13)
    ax.set_xlabel("Privacy Budget ε")
    ax.set_ylabel("Laundering Pattern")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Fig 5 saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Pattern-aware F1 analysis.")
    parser.add_argument("--data", default=str(DATA_DIR / "HI-Small_Trans.csv"))
    parser.add_argument("--patterns", default=str(DATA_DIR / "HI-Small_Patterns.txt"))
    parser.add_argument("--epsilon", type=float, nargs="+", default=EPSILON_LIST)
    parser.add_argument("--nrows", type=int, default=None)
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
    _, X_test, _, y_test, _, pt_test = grouped_train_test_split(
        X, y, pt, groups=groups, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    pattern_n = pd.Series(pt_test[y_test == 1]).value_counts()
    print(f"\nTest-set illicit transactions: {int((y_test == 1).sum())}")
    print(f"Per-pattern illicit counts (test split):\n{pattern_n.to_string()}")

    model_kwargs = {"input_dim": X.shape[1], "hidden_dims": HIDDEN_DIMS, "dropout": DROPOUT}
    all_results = {}

    model_map = {"inf (FedAvg)": RESULTS_DIR / "models" / "fedavg.pt"}
    for eps in sorted(args.epsilon, reverse=True):
        # exp3_dp_fl.py always parses --epsilon as float before building its
        # save filename, so the tag must match that (e.g. 50 -> "50_0"),
        # regardless of whether eps arrived here as an int (config default) or
        # a CLI-parsed float.
        tag = str(float(eps)).replace(".", "_")
        model_map[str(eps)] = RESULTS_DIR / "models" / f"dp_fl_eps{tag}.pt"

    for label, path in model_map.items():
        if not path.exists():
            print(f"  [skip] {label} — {path.name} not found")
            continue
        model = load_model(model_kwargs, path, device)
        pf = per_pattern_recall(model, X_test, y_test, pt_test, device)
        all_results[label] = pf
        print(f"  ε={label:8s}: {pf}")

    # Build DataFrame: rows=patterns, cols=ε
    row_patterns = [p for p in PATTERN_TYPES if any(p in v for v in all_results.values())]
    if any("unknown" in v for v in all_results.values()):
        row_patterns += ["unknown"]
    row_patterns += ["_overall"]
    table = pd.DataFrame(
        {eps_k: {pt: all_results[eps_k].get(pt, np.nan) for pt in row_patterns}
         for eps_k in all_results},
        index=row_patterns,
    )
    # Rename for display
    table.index = [
        "Unmatched (no documented pattern)" if r == "unknown"
        else "ALL laundering" if r == "_overall"
        else r
        for r in table.index
    ]

    print("\n=== Table 2: Pattern-Stratified Recall ===")
    print(table.to_string(float_format="{:.3f}".format))

    table_path = RESULTS_DIR / "tables" / "table2_pattern_results.csv"
    table.to_csv(table_path)
    print(f"Table 2 saved → {table_path}")

    n_path = RESULTS_DIR / "tables" / "table2_pattern_n.csv"
    pattern_n.to_csv(n_path, header=["n"])
    print(f"Per-pattern n saved → {n_path}")

    fig_path = RESULTS_DIR / "figures" / "fig5_pattern_aware_f1.png"
    plot_heatmap(table, fig_path)

    out = RESULTS_DIR / "tables" / "exp4_pattern_analysis.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full results → {out}")


if __name__ == "__main__":
    main()
