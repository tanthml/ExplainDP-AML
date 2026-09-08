"""Generate paper figures from saved experiment JSON results.

Fig 2 — Privacy-utility tradeoff curve (F1 vs ε)
Fig 3 — Convergence curves (FedAvg vs DP-FL ε=5)

Usage:
    python results/plot_results.py
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR, EPSILON_LIST

# Black/white academic style (R-like)
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
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.framealpha": 1.0,
    "lines.linewidth": 1.2,
    "lines.markersize": 6,
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
})


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def plot_fig2_privacy_utility():
    """F1 and AUC vs ε (Table 1 data)."""
    rows = []

    # Centralized
    c = load_json(RESULTS_DIR / "tables" / "exp1_centralized.json")
    if c and "mlp" in c:
        rows.append({"method": "Centralized", "epsilon": float("inf"),
                     "f1": c["mlp"]["f1_macro"], "auc": c["mlp"].get("auc", 0)})

    # FedAvg
    fa = load_json(RESULTS_DIR / "tables" / "exp2_fedavg.json")
    if fa and "best" in fa:
        rows.append({"method": "FedAvg", "epsilon": float("inf"),
                     "f1": fa["best"]["f1_macro"], "auc": fa["best"]["auc"]})

    # DP-FL per ε
    for eps in sorted(EPSILON_LIST):
        tag = str(eps).replace(".", "_")
        d = (load_json(RESULTS_DIR / "tables" / f"exp3_dp_fl_eps{tag}_0.json")
             or load_json(RESULTS_DIR / "tables" / f"exp3_dp_fl_eps{tag}.json"))
        if d and d.get("best"):
            best = d["best"]
            rows.append({"method": "DP-FL", "epsilon": eps,
                         "f1": best["f1_macro"], "auc": best["auc"]})

    if not rows:
        print("[skip] No data for Fig 2")
        return

    dp_rows = [r for r in rows if r["method"] == "DP-FL"]
    if dp_rows:
        eps_vals = [r["epsilon"] for r in dp_rows]
        f1_vals = [r["f1"] for r in dp_rows]
        auc_vals = [r["auc"] for r in dp_rows]

        fig, ax1 = plt.subplots(figsize=(6, 3.8))
        ax2 = ax1.twinx()
        ax1.plot(eps_vals, f1_vals, "o-", color="black", label="F1 (macro)")
        ax2.plot(eps_vals, auc_vals, "s--", color="black", dashes=(4, 2), label="AUC-ROC")

        # Reference lines
        for r in rows:
            if r["method"] == "Centralized":
                ax1.axhline(r["f1"], color="black", linestyle=":", linewidth=0.8,
                            label="Centralized F1")
            if r["method"] == "FedAvg":
                ax1.axhline(r["f1"], color="black", linestyle=(0, (3, 1, 1, 1)), linewidth=0.8,
                            label="FedAvg F1")

        ax1.set_xlabel("Privacy Budget ε (log scale)")
        ax1.set_ylabel("F1 (macro)")
        ax2.set_ylabel("AUC-ROC")
        ax2.tick_params(axis="y", direction="in")
        ax1.set_xscale("log")
        ax1.set_title("Privacy-Utility Tradeoff (IBM HI-Small)")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
        plt.tight_layout()
        out = RESULTS_DIR / "figures" / "fig2_privacy_utility_tradeoff.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"Fig 2 saved → {out}")


def plot_fig3_convergence():
    """FedAvg vs DP-FL (ε=5) convergence over rounds."""
    fa = load_json(RESULTS_DIR / "tables" / "exp2_fedavg.json")
    dp = load_json(RESULTS_DIR / "tables" / "exp3_dp_fl_eps5.json") or \
         load_json(RESULTS_DIR / "tables" / "exp3_dp_fl_eps5_0.json")

    if not fa or not dp:
        print("[skip] Missing data for Fig 3")
        return

    rounds_fa = [h["round"] for h in fa["history"]]
    f1_fa = [h["f1_macro"] for h in fa["history"]]
    rounds_dp = [h["round"] for h in dp["history"]]
    f1_dp = [h["f1_macro"] for h in dp["history"]]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(rounds_fa, f1_fa, "o-", color="black", label="FedAvg (ε=∞)")
    ax.plot(rounds_dp, f1_dp, "s--", color="black", dashes=(4, 2),
            label=f"DP-FL (ε={dp.get('epsilon', 5)})")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("F1 (macro) on Global Test Set")
    ax.set_title("Convergence — FedAvg vs DP-FL (ε=5)")
    ax.legend()
    plt.tight_layout()
    out = RESULTS_DIR / "figures" / "fig3_convergence.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Fig 3 saved → {out}")


def print_table1():
    """Print Table 1: main results summary."""
    rows = []

    c = load_json(RESULTS_DIR / "tables" / "exp1_centralized.json")
    if c:
        if "mlp" in c:
            rows.append({"Method": "Centralized MLP", "ε": "∞",
                         "F1 (macro)": c["mlp"]["f1_macro"], "AUC": c["mlp"].get("auc", "-")})
        if "xgboost" in c:
            rows.append({"Method": "Centralized XGBoost", "ε": "∞",
                         "F1 (macro)": c["xgboost"]["f1_macro"], "AUC": c["xgboost"].get("auc", "-")})

    fa = load_json(RESULTS_DIR / "tables" / "exp2_fedavg.json")
    if fa and "best" in fa:
        rows.append({"Method": "FedAvg", "ε": "∞",
                     "F1 (macro)": fa["best"]["f1_macro"], "AUC": fa["best"]["auc"]})

    for eps in sorted(EPSILON_LIST, reverse=True):
        tag = str(eps).replace(".", "_")
        d = (load_json(RESULTS_DIR / "tables" / f"exp3_dp_fl_eps{tag}_0.json")
             or load_json(RESULTS_DIR / "tables" / f"exp3_dp_fl_eps{tag}.json"))
        if d and d.get("best"):
            best = d["best"]
            rows.append({"Method": "DP-FL", "ε": eps,
                         "F1 (macro)": best["f1_macro"], "AUC": best["auc"]})

    if not rows:
        print("No results to display yet")
        return

    print("\n=== Table 1: Main Results (HI-Small) ===")
    print(f"{'Method':<25} {'ε':>8} {'F1 (macro)':>12} {'AUC':>8}")
    print("-" * 58)
    for r in rows:
        f1 = f"{r['F1 (macro)']:.4f}" if isinstance(r["F1 (macro)"], float) else str(r["F1 (macro)"])
        auc = f"{r['AUC']:.4f}" if isinstance(r["AUC"], float) else str(r["AUC"])
        print(f"{r['Method']:<25} {str(r['ε']):>8} {f1:>12} {auc:>8}")


def plot_fig6_transfer():
    """Cross-dataset AUC: HI-Small (in-dist) vs LI-Small (zero-shot transfer)."""
    import csv

    csv_path = RESULTS_DIR / "tables" / "table_supp_transfer.csv"
    if not csv_path.exists():
        print("[skip] No transfer CSV for Fig 6")
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    labels = [r["Model"].replace(" (no DP)", "") for r in rows]
    hi_auc  = [float(r["HI_Small_AUC"]) for r in rows]
    li_auc  = [float(r["LI_Small_AUC"]) for r in rows]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, hi_auc, w, label="HI-Small (in-distribution)",
           color="black", alpha=0.85)
    ax.bar(x + w/2, li_auc, w, label="LI-Small (zero-shot transfer)",
           color="white", edgecolor="black", hatch="//")

    ax.set_ylabel("AUC-ROC")
    ax.set_title("Cross-Dataset Generalization: HI-Small vs LI-Small (Zero-Shot)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0.70, 1.00)
    ax.legend(loc="lower right")
    plt.tight_layout()
    out = RESULTS_DIR / "figures" / "fig6_transfer_generalization.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Fig 6 saved → {out}")


if __name__ == "__main__":
    plot_fig2_privacy_utility()
    plot_fig3_convergence()
    plot_fig6_transfer()
    print_table1()
