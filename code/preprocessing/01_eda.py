"""EDA: IBM AML HI-Small dataset — class distribution, pattern breakdown, feature stats."""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, RESULTS_DIR


PATTERN_ALIASES_EDA = {
    "fan-out": "fan-out", "fanout": "fan-out",
    "fan-in": "fan-in", "fanin": "fan-in",
    "cycle": "cycle",
    "scatter-gather": "scatter-gather", "scattergather": "scatter-gather",
    "gather-scatter": "gather-scatter", "gatherscatter": "gather-scatter",
    "bipartite": "bipartite", "stack": "stack", "random": "random",
}


def parse_patterns_txt(path: Path) -> dict:
    """Parse Patterns.txt → {account_id: pattern_type}.

    Actual IBM AML format:
        BEGIN LAUNDERING ATTEMPT - <TYPE>:  <description>
        timestamp,from_bank,from_acct,to_bank,to_acct,...,1
        ...
        END LAUNDERING ATTEMPT - <TYPE>
    """
    account_pattern = {}
    if not path.exists():
        print(f"[warn] {path} not found — skipping pattern analysis")
        return account_pattern

    current_type = None
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("BEGIN LAUNDERING ATTEMPT"):
                after_dash = line.split(" - ", 1)[-1]
                raw_type = after_dash.split(":")[0].strip().lower()
                current_type = PATTERN_ALIASES_EDA.get(raw_type)
                if current_type is None:
                    for alias, canonical in PATTERN_ALIASES_EDA.items():
                        if raw_type.startswith(alias):
                            current_type = canonical
                            break
                continue
            if upper.startswith("END LAUNDERING ATTEMPT"):
                current_type = None
                continue
            if current_type is None:
                continue
            parts = line.split(",")
            if len(parts) >= 5:
                from_acct = parts[2].strip()
                to_acct = parts[4].strip()
                if from_acct:
                    account_pattern[from_acct] = current_type
                if to_acct:
                    account_pattern[to_acct] = current_type

    return account_pattern


def load_transactions(csv_path: Path, nrows: int = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, nrows=nrows, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def plot_class_distribution(df: pd.DataFrame, name: str) -> None:
    counts = df["Is Laundering"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Legitimate", "Laundering"], counts.values, color=["steelblue", "tomato"])
    ax.set_title(f"Class Distribution — {name}")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + counts.values.max() * 0.01, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center")
    plt.tight_layout()
    out = RESULTS_DIR / "figures" / f"eda_class_{name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out.name}")


def plot_amount_distribution(df: pd.DataFrame, name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col in zip(axes, ["Amount Paid", "Amount Received"]):
        log_vals = np.log1p(df[col])
        ax.hist(log_vals[df["Is Laundering"] == 0], bins=60, alpha=0.6, label="Legitimate", color="steelblue", density=True)
        ax.hist(log_vals[df["Is Laundering"] == 1], bins=60, alpha=0.6, label="Laundering", color="tomato", density=True)
        ax.set_xlabel(f"log1p({col})")
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_title(col)
    fig.suptitle(f"Amount Distributions — {name}")
    plt.tight_layout()
    out = RESULTS_DIR / "figures" / f"eda_amounts_{name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out.name}")


def plot_pattern_distribution(df: pd.DataFrame, name: str) -> None:
    if "pattern_type" not in df.columns:
        return
    launder = df[df["Is Laundering"] == 1]["pattern_type"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 4))
    launder.plot(kind="bar", ax=ax, color="tomato")
    ax.set_title(f"Laundering Pattern Distribution — {name}")
    ax.set_ylabel("Count")
    ax.set_xlabel("Pattern Type")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = RESULTS_DIR / "figures" / f"eda_patterns_{name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out.name}")


def plot_top_banks(df: pd.DataFrame, name: str, top_n: int = 15) -> None:
    bank_counts = df["From Bank"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 4))
    bank_counts.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title(f"Top {top_n} From Banks — {name}")
    ax.set_ylabel("Transaction Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = RESULTS_DIR / "figures" / f"eda_banks_{name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out.name}")


def main():
    parser = argparse.ArgumentParser(description="EDA for IBM AML dataset.")
    parser.add_argument("--trans", default=str(DATA_DIR / "HI-Small_Trans.csv"))
    parser.add_argument("--patterns", default=str(DATA_DIR / "HI-Small_Patterns.txt"))
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows for fast EDA")
    args = parser.parse_args()

    print("=== EDA: HI-Small ===")
    df = load_transactions(Path(args.trans), nrows=args.nrows)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"\nClass distribution:\n{df['Is Laundering'].value_counts(normalize=True)}")
    print(f"\nPayment formats:\n{df['Payment Format'].value_counts()}")
    print(f"\nTop currencies:\n{df['Payment Currency'].value_counts().head(10)}")
    print(f"\nNumber of unique banks (From): {df['From Bank'].nunique()}")

    account_pattern = parse_patterns_txt(Path(args.patterns))
    if account_pattern:
        df["pattern_type"] = df.apply(
            lambda r: account_pattern.get(str(r["Account"]),
                      account_pattern.get(str(r["Account.1"]), "unknown"))
            if r["Is Laundering"] == 1 else "none",
            axis=1,
        )
        print(f"\nPattern distribution (laundering only):")
        print(df[df["Is Laundering"] == 1]["pattern_type"].value_counts())

    plot_class_distribution(df, "HI-Small")
    plot_amount_distribution(df, "HI-Small")
    plot_pattern_distribution(df, "HI-Small")
    plot_top_banks(df, "HI-Small")
    print("\nEDA complete.")


if __name__ == "__main__":
    main()
