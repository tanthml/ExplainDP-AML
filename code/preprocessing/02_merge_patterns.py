"""Merge HI-Small_Patterns.txt into the transaction DataFrame as a pattern_type column.

Patterns.txt format (AMLSim output):
    <pattern_type_keyword>        e.g. "FanOut", "fan-out", "Cycle" ...
    <number_of_accounts>          integer line — skip
    <acct_id1> <acct_id2> ...    space-separated account IDs for this pattern group

Each block separated by blank line. Accounts appearing in a block get that pattern label.
Laundering transactions (Is Laundering=1) whose From/To account matches get the label.
All others get "none" (legitimate) or "unknown" (laundering but no matched pattern).

Output: saves processed CSV to data/HI-Small_Merged.parquet for fast reuse.
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, PATTERN_TYPES


PATTERN_ALIASES = {
    "fanin": "fan-in", "fan_in": "fan-in", "fan-in": "fan-in",
    "fanout": "fan-out", "fan_out": "fan-out", "fan-out": "fan-out",
    "cycle": "cycle",
    "scattergather": "scatter-gather", "scatter_gather": "scatter-gather", "scatter-gather": "scatter-gather",
    "gatherscatter": "gather-scatter", "gather_scatter": "gather-scatter", "gather-scatter": "gather-scatter",
    "bipartite": "bipartite",
    "stack": "stack",
    "random": "random",
}


def parse_patterns_txt(path: Path):
    """Return (account_to_pattern, account_to_group).

    account_to_pattern: {account_id (str): pattern_type (str)}
    account_to_group:   {account_id (str): laundering_group_id (int)} --
        one id per BEGIN/END block, i.e. per laundering-scheme instance.
        Used for group-aware train/test splitting so that transactions
        belonging to the same laundering attempt cannot be split across
        train and test (see grouped_train_test_split in utils.py).

    Actual IBM AML format:
        BEGIN LAUNDERING ATTEMPT - <TYPE>:  <description>
        timestamp,from_bank,from_acct,to_bank,to_acct,amt,...,1
        ...
        END LAUNDERING ATTEMPT - <TYPE>
    """
    account_to_pattern = {}
    account_to_group = {}
    if not path.exists():
        print(f"[warn] {path} not found")
        return account_to_pattern, account_to_group

    current_type = None
    current_group = -1
    next_group_id = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            upper = line.upper()
            if upper.startswith("BEGIN LAUNDERING ATTEMPT"):
                after_dash = line.split(" - ", 1)[-1]
                raw_type = after_dash.split(":")[0].strip().lower()
                current_type = PATTERN_ALIASES.get(raw_type)
                if current_type is None:
                    for alias, canonical in PATTERN_ALIASES.items():
                        if raw_type.startswith(alias):
                            current_type = canonical
                            break
                current_group = next_group_id
                next_group_id += 1
                continue

            if upper.startswith("END LAUNDERING ATTEMPT"):
                current_type = None
                current_group = -1
                continue

            if current_type is None:
                continue

            parts = line.split(",")
            if len(parts) >= 5:
                from_acct = parts[2].strip()
                to_acct = parts[4].strip()
                if from_acct:
                    account_to_pattern[from_acct] = current_type
                    account_to_group[from_acct] = current_group
                if to_acct:
                    account_to_pattern[to_acct] = current_type
                    account_to_group[to_acct] = current_group

    print(f"Parsed {len(account_to_pattern):,} account→pattern mappings "
          f"across {next_group_id:,} laundering-attempt groups")
    from collections import Counter
    print(Counter(account_to_pattern.values()))
    return account_to_pattern, account_to_group


def assign_pattern_labels(df: pd.DataFrame, account_to_pattern: dict, account_to_group: dict) -> pd.DataFrame:
    """Add 'pattern_type' and 'group_id' columns to df.

    group_id lets a group-aware split keep every transaction of a given
    laundering attempt on the same side of train/test. Legitimate and
    unmatched-laundering rows aren't linked to anything, so each gets its
    own singleton group (the row's own index, offset negative so it can
    never collide with a real 0-based laundering-group id).
    """
    from_col = "Account"
    to_col = "Account.1"

    def get_pattern(row):
        if row["Is Laundering"] == 0:
            return "none"
        pt = account_to_pattern.get(str(row[from_col]))
        if pt is None:
            pt = account_to_pattern.get(str(row[to_col]))
        return pt if pt else "unknown"

    def get_group(row):
        if row["Is Laundering"] == 1:
            g = account_to_group.get(str(row[from_col]))
            if g is None:
                g = account_to_group.get(str(row[to_col]))
            if g is not None:
                return g
        return -1 - row.name  # singleton group for legit/unmatched rows

    df = df.copy()
    df["pattern_type"] = df.apply(get_pattern, axis=1)
    df["group_id"] = df.apply(get_group, axis=1)
    return df


def main():
    parser = argparse.ArgumentParser(description="Merge pattern labels into transactions CSV.")
    parser.add_argument("--trans", default=str(DATA_DIR / "HI-Small_Trans.csv"))
    parser.add_argument("--patterns", default=str(DATA_DIR / "HI-Small_Patterns.txt"))
    parser.add_argument("--out", default=str(DATA_DIR / "HI-Small_Merged.parquet"))
    parser.add_argument("--nrows", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading {args.trans}...")
    df = pd.read_csv(args.trans, nrows=args.nrows, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    print(f"  Loaded {len(df):,} rows")

    account_to_pattern, account_to_group = parse_patterns_txt(Path(args.patterns))

    df = assign_pattern_labels(df, account_to_pattern, account_to_group)

    print(f"\npattern_type distribution:")
    print(df["pattern_type"].value_counts())
    print(f"\nLaundering pattern coverage: "
          f"{(df[df['Is Laundering']==1]['pattern_type'] != 'unknown').mean()*100:.1f}%")
    n_real_groups = (df["group_id"] >= 0).sum()
    n_unique_groups = df.loc[df["group_id"] >= 0, "group_id"].nunique()
    print(f"group_id: {n_real_groups:,} rows in {n_unique_groups:,} real "
          f"laundering-attempt groups; remaining rows are singleton groups.")

    out = Path(args.out)
    df.to_parquet(out, index=False)
    print(f"\nSaved merged data → {out}")


if __name__ == "__main__":
    main()
