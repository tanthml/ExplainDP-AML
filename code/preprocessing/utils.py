"""Shared preprocessing utilities — imported by experiment scripts.

The numbered scripts (01_eda.py etc.) are standalone CLI tools.
Experiments import from this module instead.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path


PATTERN_ALIASES = {
    "fan-out": "fan-out", "fanout": "fan-out",
    "fan-in": "fan-in", "fanin": "fan-in",
    "cycle": "cycle",
    "scatter-gather": "scatter-gather", "scattergather": "scatter-gather",
    "gather-scatter": "gather-scatter", "gatherscatter": "gather-scatter",
    "bipartite": "bipartite",
    "stack": "stack",
    "random": "random",
}

CATEGORICAL_COLS = ["Payment Format", "Payment Currency", "Receiving Currency"]


def parse_patterns_txt(path: Path):
    """Return (account_to_pattern, account_to_group).

    account_to_pattern: {account_id (str): pattern_type (str)}
    account_to_group:   {account_id (str): laundering_group_id (int)},
        one id per BEGIN/END block (laundering-scheme instance), for
        group-aware splitting -- see grouped_train_test_split below.

    Actual IBM AML format:
        BEGIN LAUNDERING ATTEMPT - <TYPE>:  <description>
        timestamp,from_bank,from_acct,to_bank,to_acct,amt,...,1
        ...
        END LAUNDERING ATTEMPT - <TYPE>

    Extracts from_acct (col index 2) and to_acct (col index 4) per block.
    """
    account_to_pattern = {}
    account_to_group = {}
    if not path.exists():
        print(f"[warn] {path} not found — pattern labels unavailable")
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
                # Extract type between " - " and next ":" or end
                after_dash = line.split(" - ", 1)[-1]
                raw_type = after_dash.split(":")[0].strip().lower()
                current_type = PATTERN_ALIASES.get(raw_type)
                if current_type is None:
                    # Try stripping trailing description words
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

            # CSV transaction line: timestamp,from_bank,from_acct,to_bank,to_acct,...
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

    return account_to_pattern, account_to_group


def assign_pattern_labels(df: pd.DataFrame, account_to_pattern: dict, account_to_group: dict = None) -> pd.DataFrame:
    """Add 'pattern_type' (none / fan-in / fan-out / ... / unknown) and
    'group_id' columns. Legit/unmatched rows get their own singleton
    group (negative, offset by row index) so they can't collide with a
    real 0-based laundering-group id.
    """
    df = df.copy()

    def get_pattern(row):
        if row["Is Laundering"] == 0:
            return "none"
        pt = account_to_pattern.get(str(row["Account"]))
        if pt is None:
            pt = account_to_pattern.get(str(row["Account.1"]))
        return pt if pt else "unknown"

    df["pattern_type"] = df.apply(get_pattern, axis=1)

    if account_to_group is not None:
        def get_group(row):
            if row["Is Laundering"] == 1:
                g = account_to_group.get(str(row["Account"]))
                if g is None:
                    g = account_to_group.get(str(row["Account.1"]))
                if g is not None:
                    return g
            return -1 - row.name

        df["group_id"] = df.apply(get_group, axis=1)

    return df


def engineer_features(df: pd.DataFrame):
    """Add derived features. Returns (df_with_features, encoders_dict)."""
    from sklearn.preprocessing import LabelEncoder
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    df["tx_hour"] = df["Timestamp"].dt.hour
    df["tx_day_of_week"] = df["Timestamp"].dt.dayofweek
    df["tx_day_of_month"] = df["Timestamp"].dt.day

    eps = 1e-9
    df["log_amount_paid"] = np.log1p(df["Amount Paid"])
    df["log_amount_received"] = np.log1p(df["Amount Received"])
    df["amount_ratio"] = df["Amount Paid"] / (df["Amount Received"] + eps)
    df["log_amount_ratio"] = np.log1p(df["amount_ratio"])
    df["amount_diff"] = np.abs(df["Amount Paid"] - df["Amount Received"])
    df["log_amount_diff"] = np.log1p(df["amount_diff"])

    df["is_same_currency"] = (df["Payment Currency"] == df["Receiving Currency"]).astype(int)
    df["is_cross_bank"] = (df["From Bank"] != df["To Bank"]).astype(int)

    for col in ["From Bank", "To Bank"]:
        freq = df[col].value_counts(normalize=True)
        df[f"{col.replace(' ', '_').lower()}_freq"] = df[col].map(freq).fillna(0.0)

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[f"{col.replace(' ', '_').lower()}_enc"] = le.fit_transform(df[col].fillna("UNK"))
        encoders[col] = le

    return df, encoders


_DROP_COLS = {
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format",
    "Is Laundering", "pattern_type", "group_id",
    "amount_ratio", "amount_diff",
}


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in _DROP_COLS]


def grouped_train_test_split(*arrays, groups, test_size, random_state=None):
    """Group-aware analogue of sklearn's train_test_split.

    Every array in *arrays must be the same length as `groups`. Splits
    each array into a (train, test) pair such that all rows sharing a
    `groups` value land on the same side -- so a laundering attempt's
    linked transactions (fan-in, cycle, ...; see group_id in
    02_merge_patterns.py) can never be split across train and test.

    Not stratified by label: GroupShuffleSplit has no stratified variant.
    In practice this is a minor approximation here, since ~99.9%% of
    groups are singleton legitimate transactions (each row is its own
    group), so the resulting illicit rate in each split lands close to
    the population rate in expectation, just with more split-to-split
    variance than an explicitly stratified split would have.

    Returns arrays interleaved as [a1_train, a1_test, a2_train, a2_test, ...],
    matching sklearn's train_test_split calling convention.
    """
    from sklearn.model_selection import GroupShuffleSplit
    n = len(groups)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(np.zeros(n), groups=np.asarray(groups)))
    out = []
    for a in arrays:
        a = np.asarray(a) if not isinstance(a, np.ndarray) else a
        out.append(a[train_idx])
        out.append(a[test_idx])
    return out


def select_f1_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """F1-maximizing decision threshold via the precision-recall curve.

    O(n log n) vs. an O(n) linear threshold scan. Callers MUST pass a
    held-out validation split here and apply the returned threshold to
    the test set -- selecting the threshold directly on the test set
    optimistically biases every downstream reported metric.
    """
    from sklearn.metrics import precision_recall_curve
    prec, rec, thresh = precision_recall_curve(y, scores)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    best_idx = int(np.argmax(f1))
    return float(thresh[best_idx]) if best_idx < len(thresh) else 0.5


def load_and_prepare(trans_path: Path, patterns_path: Path, nrows=None):
    """Full pipeline: CSV → (X_scaled, y, pattern_types, banks, feature_cols, scaler, groups)."""
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(trans_path, nrows=nrows, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    account_to_pattern, account_to_group = parse_patterns_txt(patterns_path)
    df = assign_pattern_labels(df, account_to_pattern, account_to_group)
    df, _ = engineer_features(df)

    feature_cols = get_feature_columns(df)
    X = np.array(df[feature_cols].values, dtype=np.float32)
    y = np.array(df["Is Laundering"].values, dtype=int)
    pt = np.array(df["pattern_type"].values, dtype=str) if "pattern_type" in df.columns else np.full(len(y), "none")
    banks = np.array(df["From Bank"].values, dtype=str) if "From Bank" in df.columns else None
    groups = np.array(df["group_id"].values, dtype=np.int64) if "group_id" in df.columns else np.arange(len(y))

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, pt, banks, feature_cols, scaler, groups


def load_from_features_parquet(features_path: Path):
    """Fast load from pre-computed, already-scaled features.parquet.

    Skips CSV parsing and feature engineering — ~50x faster than load_and_prepare.
    Returns the same tuple signature so callers can use it as a drop-in replacement.
    The 'scaler' slot returns None because the data is already scaled.
    """
    import json

    df = pd.read_parquet(features_path)

    names_path = features_path.parent / "feature_names.json"
    if names_path.exists():
        with open(names_path) as f:
            feature_cols = json.load(f)
    else:
        feature_cols = get_feature_columns(df)

    X = np.array(df[feature_cols].values, dtype=np.float32)
    y = np.array(df["Is Laundering"].values, dtype=int)
    pt = np.array(df["pattern_type"].values, dtype=str) if "pattern_type" in df.columns else np.full(len(y), "none")
    banks = np.array(df["From Bank"].values, dtype=str) if "From Bank" in df.columns else None
    groups = np.array(df["group_id"].values, dtype=np.int64) if "group_id" in df.columns else np.arange(len(y))

    return X, y, pt, banks, feature_cols, None, groups
