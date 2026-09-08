# ExplainDP-AML

Reference implementation for **"Explainable Differentially Private Federated
Learning for Anti-Money Laundering"**, submitted to IEEE-RIVF 2026.

> **Authors:** Tan Tran (corresponding), Khiem Nguyen, Nhan Dang, An Mai,
> Thinh Pham — College of Engineering and Computer Science, VinUniversity,
> Hanoi, Vietnam.

This is the code-only companion to the paper. It implements a federated,
differentially private (DP-FL) pipeline for anti-money laundering (AML)
detection on the IBM HI-Small benchmark:

- **Calibrated subsampling** that makes DP-SGD tractable at million-transaction
  scale while preserving the dataset's native $\approx$0.10% illicit rate
  (no oversampling/rebalancing).
- **Group-aware train/test splitting** (`GroupShuffleSplit` on
  laundering-scheme group IDs) so that a laundering attempt's linked
  transactions can't leak across the split.
- **Pattern-stratified evaluation** — per-pattern detection recall across
  all eight IBM AML laundering scheme types, under four privacy budgets
  ($\varepsilon \in \{1, 5, 10, 50\}$).
- **SHAP stability analysis** comparing feature importance rankings across
  privacy budgets.

## Dataset

Download the IBM Transactions for Anti-Money Laundering (AML) dataset from
Kaggle:

```
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
```

Place `HI-Small_Trans.csv` and `HI-Small_Patterns.txt` in `data/raw/`.

Dataset reference: Altman et al. (2023), NeurIPS 2023 Datasets and
Benchmarks Track.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

## Preprocessing

Run in order from the `code/` directory:

```bash
cd code
python3 preprocessing/02_merge_patterns.py       # assigns pattern_type + group_id
python3 preprocessing/03_feature_engineering.py  # -> data/features.parquet
```

## Experiments

All experiments use `--fast` (loads the precomputed `features.parquet`).

```bash
cd code

# Exp 1: Centralized baselines (MLP + XGBoost)
python3 experiments/exp1_centralized.py --fast --epochs 30

# Exp 2: FedAvg (no DP)
python3 experiments/exp2_fedavg.py --fast --rounds 20 --max-samples 100000

# Exp 3: DP-FL across privacy budgets
for eps in 50 10 5 1; do
    python3 experiments/exp3_dp_fl.py --fast --rounds 20 --epsilon $eps
done

# Exp 4: Per-pattern analysis
python3 experiments/exp4_pattern_analysis.py --fast

# SHAP explainability
python3 explainability/shap_analysis.py --fast --n-explain 200

# Reproduce all figures and tables
python3 results/plot_results.py
```

Splitting throughout is group-aware: `grouped_train_test_split` in
`preprocessing/utils.py` wraps `GroupShuffleSplit` on the `group_id` column
(one id per `BEGIN`/`END LAUNDERING ATTEMPT` block in
`HI-Small_Patterns.txt`), so a laundering scheme's linked transactions
always land on the same side of train/test.

## License

Code: MIT License. Dataset: CDLA-Sharing-1.0 (IBM Research).

## Citation

```bibtex
@inproceedings{Tran2026explaindpaml,
  author    = {Tran, Tan and Nguyen, Khiem and Dang, Nhan and Mai, An and Pham, Thinh},
  title     = {Explainable Differentially Private Federated Learning for Anti-Money Laundering},
  booktitle = {Proceedings of IEEE-RIVF 2026},
  year      = {2026}
}
```
