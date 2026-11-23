#!/bin/bash
# Rerun: seeds 18/19/20, per-seed checkpoints (for SHAP cross-seed
# stability), accounting info logged from exp3. Run from code/ with .venv
# active. GPU used automatically if available, falls back to CPU otherwise.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python3

echo "=== exp2 FedAvg (seeds 18 19 20) ==="
$PY experiments/exp2_fedavg.py --fast --rounds 20 --seeds 18 19 20

for eps in 50 10 5 1; do
    echo "=== exp3 DP-FL eps=$eps (seeds 18 19 20) ==="
    $PY experiments/exp3_dp_fl.py --fast --rounds 20 --epsilon $eps --seeds 18 19 20
done

echo "=== exp4 pattern analysis ==="
$PY experiments/exp4_pattern_analysis.py --fast

echo "=== DONE ==="
