#!/bin/bash
# FedProx + sampling/calibration ablation + SCAFFOLD, GPU-accelerated.
# Run from code/ with .venv active. GPU used automatically if available
# (falls back to CPU otherwise); ~2.5-3h total on an RTX 3060, longer on CPU.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python3
MU=${1:-0.001}   # FedProx coefficient, selected via a small validation sweep
SEEDS="18 19 20"
echo "Using FedProx mu=$MU"

echo "=== FedProx non-DP (mu=$MU) ==="
$PY experiments/exp2_fedavg.py --fast --rounds 20 --seeds $SEEDS --mu $MU

for eps in 50 10 5 1; do
    echo "=== DP-FedProx eps=$eps (mu=$MU) ==="
    $PY experiments/exp3_dp_fl.py --fast --rounds 20 --epsilon $eps --seeds $SEEDS --mu $MU
done

echo "=== Sampling ablation: non-DP ==="
$PY experiments/exp2_fedavg.py --fast --rounds 20 --seeds $SEEDS --sampler uniform
$PY experiments/exp2_fedavg.py --fast --rounds 20 --seeds $SEEDS --loss-type ipw
$PY experiments/exp2_fedavg.py --fast --rounds 20 --seeds $SEEDS --loss-type focal

echo "=== Sampling ablation: DP-FL ==="
for eps in 50 10 5 1; do
    echo "--- eps=$eps: uniform sampler ---"
    $PY experiments/exp3_dp_fl.py --fast --rounds 20 --epsilon $eps --seeds $SEEDS --sampler uniform
    echo "--- eps=$eps: ipw loss ---"
    $PY experiments/exp3_dp_fl.py --fast --rounds 20 --epsilon $eps --seeds $SEEDS --loss-type ipw
    echo "--- eps=$eps: focal loss ---"
    $PY experiments/exp3_dp_fl.py --fast --rounds 20 --epsilon $eps --seeds $SEEDS --loss-type focal
done

echo "=== SCAFFOLD (non-DP) ==="
$PY experiments/exp2c_scaffold.py --fast --rounds 20 --seeds $SEEDS

echo "=== DONE ==="
