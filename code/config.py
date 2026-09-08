from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

for _d in [RESULTS_DIR / "figures", RESULTS_DIR / "tables", RESULTS_DIR / "models"]:
    _d.mkdir(parents=True, exist_ok=True)

PATTERN_TYPES = [
    "fan-in", "fan-out", "cycle",
    "scatter-gather", "gather-scatter",
    "bipartite", "stack", "random",
]

EPSILON_LIST = [1, 5, 10, 50]
NUM_CLIENTS = 5
NUM_ROUNDS = 20
DELTA = 1e-5
# Hyperparameters selected by grid search over lr ∈ {5e-4,1e-3,2e-3,3e-3},
# max_grad_norm ∈ {0.5,1.0,1.5}, hidden_dims ∈ {[256,128,64],[512,256,128]}
# at ε=5 (24 configs × 20 FL rounds). Best: AUC=0.9611, F1=0.5339.
# See results/tuning/ for full grid-search logs and per-ε Phase-2 results.
MAX_GRAD_NORM = 0.5
LOCAL_EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 3e-3
HIDDEN_DIMS = [512, 256, 128]
DROPOUT = 0.3
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1
DIRICHLET_ALPHA = 0.5
