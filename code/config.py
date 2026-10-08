# Configuration for the regression parameters.

from pathlib import Path
import numpy as np

ROLL_NO = "BT2024162"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
TARGET_COL = "y"

SEED = 67

# Fraction of the 1000 data points, that is put aside for final testing metrics
HOLDOUT_FRAC = 0.20
K_FOLDS = 10

# For parallel processing (-1 = all cpus)
N_JOBS = -1

# Model search space
DEGREES = list(range(1, 21))
MODELS = ["ols", "ridge", "lasso", "elasticnet"]

# Which degrees to try for each model family.
MODEL_DEGREES = {
    "ols": DEGREES,
    "ridge": DEGREES,
    "lasso": DEGREES,
    "elasticnet": DEGREES,
}

MODEL_GRIDS = {
    "ols": {},
    # "ridge": {"lambdas": np.logspace(-3, 5, 20)},
    "ridge": {"lambdas": np.logspace(-4, 5, 100)},
    # "lasso": {"alphas": np.logspace(-3, 0, 10)},
    "lasso": {"alphas": np.logspace(-4, 2, 50)},
    "elasticnet": {
        # "alphas": np.logspace(-3, 0, 10),
        "alphas": np.logspace(-4, 2, 50),
        # "l1_ratios": [0.3, 0.7],
        "l1_ratios": [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
    },
}

# Coordinate descent settings for Lasso/ElasticNet.
SKLEARN_MAX_ITER = 2000
SKLEARN_TOL = 5e-3

# Path helpers
def train_path(var: int) -> Path:
    return DATA_DIR / f"{ROLL_NO}_train_var{var}.csv"

def test_path(var: int) -> Path:
    return DATA_DIR / f"{ROLL_NO}_test_var{var}.csv"

def results_dir(var: int) -> Path:
    d = RESULTS_DIR / f"var{var}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def prediction_path(var: int) -> Path:
    return results_dir(var) / f"{ROLL_NO}_pred_var{var}.csv"
