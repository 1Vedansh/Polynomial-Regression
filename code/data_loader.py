# read the CSV files and produce the train / hold out split.
import numpy as np
import pandas as pd
import config

def load_csv(path):
    df = pd.read_csv(path)
    feature_names = [c for c in df.columns if c != config.TARGET_COL]
    X = df[feature_names].to_numpy(dtype=float)
    y = df[config.TARGET_COL].to_numpy(dtype=float) if config.TARGET_COL in df.columns else None
    return X, y, feature_names


def load_var(var):
    X_train, y_train, train_cols = load_csv(config.train_path(var))
    X_test, _, test_cols = load_csv(config.test_path(var))
    if train_cols != test_cols:
        raise ValueError(f"Train/test feature columns differ: {train_cols} vs {test_cols}")
    return X_train, y_train, X_test, train_cols


def holdout_split(X, y, frac=config.HOLDOUT_FRAC, seed=config.SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_hold = int(round(frac * len(y)))
    hold_idx, dev_idx = idx[:n_hold], idx[n_hold:]
    return X[dev_idx], y[dev_idx], X[hold_idx], y[hold_idx]
