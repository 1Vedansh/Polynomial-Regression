"""
extracts the data, loops through all degrees and models, performs K Fold CV
finds the best configuration, evaluates it on the hold out set
then trains on all data to generate the test predictions.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.exceptions import ConvergenceWarning

import config
from data_loader import load_var

# Features & Standardization
def build_design_matrix(X, degree):
    poly = PolynomialFeatures(degree, include_bias=False)
    Phi = poly.fit_transform(X)
    return Phi

class Standardizer:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-12] = 1.0
        return self
        
    def transform(self, X):
        return (X - self.mean_) / self.std_

# Evaluation Metrics
def predict(X, theta):
    return theta[0] + X @ theta[1:]

def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

# Solvers
def fit_ols(X, y):
    X_aug = np.column_stack((np.ones(X.shape[0]), X))
    theta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    return [{"alpha": 0.0, "l1_ratio": 0.0, "theta": theta, "converged": True}]

def fit_ridge(X, y, lambdas):
    res = []
    for alpha in lambdas:
        model = Ridge(alpha=alpha, fit_intercept=True, solver='svd')
        model.fit(X, y)
        theta = np.concatenate(([model.intercept_], model.coef_))
        res.append({"alpha": float(alpha), "l1_ratio": 0.0, "theta": theta, "converged": True})
    return res

def _sparse_path(estimator_cls, X, y, alphas, **extra):
    est = estimator_cls(alpha=1.0, fit_intercept=True, warm_start=True,
                        max_iter=config.SKLEARN_MAX_ITER, tol=config.SKLEARN_TOL, **extra)
    res = []
    for alpha in sorted((float(a) for a in alphas), reverse=True):
        est.set_params(alpha=alpha)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            est.fit(X, y)
        theta = np.concatenate(([est.intercept_], est.coef_))
        res.append({"alpha": alpha, "l1_ratio": extra.get("l1_ratio", 1.0), "theta": theta, "converged": est.n_iter_ < est.max_iter})
    return res

def fit_path(model, X, y, grid):
    if model == "ols": 
        return fit_ols(X, y)
    elif model == "ridge": 
        return fit_ridge(X, y, grid["lambdas"])
    elif model == "lasso": 
        return _sparse_path(Lasso, X, y, grid["alphas"])
    elif model == "elasticnet":
        res = []
        for l1 in grid["l1_ratios"]:
            res.extend(_sparse_path(ElasticNet, X, y, grid["alphas"], l1_ratio=float(l1)))
        return res

# K-Fold CV
def kfold_indices(n, k, seed):
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), k)
    return [(np.concatenate([folds[j] for j in range(k) if j != i]), folds[i]) for i in range(k)]

# Main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", type=int, default=1)
    ap.add_argument("--degrees", nargs="+", type=int, default=None)
    args = ap.parse_args()

    if args.degrees is None:
        args.degrees = list(range(1, 11)) if args.var == 1 else list(range(1, 21))

    # Load Data
    from data_loader import load_var, holdout_split
    X, y, X_test, var_names = load_var(args.var)
    X_dev, y_dev, X_hold, y_hold = holdout_split(X, y)
    splits = kfold_indices(len(X_dev), config.K_FOLDS, config.SEED)

    best_mse = float('inf')
    best_config = None

    # Train across all degrees & models
    print(f"Starting CV for var {args.var}...\n")
    for degree in args.degrees:
        Phi_dev = build_design_matrix(X_dev, degree)
        
        for model in config.MODELS:
            if degree not in config.MODEL_DEGREES[model]: continue
            
            grid = config.MODEL_GRIDS[model]
            
            def eval_fold(tr_idx, va_idx):
                scaler = Standardizer().fit(Phi_dev[tr_idx])
                X_tr = scaler.transform(Phi_dev[tr_idx])
                X_va = scaler.transform(Phi_dev[va_idx])
                y_tr, y_va = y_dev[tr_idx], y_dev[va_idx]
                
                fold_res = []
                for fit in fit_path(model, X_tr, y_tr, grid):
                    y_hat = predict(X_va, fit["theta"])
                    fold_res.append({
                        "alpha": fit["alpha"],
                        "l1_ratio": fit["l1_ratio"],
                        "val_mse": mse(y_va, y_hat)
                    })
                return fold_res

            import time
            t0 = time.time()
            
            # Run K folds in parallel
            results = Parallel(n_jobs=config.N_JOBS, prefer="threads")(
                delayed(eval_fold)(tr, va) for tr, va in splits
            )
            
            t1 = time.time()
            elapsed_sec = t1 - t0
            
            # Find best hyper parameter for this model and degree
            df = pd.DataFrame([item for sublist in results for item in sublist])
            if df.empty: continue
            
            mean_mse = df.groupby(["alpha", "l1_ratio"])["val_mse"].mean()
            best_idx = mean_mse.idxmin()
            curr_best_mse = mean_mse[best_idx]
            
            print(f"Finished {model.upper():>10} (degree {degree:>2}) -> best MSE: {curr_best_mse:.4f}  [Time: {elapsed_sec:.1f}s]")
            
            if curr_best_mse < best_mse:
                best_mse = curr_best_mse
                best_config = {
                    "model": model,
                    "degree": degree,
                    "alpha": best_idx[0],
                    "l1_ratio": best_idx[1]
                }
                print(f"NEW OVERALL BEST")

    print(f"\nBest Model Found:")
    print(best_config)

    # Evaluate Hold out Performance
    Phi_dev = build_design_matrix(X_dev, best_config["degree"])
    scaler = Standardizer().fit(Phi_dev)
    X_tr = scaler.transform(Phi_dev)
    
    best_grid = {"lambdas": [best_config["alpha"]]} if best_config["model"] == "ridge" else \
                {"alphas": [best_config["alpha"]], "l1_ratios": [best_config["l1_ratio"]]}
                
    theta_dev = fit_path(best_config["model"], X_tr, y_dev, best_grid)[0]["theta"]
    
    Phi_hold = build_design_matrix(X_hold, best_config["degree"])
    X_ho = scaler.transform(Phi_hold)
    y_ho_pred = predict(X_ho, theta_dev)
    
    print(f"\nHold-out Set Metrics:")
    print(f"  MSE: {mse(y_hold, y_ho_pred):.4f}")
    print(f"  R2:  {r2(y_hold, y_ho_pred):.4f}")

    # Refit on ALL data and generate final Test Predictions
    X_train_all = np.vstack([X_dev, X_hold])
    y_train_all = np.concatenate([y_dev, y_hold])
    
    Phi_all = build_design_matrix(X_train_all, best_config["degree"])
    scaler = Standardizer().fit(Phi_all)
    X_tr_all = scaler.transform(Phi_all)
    
    theta_all = fit_path(best_config["model"], X_tr_all, y_train_all, best_grid)[0]["theta"]
    
    Phi_test = build_design_matrix(X_test, best_config["degree"])
    X_te = scaler.transform(Phi_test)
    y_test_pred = predict(X_te, theta_all)
    
    pred_df = pd.DataFrame({config.TARGET_COL: y_test_pred})
    pred_path = config.prediction_path(args.var)
    pred_df.to_csv(pred_path, index=False)
    print(f"\nFinal Test predictions generated successfully:")
    print(f"-> {pred_path}")