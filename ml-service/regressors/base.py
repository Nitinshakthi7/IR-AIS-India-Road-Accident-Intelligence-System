#!/usr/bin/env python3
"""
IR-AIS Regressors — Shared Evaluation Utilities
Common metrics computation and printing for all regression models.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(model, X_test, y_test, approach="base"):
    """
    Evaluate a fitted regressor and return metrics dict + R² score.

    Parameters
    ----------
    model : fitted sklearn-compatible regressor
    X_test : array-like — test features
    y_test : array-like — test target values
    approach : str — label for the training approach used

    Returns
    -------
    metrics : dict with MAE, MSE, RMSE, R²
    r2 : float — the R² score
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mae": round(float(mae), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
        "approach": approach,
    }

    return metrics, r2


def print_metrics(metrics):
    """Pretty-print regressor metrics to console."""
    print(
        f"    MAE: {metrics['mae']:.4f}, "
        f"MSE: {metrics['mse']:.4f}, "
        f"RMSE: {metrics['rmse']:.4f}, "
        f"R²: {metrics['r2']:.4f}"
    )
