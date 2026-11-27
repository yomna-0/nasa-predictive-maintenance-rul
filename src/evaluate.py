"""Evaluation metrics: RMSE, MAE, R2 and the NASA asymmetric score."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def nasa_score(y_true, y_pred):
    """NASA asymmetric scoring function: late predictions (d > 0) cost more."""
    d = y_pred - y_true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def evaluate(model, X, y_true, name="", split=""):
    """Predict with `model` on X and return (metrics dict, predictions)."""
    y_pred = model.predict(X)
    return {
        "Model": name,
        "Split": split,
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "NASA Score": nasa_score(y_true, y_pred),
    }, y_pred
