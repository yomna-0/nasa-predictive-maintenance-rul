"""Baseline models (Member 2): trivial, Ridge and Random Forest."""

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def flatten_windows(X):
    """Flatten (samples, window, features) into (samples, window * features)."""
    return X.reshape(X.shape[0], -1)


def build_trivial_model():
    """Always predicts the mean training RUL."""
    return DummyRegressor(strategy="mean")


def build_ridge(alpha):
    """Ridge regression with the given L2 strength."""
    return Ridge(alpha=alpha)


def select_ridge_alpha(alphas, X_train, y_train, X_val, y_val):
    """Pick the alpha with the lowest validation RMSE (test is never used).

    Returns (best_alpha, [(alpha, validation_rmse), ...]).
    """
    results = []
    for a in alphas:
        m = Ridge(alpha=a)
        m.fit(X_train, y_train)
        rmse = float(np.sqrt(mean_squared_error(y_val, m.predict(X_val))))
        results.append((a, rmse))
    best_alpha = min(results, key=lambda t: t[1])[0]
    return best_alpha, results


def build_random_forest(seed, n_estimators=100, max_features="sqrt",
                        min_samples_leaf=3):
    """Random forest regressor with the project settings."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=seed,
    )
