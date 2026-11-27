"""Data loading and leak-free preparation for the NASA C-MAPSS FD001 RUL task.

Every fitting decision (the train/validation split, constant-feature removal and
the scaler) is made on the training engines only, then applied to validation and
test. Nothing about the validation or test data influences the transformation,
so there is no data leakage.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

COLUMNS = (
    ["id", "cycle", "setting1", "setting2", "setting3"]
    + [f"s{i}" for i in range(1, 22)]
)


def load_config(path="configs/default.json"):
    """Load the project configuration (seeds, window size, model settings)."""
    with open(path) as f:
        return json.load(f)


def load_raw_data(data_dir):
    """Read PM_train.csv, PM_test.csv and PM_truth.csv from data_dir."""
    train_df = pd.read_csv(os.path.join(data_dir, "PM_train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "PM_test.csv"))
    truth_df = pd.read_csv(os.path.join(data_dir, "PM_truth.csv"))
    return train_df, test_df, truth_df


def add_rul(train_df):
    """Add the RUL target: last cycle of the engine minus the current cycle."""
    train_df = train_df.copy()
    train_df["RUL"] = (
        train_df.groupby("id")["cycle"].transform("max") - train_df["cycle"]
    )
    return train_df


def split_by_engine(train_df, val_size=0.2, seed=42):
    """Split row indices into train/validation by engine (no engine overlap)."""
    X = train_df.drop(columns=["RUL", "id"])
    y = train_df["RUL"]
    groups = train_df["id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))
    return train_idx, val_idx


def find_constant_features(X_train):
    """Columns with a single value on the training engines (removed everywhere)."""
    return [col for col in X_train.columns if X_train[col].nunique() <= 1]


def fit_scaler(X_train):
    """Fit a StandardScaler on the training rows only."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def create_sequences(df, feature_cols, window_size):
    """Sliding windows over each engine; target = RUL at the window's last cycle."""
    sequences = []
    targets = []

    for engine_id, engine_data in df.groupby("id"):
        engine_data = engine_data.sort_values("cycle")
        features = engine_data[feature_cols].values
        rul = engine_data["RUL"].values

        for i in range(len(features) - window_size + 1):
            sequences.append(features[i:i + window_size])
            targets.append(rul[i + window_size - 1])

    return np.array(sequences), np.array(targets)


def create_test_sequences(df, feature_cols, window_size):
    """One window per test engine: its last `window_size` cycles."""
    sequences = []

    for engine_id, engine_data in df.groupby("id"):
        engine_data = engine_data.sort_values("cycle")
        features = engine_data[feature_cols].values

        if len(features) < window_size:
            raise ValueError(f"Engine {engine_id} has insufficient cycles.")

        sequences.append(features[-window_size:])

    return np.array(sequences)


def create_validation_sequences(df, feature_cols, window_size, random_state=42):
    """One reproducible simulated pre-failure cutoff per validation engine.

    The validation window ends before engine failure and its target is the
    remaining life at that cutoff, matching the censored test setup. The final
    row (RUL = 0 on fully observed training engines) is never chosen.
    """
    rng = np.random.default_rng(random_state)
    sequences = []
    targets = []

    for engine_id, engine_data in df.groupby("id"):
        engine_data = engine_data.sort_values("cycle")
        features = engine_data[feature_cols].values
        rul = engine_data["RUL"].values

        if len(features) < window_size:
            raise ValueError(
                f"Validation engine {engine_id} has fewer than {window_size} cycles."
            )

        end_idx = int(rng.integers(window_size - 1, len(features) - 1))
        sequences.append(features[end_idx - window_size + 1:end_idx + 1])
        targets.append(rul[end_idx])

    return np.asarray(sequences), np.asarray(targets)


def save_prepared_data(path, **arrays):
    """Save the prepared arrays so every model starts from exactly the same data."""
    np.savez_compressed(path, **arrays)


def save_scaler(scaler, path):
    """Persist the fitted scaler (needed later by src/predict.py)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
