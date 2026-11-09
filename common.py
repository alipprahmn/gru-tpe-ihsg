"""Shared helpers for the robustness experiments (21-24)."""
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler

WINDOW = 30
SEEDS = [42, 7, 13, 21, 99, 123, 256, 512, 777, 2024]

# Manually set configuration (scripts 10 and 11) and GRU-TPE configuration (trial 51 of 13_optuna_gru.py)
CONFIGS = {
    "LSTM":    dict(cell="LSTM", units=64, dropout=0.2, lr=0.001, batch=64),
    "GRU":     dict(cell="GRU",  units=64, dropout=0.2, lr=0.001, batch=64),
    "GRU-TPE": dict(cell="GRU",  units=64, dropout=0.1, lr=0.0018829770816374, batch=32),
}


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_series():
    df = pd.read_csv("final_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def make_level_windows(df, train_end, test_start, test_end, use_djia=True):
    """Level formulation used in the thesis: Min-Max scaled IHSG_Close and DJIA_Lag1,
    scalers fitted on rows [0, train_end). Test samples cover rows [test_start, test_end)."""
    cols = ["IHSG_Close", "DJIA_Lag1"] if use_djia else ["IHSG_Close"]
    scalers = {c: MinMaxScaler().fit(df.loc[:train_end - 1, [c]]) for c in cols}
    scaled = np.column_stack([scalers[c].transform(df[[c]]).ravel() for c in cols])
    target = scaled[:, 0]
    X_tr = np.array([scaled[i - WINDOW:i] for i in range(WINDOW, train_end)])
    y_tr = target[WINDOW:train_end]
    X_te = np.array([scaled[i - WINDOW:i] for i in range(test_start, test_end)])
    inverse = lambda v: scalers["IHSG_Close"].inverse_transform(np.asarray(v).reshape(-1, 1)).ravel()
    return X_tr, y_tr, X_te, inverse


def make_return_windows(df, train_end, test_start, test_end):
    """Return formulation: inputs are daily returns of IHSG and DJIA_Lag1 (standardised on the
    training rows), the target is the next-day IHSG return; levels are rebuilt as y_{t-1}(1 + r_hat)."""
    r = df[["IHSG_Close", "DJIA_Lag1"]].pct_change()
    r.iloc[0] = 0.0
    scaler = StandardScaler().fit(r.iloc[1:train_end])
    scaled = scaler.transform(r)
    X_tr = np.array([scaled[i - WINDOW:i] for i in range(WINDOW + 1, train_end)])
    y_tr = scaled[WINDOW + 1:train_end, 0]
    X_te = np.array([scaled[i - WINDOW:i] for i in range(test_start, test_end)])
    mu, sd = scaler.mean_[0], scaler.scale_[0]
    prev = df["IHSG_Close"].values[test_start - 1:test_end - 1]
    to_level = lambda v: prev * (1 + (np.asarray(v).ravel() * sd + mu))
    return X_tr, y_tr, X_te, to_level


def build_model(cfg, n_features):
    cell = GRU if cfg["cell"] == "GRU" else LSTM
    model = Sequential([Input(shape=(WINDOW, n_features)), cell(cfg["units"]), Dropout(cfg["dropout"]), Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["lr"]), loss="mse")
    return model


def train_predict(cfg, X_tr, y_tr, X_te, seed):
    set_seed(seed)
    model = build_model(cfg, X_tr.shape[2])
    history = model.fit(X_tr, y_tr, validation_split=0.2, epochs=100, batch_size=cfg["batch"],
                        callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
                        verbose=0)
    return model.predict(X_te, verbose=0).ravel(), min(history.history["val_loss"])
