"""22_multiseed.py - retrain each configuration with 10 random seeds on the main split
(train rows 0-2068, test rows 2069-2586) and save every test prediction.
Usage: python 22_multiseed.py LSTM GRU GRU-TPE GRU-TPE-noDJIA GRU-TPE-return LSTM-TPE"""
import os
import sys
import time
import pandas as pd
from common import (SEEDS, CONFIGS, load_series, make_level_windows, make_return_windows, train_predict)

TRAIN_END, TEST_END = 2069, 2587
df = load_series()
dates = df["Date"].iloc[TRAIN_END:TEST_END].dt.strftime("%Y-%m-%d").values
actual = df["IHSG_Close"].iloc[TRAIN_END:TEST_END].values


def config_for(name):
    if name == "LSTM-TPE":
        res = pd.read_csv("optuna_lstm_results.csv")
        best = res.loc[res["value"].idxmin()]
        return dict(cell="LSTM", units=int(best["params_units"]), dropout=float(best["params_dropout_rate"]),
                    lr=float(best["params_learning_rate"]), batch=int(best["params_batch_size"]))
    return CONFIGS[name.replace("-noDJIA", "").replace("-return", "")]


out_file = "multiseed_predictions.csv"
for name in sys.argv[1:]:
    cfg = config_for(name)
    if name.endswith("-return"):
        X_tr, y_tr, X_te, back = make_return_windows(df, TRAIN_END, TRAIN_END, TEST_END)
    else:
        X_tr, y_tr, X_te, back = make_level_windows(df, TRAIN_END, TRAIN_END, TEST_END,
                                                    use_djia=not name.endswith("-noDJIA"))
    for seed in SEEDS:
        t0 = time.time()
        pred, val = train_predict(cfg, X_tr, y_tr, X_te, seed)
        rows = pd.DataFrame({"model": name, "seed": seed, "Date": dates, "actual": actual,
                             "pred": back(pred), "val_loss": val})
        rows.to_csv(out_file, mode="a", header=not os.path.exists(out_file), index=False)
        print(name, seed, "%.1fs" % (time.time() - t0), flush=True)
