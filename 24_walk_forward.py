"""24_walk_forward.py - expanding-window walk-forward evaluation over the test period.
The 518 test days are split into three blocks; before each block every network is retrained
(scalers refitted) on all data available up to the block start, with three seeds.
Hyperparameters stay fixed at the values tuned on the original training set only."""
import os
import sys
import time
import pandas as pd
from common import SEEDS, CONFIGS, load_series, make_level_windows, train_predict

ORIGINS = [(2069, 2242), (2242, 2415), (2415, 2587)]
df = load_series()
models = sys.argv[1:]

def config_for(name):
    if name == "LSTM-TPE":
        res = pd.read_csv("optuna_lstm_results.csv")
        b = res.loc[res["value"].idxmin()]
        return dict(cell="LSTM", units=int(b["params_units"]), dropout=float(b["params_dropout_rate"]),
                    lr=float(b["params_learning_rate"]), batch=int(b["params_batch_size"]))
    return CONFIGS[name]

out_file = "walkforward_predictions.csv"
for name in models:
    cfg = config_for(name)
    for k, (start, end) in enumerate(ORIGINS, 1):
        X_tr, y_tr, X_te, inverse = make_level_windows(df, start, start, end)
        for seed in SEEDS[:3]:
            t0 = time.time()
            pred, _ = train_predict(cfg, X_tr, y_tr, X_te, seed)
            pd.DataFrame({"model": name, "block": k, "seed": seed,
                          "Date": df["Date"].iloc[start:end].dt.strftime("%Y-%m-%d"),
                          "actual": df["IHSG_Close"].iloc[start:end].values, "pred": inverse(pred)}
                         ).to_csv(out_file, mode="a", header=not os.path.exists(out_file), index=False)
            print(name, k, seed, "%.1fs" % (time.time() - t0), flush=True)
