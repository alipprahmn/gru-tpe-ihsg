"""23_arima.py - ARIMA benchmark. The order is selected by AIC on the training rows only
(d = 1, p and q from 0 to 3, with and without drift); parameters are then frozen and
one-step-ahead forecasts are produced for the test rows by filtering the full series.
Run for the main split and for the three walk-forward origins used in 24_walk_forward.py."""
import itertools
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from common import load_series

warnings.filterwarnings("ignore")
df = load_series()
y = df["IHSG_Close"].values
SPLITS = {"main": (2069, 2587), "wf1": (2069, 2242), "wf2": (2242, 2415), "wf3": (2415, 2587)}

rows, orders = [], []
for name, (train_end, test_end) in SPLITS.items():
    best = None
    for p, q, trend in itertools.product(range(4), range(4), ["n", "t"]):
        try:
            res = ARIMA(y[:train_end], order=(p, 1, q), trend=trend).fit()
        except Exception:
            continue
        if best is None or res.aic < best[0]:
            best = (res.aic, (p, 1, q), trend, res)
    aic, order, trend, res = best
    full = res.apply(y[:test_end])                      # same parameters, filtered over the full series
    pred = full.get_prediction(start=train_end, end=test_end - 1).predicted_mean
    orders.append({"split": name, "order": str(order), "trend": trend, "aic": aic})
    rows.append(pd.DataFrame({"split": name, "Date": df["Date"].iloc[train_end:test_end].dt.strftime("%Y-%m-%d"),
                              "actual": y[train_end:test_end], "pred": np.asarray(pred)}))
    print(name, order, trend, "AIC %.1f" % aic, flush=True)

pd.concat(rows).to_csv("arima_predictions.csv", index=False)
pd.DataFrame(orders).to_csv("arima_orders.csv", index=False)
