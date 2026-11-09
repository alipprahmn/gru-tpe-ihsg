"""
25_robustness_analysis.py

Aggregates the robustness experiments for the article:
  - original run (seed 42, scripts 10-15) plus ARIMA and random walk, with Diebold-Mariano
    tests and Holm-adjusted p-values;
  - 10-seed results of every configuration (script 22);
  - paired comparisons across seeds;
  - expanding-window walk-forward results (script 24) with ARIMA re-estimated per block (script 23).

Required files: baseline_predictions.csv, optimized_gru_predictions.csv, train_data.csv,
multiseed_predictions.csv, walkforward_predictions.csv, arima_predictions.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

rmse = lambda e: float(np.sqrt(np.mean(np.square(e))))


def dm_test(actual, pred_a, pred_b):
    """Positive statistic: model B is more accurate than model A (squared-error loss, h = 1)."""
    d = (actual - pred_a) ** 2 - (actual - pred_b) ** 2
    s = d.mean() / np.sqrt(np.var(d) / len(d))
    return s, 2 * (1 - stats.t.cdf(abs(s), df=len(d) - 1))


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


# ------------------------------------------------------------------ 1. original run
base = pd.read_csv("baseline_predictions.csv")
opt = pd.read_csv("optimized_gru_predictions.csv")
train = pd.read_csv("train_data.csv")
arima = pd.read_csv("arima_predictions.csv")
y = base["Actual_IHSG"].values
prev = np.r_[train["IHSG_Close"].values[-1], y[:-1]]
orig = {"Random walk": prev,
        "ARIMA": arima.loc[arima["split"] == "main", "pred"].values,
        "LSTM": base["LSTM_Prediction"].values,
        "GRU": base["GRU_Prediction"].values,
        "GRU-TPE": opt["Predicted_IHSG"].values}
_ms = pd.read_csv("multiseed_predictions.csv")
orig["LSTM-TPE"] = _ms[(_ms["model"] == "LSTM-TPE") & (_ms["seed"] == 42)]["pred"].values
rw = rmse(y - prev)
rows = []
for k, p in orig.items():
    e = y - p
    rows.append({"Method": k, "RMSE": rmse(e), "MAE": np.mean(np.abs(e)), "MAPE (%)": 100 * np.mean(np.abs(e / y)),
                 "R2": 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2), "U": rmse(e) / rw})
print("ORIGINAL RUN (seed 42) WITH BENCHMARKS")
print(pd.DataFrame(rows).round(4).to_string(index=False))

pairs = [("LSTM", "LSTM-TPE"), ("GRU", "GRU-TPE"), ("LSTM-TPE", "GRU-TPE"),
         ("Random walk", "ARIMA"), ("Random walk", "LSTM"), ("Random walk", "LSTM-TPE"),
         ("Random walk", "GRU"), ("Random walk", "GRU-TPE")]
dm = [(a, b) + dm_test(y, orig[a], orig[b]) for a, b in pairs]
dm = pd.DataFrame(dm, columns=["A", "B", "DM", "p"])
dm["p_Holm"] = holm(dm["p"])
print("\nDIEBOLD-MARIANO TESTS, ORIGINAL RUN (Holm-adjusted over all pairs)")
print(dm.round(4).to_string(index=False))
dm.to_csv("article_dm_holm.csv", index=False)

# ------------------------------------------------------------------ 2. ten seeds
ms = pd.read_csv("multiseed_predictions.csv")
ms_rows = []
per_seed = []
for (model, seed), g in ms.groupby(["model", "seed"], sort=False):
    a, p = g["actual"].values, g["pred"].values
    assert np.allclose(a, y)
    s, pv = dm_test(a, prev, p)
    corr_dir = (np.sign((p - prev)[1:]) == np.sign((a - prev)[1:])).mean() * 100
    per_seed.append({"model": model, "seed": seed, "RMSE": rmse(a - p), "U": rmse(a - p) / rw,
                     "DM_vs_RW": s, "p_vs_RW": pv, "mean_abs_change": np.mean(np.abs(p - prev)), "DA": corr_dir})
per_seed = pd.DataFrame(per_seed)
per_seed.to_csv("article_multiseed_per_seed.csv", index=False)
order = ["LSTM", "LSTM-TPE", "GRU", "GRU-TPE", "GRU-TPE-noDJIA", "GRU-TPE-return"]
summary = (per_seed.groupby("model")
           .agg(RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"), U_mean=("U", "mean"),
                U_min=("U", "min"), U_max=("U", "max"),
                seeds_beating_RW=("U", lambda u: int((u < 1).sum())),
                seeds_sig_worse=("p_vs_RW", lambda p: None),
                change_mean=("mean_abs_change", "mean"), DA_mean=("DA", "mean"))
           .reindex([m for m in order if m in set(per_seed["model"])]))
summary["seeds_sig_worse"] = [int(((per_seed.model == m) & (per_seed.DM_vs_RW < 0) & (per_seed.p_vs_RW < 0.05)).sum())
                              for m in summary.index]
summary["seeds_sig_better"] = [int(((per_seed.model == m) & (per_seed.DM_vs_RW > 0) & (per_seed.p_vs_RW < 0.05)).sum())
                               for m in summary.index]
print("\nTEN-SEED SUMMARY (main split; random walk RMSE = %.3f)" % rw)
print(summary.round(3).to_string())
summary.to_csv("article_multiseed_summary.csv")

print("\nPAIRED COMPARISONS ACROSS SEEDS (RMSE of A minus RMSE of B)")
piv = per_seed.pivot(index="seed", columns="model", values="RMSE")
for a, b in [("GRU", "GRU-TPE"), ("LSTM", "LSTM-TPE"), ("LSTM-TPE", "GRU-TPE"),
             ("GRU-TPE", "GRU-TPE-noDJIA"), ("GRU-TPE", "GRU-TPE-return")]:
    if a in piv and b in piv:
        diff = piv[a] - piv[b]
        w = stats.wilcoxon(piv[a], piv[b])
        print("%-10s vs %-15s mean diff %7.3f  sd %6.3f  wins %d/10  Wilcoxon p = %.4f"
              % (a, b, diff.mean(), diff.std(), int((diff > 0).sum()), w.pvalue))

# ------------------------------------------------------------------ 3. walk-forward
wf = pd.read_csv("walkforward_predictions.csv")
wf_arima = arima[arima["split"].isin(["wf1", "wf2", "wf3"])]
assert np.allclose(wf_arima["actual"].values, y)
wf_rows = [{"Method": "Random walk", "RMSE_mean": rw, "RMSE_sd": 0.0, "U_mean": 1.0},
           {"Method": "ARIMA (re-estimated per block)", "RMSE_mean": rmse(y - wf_arima["pred"].values),
            "RMSE_sd": 0.0, "U_mean": rmse(y - wf_arima["pred"].values) / rw}]
for model, g in wf.groupby("model", sort=False):
    r = []
    for seed, gs in g.groupby("seed"):
        gs = gs.sort_values("Date")
        assert np.allclose(gs["actual"].values, y)
        r.append(rmse(y - gs["pred"].values))
    wf_rows.append({"Method": model, "RMSE_mean": np.mean(r), "RMSE_sd": np.std(r, ddof=1), "U_mean": np.mean(r) / rw})
wf_table = pd.DataFrame(wf_rows)
s_ar, p_ar = dm_test(y, prev, wf_arima["pred"].values)
print("DM random walk vs walk-forward ARIMA: %.3f (p = %.4f)" % (s_ar, p_ar))
print("\nWALK-FORWARD (three expanding-window blocks, three seeds per network)")
print(wf_table.round(3).to_string(index=False))
wf_table.to_csv("article_walkforward.csv", index=False)
