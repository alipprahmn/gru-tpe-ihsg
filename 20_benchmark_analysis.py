"""
20_benchmark_analysis.py

Post-hoc analysis for the journal article. The script does NOT retrain any model.
It only reads the prediction files and the Optuna results that were produced by
scripts 07-17, then computes:

  1. Accuracy metrics (RMSE, MAE, MAPE, R2) for LSTM, GRU, GRU-TPE, and a
     random-walk (persistence) benchmark, plus RMSE relative to the random walk.
  2. Diebold-Mariano tests (squared-error loss, h = 1, p-value from Student t
     with n - 1 degrees of freedom) for every pair of forecasts.
  3. Change-based diagnostics: mean absolute predicted change, correlation
     between predicted and actual change, and directional accuracy with a
     95% Wilson score interval.
  4. RMSE per test year and per volatility regime (20-day rolling standard
     deviation of daily returns, split at the median).
  5. A summary of the TPE search (objective by hyperparameter value).
  6. Figures 1-3 for the article.

Required files in the working directory:
  baseline_predictions.csv, optimized_gru_predictions.csv,
  train_data.csv, test_data.csv, optuna_gru_results.csv
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================
# 1. LOAD DATA
# ==========================================

baseline = pd.read_csv("baseline_predictions.csv")
optimized = pd.read_csv("optimized_gru_predictions.csv")
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")
optuna_results = pd.read_csv("optuna_gru_results.csv")

y = baseline["Actual_IHSG"].values
assert np.allclose(y, test_data["IHSG_Close"].values), "Actual values do not match test_data.csv"

# Random-walk benchmark: forecast for day t is the actual value of day t-1.
# The first test day uses the last training observation.
previous_actual = np.r_[train_data["IHSG_Close"].values[-1], y[:-1]]

forecasts = {
    "Random walk": previous_actual,
    "LSTM": baseline["LSTM_Prediction"].values,
    "GRU": baseline["GRU_Prediction"].values,
    "GRU-TPE": optimized["Predicted_IHSG"].values,
}
models = ["LSTM", "GRU", "GRU-TPE"]
n = len(y)


# ==========================================
# 2. ACCURACY METRICS
# ==========================================

def rmse(e):
    return np.sqrt(np.mean(e ** 2))

rw_rmse = rmse(y - previous_actual)

rows = []
for name, pred in forecasts.items():
    e = y - pred
    rows.append({
        "Model": name,
        "RMSE": rmse(e),
        "MAE": np.mean(np.abs(e)),
        "MAPE (%)": 100 * np.mean(np.abs(e / y)),
        "R2": 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2),
        "RMSE / RMSE_RW": rmse(e) / rw_rmse,
        "Mean error": e.mean(),
    })
accuracy = pd.DataFrame(rows)
print("\nACCURACY ON THE TEST SET (n = %d)" % n)
print(accuracy.round(4).to_string(index=False))
accuracy.to_csv("article_accuracy.csv", index=False)


# ==========================================
# 3. DIEBOLD-MARIANO TESTS
# ==========================================

def diebold_mariano(actual, pred_a, pred_b):
    """Positive statistic: model B is more accurate than model A."""
    d = (actual - pred_a) ** 2 - (actual - pred_b) ** 2
    m = len(d)
    statistic = d.mean() / np.sqrt(np.var(d) / m)
    p_value = 2 * (1 - stats.t.cdf(abs(statistic), df=m - 1))
    return statistic, p_value

pairs = [
    ("LSTM", "GRU"), ("GRU", "GRU-TPE"), ("LSTM", "GRU-TPE"),
    ("Random walk", "LSTM"), ("Random walk", "GRU"), ("Random walk", "GRU-TPE"),
]
dm_rows = []
for a, b in pairs:
    s, p = diebold_mariano(y, forecasts[a], forecasts[b])
    dm_rows.append({"Model A": a, "Model B": b, "DM": s, "p-value": p})
dm_table = pd.DataFrame(dm_rows)
print("\nDIEBOLD-MARIANO TESTS")
print(dm_table.round(4).to_string(index=False))
dm_table.to_csv("article_dm_tests.csv", index=False)


# ==========================================
# 4. CHANGE-BASED DIAGNOSTICS AND DIRECTIONAL ACCURACY
# ==========================================

def wilson_interval(k, m, z=1.959964):
    p = k / m
    centre = (p + z ** 2 / (2 * m)) / (1 + z ** 2 / m)
    half = z * np.sqrt(p * (1 - p) / m + z ** 2 / (4 * m ** 2)) / (1 + z ** 2 / m)
    return 100 * (centre - half), 100 * (centre + half)

actual_change = y - previous_actual
diag_rows = []
for name in models:
    pred_change = forecasts[name] - previous_actual
    # directional accuracy from the second test day onward (n - 1 days)
    correct = (np.sign(pred_change[1:]) == np.sign(actual_change[1:]))
    k, m = int(correct.sum()), len(correct)
    lo, hi = wilson_interval(k, m)
    diag_rows.append({
        "Model": name,
        "Mean |predicted change|": np.mean(np.abs(pred_change)),
        "Corr(predicted change, actual change)": np.corrcoef(pred_change, actual_change)[0, 1],
        "Correct directions": k,
        "Days": m,
        "DA (%)": 100 * k / m,
        "CI low (%)": lo,
        "CI high (%)": hi,
        "Binomial p (vs 50%)": stats.binomtest(k, m, 0.5).pvalue,
    })
diagnostics = pd.DataFrame(diag_rows)

# MSE decomposition relative to the random walk:
# MSE_model = MSE_RW + E[pc^2] - 2 E[dy * pc], where pc = predicted change, dy = actual change
dec_rows = []
for name in models:
    pc = forecasts[name] - previous_actual
    dec_rows.append({
        "Model": name,
        "MSE_RW": np.mean(actual_change ** 2),
        "E[pc^2]": np.mean(pc ** 2),
        "2E[dy*pc]": 2 * np.mean(actual_change * pc),
        "MSE_model": np.mean((y - forecasts[name]) ** 2),
    })
decomposition = pd.DataFrame(dec_rows)
print("\nMSE DECOMPOSITION RELATIVE TO THE RANDOM WALK")
print(decomposition.round(1).to_string(index=False))
decomposition.to_csv("article_mse_decomposition.csv", index=False)
print("\nMean |actual change|: %.2f index points" % np.mean(np.abs(actual_change)))
print("CHANGE-BASED DIAGNOSTICS")
print(diagnostics.round(4).to_string(index=False))
diagnostics.to_csv("article_change_diagnostics.csv", index=False)


# ==========================================
# 5. RMSE BY YEAR AND BY VOLATILITY REGIME
# ==========================================

dates = pd.to_datetime(test_data["Date"]).values
years = pd.DatetimeIndex(dates).year.values

returns = pd.Series(y).pct_change()
volatility = returns.rolling(20).std()          # sample std (ddof = 1)
median_vol = volatility.median()
has_vol = volatility.notna().values
high = has_vol & (volatility.values > median_vol)
low = has_vol & (volatility.values <= median_vol)

segments = [("Year 2023", years == 2023), ("Year 2024", years == 2024), ("Year 2025", years == 2025),
            ("Low volatility", low), ("High volatility", high)]
seg_rows = []
for label, mask in segments:
    row = {"Segment": label, "n": int(mask.sum())}
    for name, pred in forecasts.items():
        row[name] = rmse((y - pred)[mask])
    seg_rows.append(row)
segment_table = pd.DataFrame(seg_rows)
print("\nMedian 20-day volatility: %.4f%%" % (100 * median_vol))
print("RMSE BY SEGMENT")
print(segment_table.round(2).to_string(index=False))
segment_table.to_csv("article_segments.csv", index=False)


# ==========================================
# 6. TPE SEARCH SUMMARY
# ==========================================

opt = optuna_results.copy()
opt["duration_s"] = pd.to_timedelta(opt["duration"]).dt.total_seconds()
total_time = pd.to_datetime(opt["datetime_complete"]).max() - pd.to_datetime(opt["datetime_start"]).min()
best = opt.loc[opt["value"].idxmin()]
print("\nTPE SEARCH")
print("Total time:", total_time, "| mean trial: %.1f s" % opt["duration_s"].mean())
print("Best trial:", int(best["number"]), "value: %.9f" % best["value"])
print("Top-10 trials within %.2f%% of the best value"
      % (100 * (opt["value"].nsmallest(10).max() / opt["value"].min() - 1)))
for col in ["params_units", "params_dropout_rate", "params_batch_size"]:
    print(opt.groupby(col)["value"].agg(["count", "median", "min"]).to_string())
opt["lr_range"] = pd.cut(opt["params_learning_rate"], [1e-4, 1e-3, 3e-3, 1e-2])
print(opt.groupby("lr_range", observed=False)["value"].agg(["count", "median", "min"]).to_string())


# ==========================================
# 7. FIGURES
# ==========================================

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.linewidth": 0.8})
GREY, SLATE, RED, BLACK = "#9ca0a8", "#4a5468", "#c0392b", "#222222"
colours = {"Random walk": BLACK, "LSTM": GREY, "GRU": SLATE, "GRU-TPE": RED}

# ---- Figure 1: objective value by hyperparameter value
fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.2), dpi=300, sharey=True)
jitter = np.random.default_rng(0)
vals = opt["value"] * 1e4
for ax, col, label in zip(axes[:3],
                          ["params_units", "params_dropout_rate", "params_batch_size"],
                          ["Units", "Dropout rate", "Batch size"]):
    levels = sorted(opt[col].unique())
    for i, lv in enumerate(levels):
        sel = opt[col] == lv
        ax.scatter(i + jitter.uniform(-0.18, 0.18, sel.sum()), vals[sel], s=7, color=GREY, alpha=0.8, lw=0)
        ax.hlines(np.median(vals[sel]), i - 0.3, i + 0.3, color=BLACK, lw=1.2)
    ax.set_xticks(range(len(levels)), [("%g" % lv) for lv in levels])
    ax.set_xlabel(label)
ax = axes[3]
ax.scatter(opt["params_learning_rate"], vals, s=7, color=GREY, alpha=0.8, lw=0)
ax.scatter(best["params_learning_rate"], best["value"] * 1e4, s=22, facecolor="none", edgecolor=RED, lw=1.2)
ax.set_xscale("log"); ax.set_xlabel("Learning rate")
axes[0].set_ylabel("Validation loss (×10⁻⁴)")
axes[0].set_ylim(2.0, 4.6)
for a in axes:
    a.grid(axis="y", ls=":", lw=0.6, color="#bbbbbb")
    a.tick_params(length=3, width=0.8, labelsize=8)
fig.tight_layout(w_pad=0.6)
fig.savefig("fig_tpe_landscape.png", dpi=300)
plt.close(fig)

# ---- Figure 2: predicted change versus actual change
fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.35), dpi=300, sharex=True, sharey=True)
lim = 260
for ax, name in zip(axes, models):
    pc = forecasts[name] - previous_actual
    ax.scatter(actual_change, pc, s=5, color=colours[name], alpha=0.6, lw=0)
    ax.axhline(0, color="#888888", lw=0.6); ax.axvline(0, color="#888888", lw=0.6)
    ax.plot([-lim, lim], [-lim, lim], ls="--", lw=0.7, color=BLACK)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("Actual change (points)")
    ax.tick_params(length=3, width=0.8, labelsize=8)
axes[0].set_ylabel("Predicted change (points)")
fig.tight_layout(w_pad=0.6)
fig.savefig("fig_change_scatter.png", dpi=300)
plt.close(fig)

# ---- Figure 3: RMSE by year and by volatility regime, including the random walk
order = ["Random walk", "LSTM", "GRU", "GRU-TPE"]
fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.5), dpi=300,
                           gridspec_kw={"width_ratios": [3, 2]})
w = 0.2
x = np.arange(3)
for i, k in enumerate(order):
    a.bar(x + (i - 1.5) * w, segment_table[k].values[:3], w, color=colours[k], label=k, zorder=3)
a.set_xticks(x, ["2023", "2024", "2025"]); a.set_xlabel("Test year")
a.set_ylabel("RMSE (index points)"); a.set_title("(a) By test year", fontsize=9)
a.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
a.set_ylim(0, 125)
x2 = np.arange(2)
for i, k in enumerate(order):
    b.bar(x2 + (i - 1.5) * w, segment_table[k].values[3:], w, color=colours[k], zorder=3)
b.set_xticks(x2, ["Low volatility", "High volatility"]); b.set_title("(b) By volatility regime", fontsize=9)
b.set_ylim(0, 125)
for ax in (a, b):
    ax.grid(axis="y", ls=":", lw=0.6, color="#bbbbbb", zorder=0)
    ax.tick_params(length=3, width=0.8, labelsize=8)
fig.tight_layout(w_pad=1.5)
fig.savefig("fig_segments.png", dpi=300)
plt.close(fig)

print("\nSaved: article_accuracy.csv, article_dm_tests.csv, article_change_diagnostics.csv,")
print("       article_segments.csv, fig_tpe_landscape.png, fig_change_scatter.png, fig_segments.png")
