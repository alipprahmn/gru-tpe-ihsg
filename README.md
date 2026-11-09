# Effect of TPE Optimisation on GRU Accuracy Relative to Random Walk Forecasts of IHSG

Code and data for the article submitted to *bit-Tech (Binary Digital - Technology)*.
Authors: Alief Rahman, Ani Dijah Rahajoe (corresponding author), Budi Mukhamad Mulyo,
Informatics Study Program, Universitas Pembangunan Nasional "Veteran" Jawa Timur.

## Contents

| Step | Script | Output |
|---|---|---|
| Data download | `01_download_data.py` | `IHSG.csv`, `DJIA.csv` |
| Preprocessing | `02`-`09` | `final_data.csv`, `train_data.csv`, `test_data.csv`, `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy` |
| Manually configured networks | `10_baseline_lstm.py`, `11_baseline_gru.py`, `12_evaluate_baseline_models.py` | `baseline_predictions.csv` |
| TPE search for the GRU | `13_optuna_gru.py` | `optuna_gru_results.csv` |
| GRU-TPE | `14_train_optimized_gru.py`, `15_evaluate_optimized_gru.py` | `optimized_gru_predictions.csv` |
| Original metrics and plots | `16`-`19` | comparison tables and figures |
| Benchmark analysis (Figures 1-3, Tables 4-8) | `20_benchmark_analysis.py` | `article_*.csv`, `fig_*.png` |
| TPE search for the LSTM (same space and budget) | `21_optuna_lstm.py` | `optuna_lstm_results.csv` |
| Ten-seed retraining | `22_multiseed.py` | `multiseed_predictions.csv` |
| ARIMA benchmark | `23_arima.py` | `arima_predictions.csv`, `arima_orders.csv` |
| Walk-forward evaluation | `24_walk_forward.py` | `walkforward_predictions.csv` |
| Robustness tables (Tables 9-10, Holm-adjusted DM tests) | `25_robustness_analysis.py` | `article_*.csv` |

`common.py` holds the shared data-window and model functions used by scripts 21-25.

## How to reproduce

```bash
pip install -r requirements.txt
python 01_download_data.py          # optional: the downloaded CSV files are already included
python 03_merge_data.py && python 04_create_lag.py && python 05_handle_missing.py
python 06_feature_selection.py && python 07_split_data.py && python 08_normalize_data.py
python 09_create_sliding_window.py
python 10_baseline_lstm.py && python 11_baseline_gru.py && python 12_evaluate_baseline_models.py
python 13_optuna_gru.py && python 14_train_optimized_gru.py && python 15_evaluate_optimized_gru.py
python 20_benchmark_analysis.py
python 21_optuna_lstm.py
python 22_multiseed.py LSTM LSTM-TPE GRU GRU-TPE GRU-TPE-noDJIA GRU-TPE-return
python 23_arima.py
python 24_walk_forward.py LSTM LSTM-TPE GRU GRU-TPE
python 25_robustness_analysis.py
```

Seed 42 reproduces the original predictions exactly with TensorFlow 2.21 and Keras 3.15.1 on CPU.
Other hardware or library versions may give slightly different numbers.

## Data

Daily closing values of the Indonesia Composite Index (`^JKSE`) and the Dow Jones Industrial Average (`^DJI`)
from Yahoo Finance, 1 January 2015 to 31 December 2025. The data are provided for research reproducibility only.

## Licence

Code: MIT. Article text and figures: CC BY-SA 4.0, following the journal licence.
