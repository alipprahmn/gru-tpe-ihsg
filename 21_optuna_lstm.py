"""21_optuna_lstm.py - TPE search for the LSTM with exactly the same search space,
objective, budget (100 trials), and sampler seed as 13_optuna_gru.py."""
import numpy as np
import optuna
from tensorflow.keras.callbacks import EarlyStopping
from common import set_seed, build_model

SEED = 42
set_seed(SEED)
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")


def objective(trial):
    cfg = dict(cell="LSTM",
               units=trial.suggest_int("units", 32, 128, step=32),
               dropout=trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1),
               lr=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
               batch=trial.suggest_categorical("batch_size", [16, 32, 64]))
    # as in 13_optuna_gru.py, the seed is set once at the start, not inside each trial
    model = build_model(cfg, X_train.shape[2])
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=cfg["batch"],
                        callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
                        verbose=0)
    return min(history.history["val_loss"])


study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=100)
print("Best trial:", study.best_trial.number, "value:", study.best_value, study.best_params)
study.trials_dataframe().to_csv("optuna_lstm_results.csv", index=False)
