import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import optuna

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping


# ==========================================
# 1. SET RANDOM SEED
# ==========================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)

np.random.seed(SEED)

tf.keras.utils.set_random_seed(SEED)


print(
    "Random seed set to:",
    SEED
)


# ==========================================
# 2. LOAD TRAINING DATA
# ==========================================

X_train = np.load(
    "X_train.npy"
)

y_train = np.load(
    "y_train.npy"
)


print(
    "\nTRAINING DATA"
)

print(
    "X_train shape:",
    X_train.shape
)

print(
    "y_train shape:",
    y_train.shape
)


# ==========================================
# 3. DEFINE OBJECTIVE FUNCTION
# ==========================================

def objective(trial):


    # ======================================
    # HYPERPARAMETER SEARCH SPACE
    # ======================================

    units = trial.suggest_int(
        "units",
        32,
        128,
        step=32
    )


    dropout_rate = trial.suggest_float(
        "dropout_rate",
        0.1,
        0.5,
        step=0.1
    )


    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-4,
        1e-2,
        log=True
    )


    batch_size = trial.suggest_categorical(
        "batch_size",
        [16, 32, 64]
    )


    # ======================================
    # BUILD GRU MODEL
    # ======================================

    model = Sequential([

        Input(
            shape=(
                X_train.shape[1],
                X_train.shape[2]
            )
        ),

        GRU(
            units=units
        ),

        Dropout(
            dropout_rate
        ),

        Dense(
            1
        )

    ])


    # ======================================
    # COMPILE MODEL
    # ======================================

    optimizer = tf.keras.optimizers.Adam(

        learning_rate=learning_rate

    )


    model.compile(

        optimizer=optimizer,

        loss="mse"

    )


    # ======================================
    # EARLY STOPPING
    # ======================================

    early_stopping = EarlyStopping(

        monitor="val_loss",

        patience=10,

        restore_best_weights=True

    )


    # ======================================
    # TRAIN MODEL
    # ======================================

    history = model.fit(

        X_train,

        y_train,

        validation_split=0.2,

        epochs=100,

        batch_size=batch_size,

        callbacks=[
            early_stopping
        ],

        verbose=0

    )


    # ======================================
    # GET BEST VALIDATION LOSS
    # ======================================

    best_val_loss = min(

        history.history[
            "val_loss"
        ]

    )


    return best_val_loss


# ==========================================
# 4. CREATE OPTUNA STUDY
# ==========================================

study = optuna.create_study(

    direction="minimize",

    sampler=optuna.samplers.TPESampler(
        seed=SEED
    )

)


# ==========================================
# 5. START OPTIMIZATION
# ==========================================

print(
    "\nSTARTING HYPERPARAMETER OPTIMIZATION..."
)


study.optimize(

    objective,

    n_trials=100

)


# ==========================================
# 6. DISPLAY BEST RESULTS
# ==========================================

print(
    "\nOPTIMIZATION COMPLETED"
)


print(
    "\nBEST TRIAL:"
)

print(
    "Trial Number:",
    study.best_trial.number
)


print(
    "Validation Loss:",
    study.best_value
)


print(
    "\nBEST HYPERPARAMETERS:"
)


for key, value in study.best_params.items():

    print(
        key,
        ":",
        value
    )


# ==========================================
# 7. SAVE ALL OPTUNA RESULTS
# ==========================================

results = study.trials_dataframe()


results.to_csv(

    "optuna_gru_results.csv",

    index=False

)


print(
    "\nOPTUNA RESULTS SAVED AS:"
)

print(
    "optuna_gru_results.csv"
)