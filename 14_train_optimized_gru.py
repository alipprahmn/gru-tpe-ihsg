import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

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
# 3. LOAD OPTUNA RESULTS
# ==========================================

results = pd.read_csv(
    "optuna_gru_results.csv"
)


# ==========================================
# 4. GET BEST HYPERPARAMETERS
# ==========================================

best_result = results.loc[

    results[
        "value"
    ].idxmin()

]


units = int(

    best_result[
        "params_units"
    ]

)


dropout_rate = float(

    best_result[
        "params_dropout_rate"
    ]

)


learning_rate = float(

    best_result[
        "params_learning_rate"
    ]

)


batch_size = int(

    best_result[
        "params_batch_size"
    ]

)


print(
    "\nBEST OPTUNA TRIAL"
)


print(
    "Trial number:",
    int(
        best_result[
            "number"
        ]
    )
)


print(
    "Validation loss:",
    best_result[
        "value"
    ]
)


print(
    "\nBEST HYPERPARAMETERS"
)


print(
    "Units:",
    units
)


print(
    "Dropout Rate:",
    dropout_rate
)


print(
    "Learning Rate:",
    learning_rate
)


print(
    "Batch Size:",
    batch_size
)


# ==========================================
# 5. CREATE OPTIMIZER
# ==========================================

optimizer = tf.keras.optimizers.Adam(

    learning_rate=learning_rate

)


# ==========================================
# 6. BUILD OPTIMIZED GRU MODEL
# ==========================================

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


# ==========================================
# 7. COMPILE MODEL
# ==========================================

model.compile(

    optimizer=optimizer,

    loss="mse"

)


print(
    "\nOPTIMIZED GRU MODEL SUMMARY"
)


model.summary()


# ==========================================
# 8. DEFINE EARLY STOPPING
# ==========================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True

)


# ==========================================
# 9. TRAIN OPTIMIZED GRU MODEL
# ==========================================

history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    epochs=100,

    batch_size=batch_size,

    callbacks=[

        early_stopping

    ],

    verbose=1

)


# ==========================================
# 10. GET BEST VALIDATION LOSS
# ==========================================

best_validation_loss = min(

    history.history[
        "val_loss"
    ]

)


print(
    "\nTRAINING COMPLETED"
)


print(
    "Best validation loss:",
    best_validation_loss
)


# ==========================================
# 11. SAVE MODEL
# ==========================================

model.save(

    "optimized_gru.keras"

)


print(
    "\nModel saved as:"
)


print(
    "optimized_gru.keras"
)