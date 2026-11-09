import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error
)


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
# 3. LOAD TESTING DATA
# ==========================================

X_test = np.load(
    "X_test.npy"
)

y_test = np.load(
    "y_test.npy"
)


print(
    "\nTESTING DATA"
)

print(
    "X_test shape:",
    X_test.shape
)

print(
    "y_test shape:",
    y_test.shape
)


# ==========================================
# 4. BUILD BASELINE LSTM MODEL
# ==========================================

model = Sequential([

    Input(
        shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    LSTM(
        64
    ),

    Dropout(
        0.2
    ),

    Dense(
        1
    )

])


# ==========================================
# 5. COMPILE MODEL
# ==========================================

model.compile(

    optimizer="adam",

    loss="mse"

)


print(
    "\nBASELINE LSTM MODEL SUMMARY"
)

model.summary()


# ==========================================
# 6. DEFINE EARLY STOPPING
# ==========================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True

)


# ==========================================
# 7. TRAIN MODEL
# ==========================================

history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    epochs=100,

    batch_size=64,

    callbacks=[

        early_stopping

    ],

    verbose=1

)


# ==========================================
# 8. GET BEST VALIDATION LOSS
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
    "\nBest validation loss:"
)

print(
    best_validation_loss
)


# ==========================================
# 9. SAVE MODEL
# ==========================================

model.save(

    "baseline_lstm.keras"

)


print(
    "\nModel saved as:"
)

print(
    "baseline_lstm.keras"
)


# ==========================================
# 10. LOAD TRAINING DATA FOR SCALER
# ==========================================

train_data = pd.read_csv(

    "train_data.csv"

)


print(
    "\nTRAINING DATA FOR SCALER"
)

print(
    train_data.head()
)


# ==========================================
# 11. CREATE TARGET SCALER
# ==========================================

target_scaler = MinMaxScaler()


target_scaler.fit(

    train_data[
        ["IHSG_Close"]
    ]

)


print(
    "\nIHSG SCALER PARAMETERS"
)

print(
    "Minimum:",
    target_scaler.data_min_[0]
)

print(
    "Maximum:",
    target_scaler.data_max_[0]
)


# ==========================================
# 12. MAKE PREDICTIONS
# ==========================================

y_pred_scaled = model.predict(

    X_test

)


# ==========================================
# 13. RESHAPE ACTUAL VALUES
# ==========================================

y_test_reshaped = y_test.reshape(

    -1,

    1

)


# ==========================================
# 14. INVERSE TRANSFORM
# ==========================================

y_actual = target_scaler.inverse_transform(

    y_test_reshaped

)


y_pred = target_scaler.inverse_transform(

    y_pred_scaled

)


# ==========================================
# 15. CALCULATE RMSE
# ==========================================

rmse = np.sqrt(

    mean_squared_error(

        y_actual,

        y_pred

    )

)


# ==========================================
# 16. CALCULATE MAPE
# ==========================================

mape = (

    mean_absolute_percentage_error(

        y_actual,

        y_pred

    )

    * 100

)


# ==========================================
# 17. DISPLAY EVALUATION RESULTS
# ==========================================

print(
    "\nBASELINE LSTM EVALUATION RESULTS"
)


print(
    "RMSE:",
    rmse
)


print(
    "MAPE:",
    mape,
    "%"
)


# ==========================================
# 18. CREATE PREDICTION DATAFRAME
# ==========================================

predictions = pd.DataFrame({

    "Actual_IHSG":

        y_actual.flatten(),

    "Predicted_IHSG":

        y_pred.flatten()

})


print(
    "\nFIRST 10 PREDICTIONS"
)


print(

    predictions.head(

        10

    )

)


# ==========================================
# 19. SAVE PREDICTIONS
# ==========================================

predictions.to_csv(

    "baseline_lstm_predictions.csv",

    index=False

)


print(
    "\nPREDICTIONS SAVED AS:"
)

print(
    "baseline_lstm_predictions.csv"
)