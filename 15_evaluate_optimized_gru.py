import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error
)

from sklearn.preprocessing import MinMaxScaler


# ==========================================
# 1. LOAD TEST DATA
# ==========================================

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")


print("TEST DATA")

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ==========================================
# 2. LOAD ORIGINAL TRAINING DATA
# ==========================================

train_data = pd.read_csv(
    "train_data.csv"
)


print("\nTRAINING DATA FOR SCALER")

print(train_data.head())


# ==========================================
# 3. RECREATE IHSG SCALER
# ==========================================

ihsg_scaler = MinMaxScaler()

ihsg_scaler.fit(

    train_data[
        ["IHSG_Close"]
    ]

)


print("\nIHSG SCALER PARAMETERS")

print(
    "Minimum:",
    ihsg_scaler.data_min_[0]
)

print(
    "Maximum:",
    ihsg_scaler.data_max_[0]
)


# ==========================================
# 4. LOAD OPTIMIZED GRU MODEL
# ==========================================

model = load_model(
    "optimized_gru.keras"
)


print(
    "\nOPTIMIZED GRU MODEL LOADED SUCCESSFULLY"
)


# ==========================================
# 5. MAKE PREDICTIONS
# ==========================================

predictions = model.predict(
    X_test
)


# ==========================================
# 6. INVERSE TRANSFORM DATA
# ==========================================

y_test_original = ihsg_scaler.inverse_transform(

    y_test.reshape(-1, 1)

)


predictions_original = ihsg_scaler.inverse_transform(

    predictions

)


# ==========================================
# 7. CALCULATE EVALUATION METRICS
# ==========================================

rmse = np.sqrt(

    mean_squared_error(

        y_test_original,

        predictions_original

    )

)


mape = mean_absolute_percentage_error(

    y_test_original,

    predictions_original

) * 100


# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

print(
    "\nOPTIMIZED GRU EVALUATION RESULTS"
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
# 9. CREATE PREDICTION DATAFRAME
# ==========================================

results = pd.DataFrame({

    "Actual_IHSG":

        y_test_original.flatten(),

    "Predicted_IHSG":

        predictions_original.flatten()

})


print(
    "\nFIRST 10 PREDICTIONS"
)

print(
    results.head(10)
)


# ==========================================
# 10. SAVE PREDICTIONS
# ==========================================

results.to_csv(

    "optimized_gru_predictions.csv",

    index=False

)


# ==========================================
# 11. CREATE MODEL COMPARISON
# ==========================================

baseline_results = pd.read_csv(

    "baseline_model_comparison.csv"

)


optimized_result = pd.DataFrame({

    "Model":

        ["Optimized GRU"],

    "RMSE":

        [rmse],

    "MAPE (%)":

        [mape]

})


comparison = pd.concat(

    [

        baseline_results,

        optimized_result

    ],

    ignore_index=True

)


print(
    "\nFINAL MODEL COMPARISON"
)

print(
    comparison
)


# ==========================================
# 12. SAVE FINAL COMPARISON
# ==========================================

comparison.to_csv(

    "final_model_comparison.csv",

    index=False

)


print(
    "\nFILES SUCCESSFULLY SAVED:"
)

print(
    "- optimized_gru_predictions.csv"
)

print(
    "- final_model_comparison.csv"
)