import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# =========================
# LOAD TEST DATA
# =========================

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("TEST DATA")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# =========================
# LOAD ORIGINAL TRAINING DATA
# FOR INVERSE NORMALIZATION
# =========================

train_data = pd.read_csv("train_data.csv")

ihsg_min = train_data["IHSG_Close"].min()
ihsg_max = train_data["IHSG_Close"].max()

print("\nIHSG SCALER PARAMETERS")
print("Minimum:", ihsg_min)
print("Maximum:", ihsg_max)

# =========================
# INVERSE TRANSFORM FUNCTION
# =========================

def inverse_transform(values):
    return values * (ihsg_max - ihsg_min) + ihsg_min

# =========================
# LOAD MODELS
# =========================

lstm_model = load_model("baseline_lstm.keras")
gru_model = load_model("baseline_gru.keras")

print("\nMODELS LOADED SUCCESSFULLY")

# =========================
# MAKE PREDICTIONS
# =========================

lstm_predictions = lstm_model.predict(X_test)
gru_predictions = gru_model.predict(X_test)

# Convert predictions to 1D
lstm_predictions = lstm_predictions.flatten()
gru_predictions = gru_predictions.flatten()

# =========================
# INVERSE NORMALIZATION
# =========================

y_test_original = inverse_transform(y_test)

lstm_predictions_original = inverse_transform(
    lstm_predictions
)

gru_predictions_original = inverse_transform(
    gru_predictions
)

# =========================
# CALCULATE RMSE
# =========================

lstm_rmse = np.sqrt(
    mean_squared_error(
        y_test_original,
        lstm_predictions_original
    )
)

gru_rmse = np.sqrt(
    mean_squared_error(
        y_test_original,
        gru_predictions_original
    )
)

# =========================
# CALCULATE MAPE
# =========================

lstm_mape = (
    mean_absolute_percentage_error(
        y_test_original,
        lstm_predictions_original
    )
    * 100
)

gru_mape = (
    mean_absolute_percentage_error(
        y_test_original,
        gru_predictions_original
    )
    * 100
)

# =========================
# DISPLAY RESULTS
# =========================

print("\nBASELINE MODEL EVALUATION RESULTS")

print("\nLSTM")
print("RMSE:", lstm_rmse)
print("MAPE:", lstm_mape, "%")

print("\nGRU")
print("RMSE:", gru_rmse)
print("MAPE:", gru_mape, "%")

# =========================
# CREATE COMPARISON TABLE
# =========================

comparison = pd.DataFrame({
    "Model": ["LSTM", "GRU"],
    "RMSE": [lstm_rmse, gru_rmse],
    "MAPE (%)": [lstm_mape, gru_mape]
})

print("\nMODEL COMPARISON")
print(comparison)

# =========================
# SAVE COMPARISON RESULTS
# =========================

comparison.to_csv(
    "baseline_model_comparison.csv",
    index=False
)

# =========================
# SAVE PREDICTIONS
# =========================

predictions = pd.DataFrame({
    "Actual_IHSG": y_test_original,
    "LSTM_Prediction": lstm_predictions_original,
    "GRU_Prediction": gru_predictions_original
})

predictions.to_csv(
    "baseline_predictions.csv",
    index=False
)

print("\nFILES SUCCESSFULLY SAVED:")
print("- baseline_model_comparison.csv")
print("- baseline_predictions.csv")