import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load training and testing datasets
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

# Create separate scalers
target_scaler = MinMaxScaler()
feature_scaler = MinMaxScaler()


# =========================
# NORMALIZE TRAINING DATA
# =========================

train_normalized = train_data.copy()

# Fit and transform IHSG_Close
train_normalized["IHSG_Close"] = target_scaler.fit_transform(
    train_data[["IHSG_Close"]]
)

# Fit and transform DJIA_Lag1
train_normalized["DJIA_Lag1"] = feature_scaler.fit_transform(
    train_data[["DJIA_Lag1"]]
)


# =========================
# NORMALIZE TESTING DATA
# =========================

test_normalized = test_data.copy()

# Transform testing data using training parameters
test_normalized["IHSG_Close"] = target_scaler.transform(
    test_data[["IHSG_Close"]]
)

test_normalized["DJIA_Lag1"] = feature_scaler.transform(
    test_data[["DJIA_Lag1"]]
)


# =========================
# DISPLAY RESULTS
# =========================

print("TRAINING DATA BEFORE NORMALIZATION:")
print(train_data.head())

print("\nTRAINING DATA AFTER NORMALIZATION:")
print(train_normalized.head())

print("\nTESTING DATA BEFORE NORMALIZATION:")
print(test_data.head())

print("\nTESTING DATA AFTER NORMALIZATION:")
print(test_normalized.head())


# =========================
# DISPLAY SCALER PARAMETERS
# =========================

print("\nIHSG_Close scaler parameters:")
print(f"Minimum: {target_scaler.data_min_[0]}")
print(f"Maximum: {target_scaler.data_max_[0]}")

print("\nDJIA_Lag1 scaler parameters:")
print(f"Minimum: {feature_scaler.data_min_[0]}")
print(f"Maximum: {feature_scaler.data_max_[0]}")


# =========================
# SAVE NORMALIZED DATA
# =========================

train_normalized.to_csv(
    "train_normalized.csv",
    index=False
)

test_normalized.to_csv(
    "test_normalized.csv",
    index=False
)

print("\nNormalized datasets successfully saved as:")
print("- train_normalized.csv")
print("- test_normalized.csv")