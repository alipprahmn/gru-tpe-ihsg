import pandas as pd
import numpy as np

# =========================
# LOAD NORMALIZED DATA
# =========================

train_data = pd.read_csv("train_normalized.csv")
test_data = pd.read_csv("test_normalized.csv")

window_size = 30


# =========================
# CREATE TRAINING WINDOWS
# =========================

def create_training_sequences(data, window_size):
    X = []
    y = []

    for i in range(window_size, len(data)):
        X.append(
            data[["IHSG_Close", "DJIA_Lag1"]]
            .iloc[i - window_size:i]
            .values
        )

        y.append(data["IHSG_Close"].iloc[i])

    return np.array(X), np.array(y)


X_train, y_train = create_training_sequences(
    train_data,
    window_size
)


# =========================
# CREATE TESTING WINDOWS
# =========================

# Take the last 30 observations from training data
train_history = train_data.tail(window_size)

# Combine training history with testing data
test_with_history = pd.concat(
    [train_history, test_data],
    ignore_index=True
)


def create_testing_sequences(data, window_size):
    X = []
    y = []

    for i in range(window_size, len(data)):
        X.append(
            data[["IHSG_Close", "DJIA_Lag1"]]
            .iloc[i - window_size:i]
            .values
        )

        y.append(data["IHSG_Close"].iloc[i])

    return np.array(X), np.array(y)


X_test, y_test = create_testing_sequences(
    test_with_history,
    window_size
)


# =========================
# DISPLAY RESULTS
# =========================

print("TRAINING DATA")
print(f"Total observations: {len(train_data)}")

print("\nTESTING DATA")
print(f"Total observations: {len(test_data)}")

print("\nWINDOW SIZE")
print(window_size)


print("\nTRAINING SEQUENCES")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

print("\nTESTING SEQUENCES")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# =========================
# DISPLAY FIRST EXAMPLE
# =========================

print("\nFIRST TRAINING WINDOW:")
print(X_train[0])

print("\nFIRST TRAINING TARGET:")
print(y_train[0])

print("\nFIRST TESTING WINDOW:")
print(X_test[0])

print("\nFIRST TESTING TARGET:")
print(y_test[0])


# =========================
# SAVE ARRAYS
# =========================

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("\nSequence datasets successfully saved as:")
print("- X_train.npy")
print("- y_train.npy")
print("- X_test.npy")
print("- y_test.npy")