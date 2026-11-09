import pandas as pd

# Load merged dataset
data = pd.read_csv("merged_data.csv")

# Convert Date column to datetime
data["Date"] = pd.to_datetime(data["Date"])

# Create DJIA Lag-1 feature
data["DJIA_Lag1"] = data["DJIA_Close"].shift(1)

# Display first rows
print("DATA AFTER CREATING DJIA_Lag1:")
print(data.head())

# Display missing values
print("\nMissing values:")
print(data.isnull().sum())

# Display total observations
print("\nTotal observations:", len(data))

# Save dataset
data.to_csv("lag_data.csv", index=False)

print("\nDataset saved as: lag_data.csv")