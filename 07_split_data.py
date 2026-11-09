import pandas as pd

# Load the selected dataset
df = pd.read_csv("selected_data.csv")

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Sort data chronologically
df = df.sort_values("Date").reset_index(drop=True)

print("DATA BEFORE SPLITTING:")
print(df.head())

print("\nLast 5 observations:")
print(df.tail())

print(f"\nTotal observations: {len(df)}")


# Define the training proportion
train_ratio = 0.8

# Calculate the number of training observations
train_size = int(len(df) * train_ratio)

# Split data chronologically
train_data = df.iloc[:train_size].copy()
test_data = df.iloc[train_size:].copy()


# Display training data information
print("\nTRAINING DATA:")
print(train_data.head())

print("\nTraining data period:")
print(f"Start: {train_data['Date'].min().date()}")
print(f"End: {train_data['Date'].max().date()}")

print(f"Total training observations: {len(train_data)}")


# Display testing data information
print("\nTESTING DATA:")
print(test_data.head())

print("\nTesting data period:")
print(f"Start: {test_data['Date'].min().date()}")
print(f"End: {test_data['Date'].max().date()}")

print(f"Total testing observations: {len(test_data)}")


# Save the datasets
train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

print("\nDatasets successfully saved as:")
print("- train_data.csv")
print("- test_data.csv")