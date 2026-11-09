import pandas as pd

# Load dataset
df = pd.read_csv("lag_data.csv")

# Display dataset before handling missing values
print("DATA BEFORE HANDLING MISSING VALUES:")
print(df.head())

print("\nMissing values before:")
print(df.isnull().sum())

print("\nTotal observations before:", len(df))

# Remove rows containing missing values
df = df.dropna().reset_index(drop=True)

# Display dataset after handling missing values
print("\nDATA AFTER HANDLING MISSING VALUES:")
print(df.head())

print("\nMissing values after:")
print(df.isnull().sum())

print("\nTotal observations after:", len(df))

# Select variables used in the model
final_data = df[["Date", "IHSG_Close", "DJIA_Lag1"]]

# Save final dataset
final_data.to_csv("final_data.csv", index=False)

print("\nFinal dataset saved as: final_data.csv")