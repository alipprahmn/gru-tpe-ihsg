import pandas as pd

# Load the final dataset
df = pd.read_csv("final_data.csv")

print("DATA BEFORE FEATURE SELECTION:")
print(df.head())

print("\nColumns before selection:")
print(df.columns.tolist())

print(f"\nTotal observations: {len(df)}")


# Select variables used in the model
selected_df = df[[
    "Date",
    "IHSG_Close",
    "DJIA_Lag1"
]]

print("\nDATA AFTER FEATURE SELECTION:")
print(selected_df.head())

print("\nColumns after selection:")
print(selected_df.columns.tolist())

print(f"\nTotal observations: {len(selected_df)}")


# Save the selected dataset
selected_df.to_csv("selected_data.csv", index=False)

print("\nSelected dataset saved as: selected_data.csv")