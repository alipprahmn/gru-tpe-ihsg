import pandas as pd

# Load IHSG data
ihsg = pd.read_csv(
    "IHSG.csv",
    skiprows=3,
    names=[
        "Date",
        "IHSG_Adj_Close",
        "IHSG_Close_Original",
        "IHSG_High",
        "IHSG_Low",
        "IHSG_Open",
        "IHSG_Volume"
    ]
)

# Load DJIA data
djia = pd.read_csv(
    "DJIA.csv",
    skiprows=3,
    names=[
        "Date",
        "DJIA_Adj_Close",
        "DJIA_Close",
        "DJIA_High",
        "DJIA_Low",
        "DJIA_Open",
        "DJIA_Volume"
    ]
)

# Convert Date columns
ihsg["Date"] = pd.to_datetime(ihsg["Date"])
djia["Date"] = pd.to_datetime(djia["Date"])

# Use Adjusted Close for IHSG
ihsg["IHSG_Close"] = pd.to_numeric(ihsg["IHSG_Adj_Close"])

# Convert DJIA Close to numeric
djia["DJIA_Close"] = pd.to_numeric(djia["DJIA_Close"])

# Select required columns
ihsg = ihsg[["Date", "IHSG_Close"]]
djia = djia[["Date", "DJIA_Close"]]

# Merge datasets based on Date
merged_data = pd.merge(
    ihsg,
    djia,
    on="Date",
    how="inner"
)

# Sort data chronologically
merged_data = merged_data.sort_values("Date").reset_index(drop=True)

# Display results
print("MERGED DATA:")
print(merged_data.head())

print("\nLast 5 observations:")
print(merged_data.tail())

print("\nTotal observations after merging:", len(merged_data))

# Save merged dataset
merged_data.to_csv("merged_data.csv", index=False)

print("\nMerged dataset saved as: merged_data.csv")