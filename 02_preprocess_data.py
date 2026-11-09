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

# Use IHSG Adjusted Close
ihsg["IHSG_Close"] = pd.to_numeric(ihsg["IHSG_Adj_Close"])

# Use DJIA Close
djia["DJIA_Close"] = pd.to_numeric(djia["DJIA_Close"])

# Select required columns
ihsg = ihsg[["Date", "IHSG_Close"]]
djia = djia[["Date", "DJIA_Close"]]

# Display results
print("IHSG DATA:")
print(ihsg.head())

print("\nDJIA DATA:")
print(djia.head())

print("\nTotal IHSG observations:", len(ihsg))
print("Total DJIA observations:", len(djia))