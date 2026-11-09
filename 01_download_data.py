import yfinance as yf
import pandas as pd

# ==========================================
# DOWNLOAD IHSG AND DJIA DATA FROM YAHOO FINANCE
# ==========================================

# Define the research period
start_date = "2015-01-01"
end_date = "2025-12-31"

# Download IHSG data
ihsg = yf.download(
    "^JKSE",
    start=start_date,
    end=end_date,
    auto_adjust=False
)

# Download DJIA data
djia = yf.download(
    "^DJI",
    start=start_date,
    end=end_date,
    auto_adjust=False
)

# Display IHSG information
print("\n===== IHSG DATA =====")
print("\nFirst 5 observations:")
print(ihsg.head())

print("\nLast 5 observations:")
print(ihsg.tail())

print("\nTotal IHSG observations:", len(ihsg))

# Display DJIA information
print("\n===== DJIA DATA =====")
print("\nFirst 5 observations:")
print(djia.head())

print("\nLast 5 observations:")
print(djia.tail())

print("\nTotal DJIA observations:", len(djia))

# Save datasets as CSV files
ihsg.to_csv("IHSG.csv")
djia.to_csv("DJIA.csv")

print("\nDatasets successfully saved:")
print("- IHSG.csv")
print("- DJIA.csv")