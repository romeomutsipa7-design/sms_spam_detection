
# DSI235 Machine Learning Capstone Project
# Project: SMS Spam Detection
# Step 1: Download and load the dataset

import pandas as pd
import urllib.request
import zipfile
import io

# Dataset download address
url = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/00228/"
    "smsspamcollection.zip"
)

print("Downloading the SMS dataset...")

# Download the dataset
response = urllib.request.urlopen(url, timeout=60)
zip_data = zipfile.ZipFile(io.BytesIO(response.read()))

# Read the messages
with zip_data.open("SMSSpamCollection") as file:
    df = pd.read_csv(
        file,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )

# Display the first five records
print("\nFirst five messages:")
print(df.head())

# Display dataset dimensions
print("\nNumber of rows and columns:")
print(df.shape)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Count message categories
print("\nSpam and legitimate messages:")
print(df["label"].value_counts())

print("\nStep 1 completed successfully!")
