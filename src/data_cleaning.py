import pandas as pd

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Quick data inspection
print(df.head())
print(df.shape)
print(df.describe())
print(df.info())
print(df.isnull().sum())

# Clean dataset
df = df.drop_duplicates()  # Remove exact duplicate rows
df = df.drop_duplicates(subset=['track_name', 'artists'], keep='first')  # Remove duplicate songs by the same artist
df = df.dropna()  # Remove rows with missing values

# Save cleaned dataset
df.to_csv("data/cleaned_dataset.csv", index=False)

# Verify no duplicates remain
duplicates = df[df.duplicated(subset=['track_name', 'artists'])]
print("Number of duplicates remaining:", len(duplicates))
