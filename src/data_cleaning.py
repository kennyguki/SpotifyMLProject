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
df = df.drop_duplicates()
df = df.dropna()

# Save cleaned dataset
df.to_csv("data/cleaned_dataset.csv", index=False)

df[df.duplicated()]