
import xarray as xr
import pandas as pd

# Load the existing cleaned rainfall dataset
file_path = "data/processed/india_rainfall_2024.nc"
dataset = xr.open_dataset(file_path)

rainfall = dataset["RAINFALL"]

# Convert rainfall data into a DataFrame
df = rainfall.to_dataframe().reset_index()

# Sort data by location and date
df["TIME"] = pd.to_datetime(df["TIME"])
df = df.sort_values(["location", "TIME"])

# Create previous-day rainfall features
df["RAIN_LAG_1"] = (
    df.groupby("location")["RAINFALL"].shift(1)
)

df["RAIN_LAG_3"] = (
    df.groupby("location")["RAINFALL"].shift(3)
)

df["RAIN_LAG_7"] = (
    df.groupby("location")["RAINFALL"].shift(7)
)

# Create rolling rainfall averages using past observations
df["RAIN_ROLLING_3"] = (
    df.groupby("location")["RAINFALL"]
    .transform(lambda x: x.shift(1).rolling(3).mean())
)

df["RAIN_ROLLING_7"] = (
    df.groupby("location")["RAINFALL"]
    .transform(lambda x: x.shift(1).rolling(7).mean())
)

# Extract seasonal information
df["MONTH"] = df["TIME"].dt.month
df["DAY_OF_YEAR"] = df["TIME"].dt.dayofyear

# Predict rainfall for the following day
df["NEXT_DAY_RAINFALL"] = (
    df.groupby("location")["RAINFALL"].shift(-1)
)

# Remove rows without sufficient historical information
df = df.dropna().reset_index(drop=True)

# Save the machine learning dataset
output_path = "data/processed/rainfall_ml_features_2024.csv"
df.to_csv(output_path, index=False)

dataset.close()

print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
print("\nDataset shape:", df.shape)

print("\nAvailable features:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head().to_string(index=False))

print("\nSaved at:", output_path)
