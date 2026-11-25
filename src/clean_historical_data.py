
import xarray as xr
import numpy as np
import os

input_path = "data/processed/india_rainfall_2015_2024.nc"

output_path = "data/processed/india_rainfall_cleaned_2015_2024.nc"

print("AEGISPULSE - HISTORICAL DATA CLEANING")
print("=" * 60)

# Load merged dataset
ds = xr.open_dataset(input_path, engine="scipy")

rainfall = ds["RAINFALL"]

print("\nOriginal rainfall shape:", rainfall.shape)

# Combine latitude and longitude into one location dimension
stacked = rainfall.stack(
    location=("LATITUDE", "LONGITUDE")
)

# Retain only locations with complete observations
cleaned = stacked.dropna(
    dim="location",
    how="any"
)

# Restore latitude and longitude as location coordinates
cleaned = cleaned.reset_index("location")

# Arrange dimensions
cleaned = cleaned.transpose("TIME", "location")

# Verify missing values
missing = int(cleaned.isnull().sum().values)

print("\nCLEANING RESULTS")
print("=" * 60)

print("Original grid locations:", stacked.sizes["location"])
print("Selected grid locations:", cleaned.sizes["location"])
print("Total days:", cleaned.sizes["TIME"])
print("Cleaned shape:", cleaned.shape)
print("Missing values:", missing)

print("\nRAINFALL STATISTICS")
print("Minimum:", float(cleaned.min().values))
print("Maximum:", float(cleaned.max().values))
print("Average:", float(cleaned.mean().values))

# Verify coordinates
print("\nLOCATION COORDINATES")
print("Latitude:", cleaned["LATITUDE"].min().values,
      "to", cleaned["LATITUDE"].max().values)

print("Longitude:", cleaned["LONGITUDE"].min().values,
      "to", cleaned["LONGITUDE"].max().values)

assert missing == 0, "Missing values found!"
assert cleaned.sizes["location"] == 4885

# Save cleaned dataset
os.makedirs("data/processed", exist_ok=True)

cleaned.to_dataset(name="RAINFALL").to_netcdf(
    output_path,
    engine="scipy",
    format="NETCDF3_64BIT"
)

print("\nCleaned dataset saved successfully!")
print("Location:", output_path)

ds.close()
