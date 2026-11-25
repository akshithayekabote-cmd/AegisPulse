
import xarray as xr
import numpy as np
import os

file_path = "data/processed/india_rainfall_2015_2024.nc"

print("AEGISPULSE - MERGED DATA VALIDATION")
print("=" * 55)

# Open merged dataset
ds = xr.open_dataset(file_path, engine="scipy")

rainfall = ds["RAINFALL"]
dates = ds["TIME"].values

# Dataset information
print("\nDATASET INFORMATION")
print("Dimensions:", ds.sizes)
print("Rainfall shape:", rainfall.shape)
print("Start date:", dates[0])
print("End date:", dates[-1])

# Date validation
print("\nDATE VALIDATION")
print("Total days:", len(dates))
print("Unique dates:", len(np.unique(dates)))
print("Duplicate dates:", len(dates) - len(np.unique(dates)))

# Missing-value validation
print("\nMISSING VALUE ANALYSIS")
missing = int(rainfall.isnull().sum().values)
total = rainfall.size
valid = total - missing

print("Total observations:", total)
print("Valid observations:", valid)
print("Missing observations:", missing)
print("Missing percentage:", round(missing / total * 100, 2), "%")

# Rainfall statistics
print("\nRAINFALL STATISTICS")
print("Minimum:", float(rainfall.min().values))
print("Maximum:", float(rainfall.max().values))
print("Average:", float(rainfall.mean().values))

# Yearly observation counts
print("\nYEARLY OBSERVATION COUNTS")

year_counts = ds.TIME.dt.year.to_series().value_counts().sort_index()

print(year_counts.to_string())

print("\nVALIDATION COMPLETED")

ds.close()
