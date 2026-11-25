
import xarray as xr
import numpy as np

file_path = "data/processed/india_rainfall_2015_2024.nc"

print("AEGISPULSE - SPATIAL COVERAGE ANALYSIS")
print("=" * 60)

ds = xr.open_dataset(file_path, engine="scipy")

rainfall = ds["RAINFALL"].values

# Check valid observations
valid_mask = np.isfinite(rainfall)

# Overall coverage
total = rainfall.size
valid = np.sum(valid_mask)

print("\nOVERALL COVERAGE")
print("Total observations:", total)
print("Valid observations:", valid)
print("Missing observations:", total - valid)
print("Coverage:", round(valid / total * 100, 2), "%")

# Yearly coverage
print("\nYEARLY COVERAGE")
print("=" * 60)

years = ds["TIME"].dt.year.values

for year in np.unique(years):

    year_mask = years == year
    yearly_data = rainfall[year_mask]

    yearly_valid = np.isfinite(yearly_data).sum()
    yearly_total = yearly_data.size

    percentage = yearly_valid / yearly_total * 100

    print(
        year,
        "| Valid:", yearly_valid,
        "| Coverage:", round(percentage, 2), "%"
    )

# Valid observation count per grid location
valid_days = np.sum(valid_mask, axis=0)

total_days = len(years)

print("\nSPATIAL COVERAGE PER LOCATION")
print("=" * 60)

print("Total grid locations:", valid_days.size)
print("Maximum valid days:", valid_days.max())
print("Minimum valid days:", valid_days.min())

print("\nLOCATIONS WITH SUFFICIENT COVERAGE")

for threshold in [100, 99, 95, 90, 80]:

    required_days = int(np.ceil(total_days * threshold / 100))

    count = np.sum(valid_days >= required_days)

    print(
        f"{threshold}% coverage:",
        count,
        "locations"
    )

# Completely observed locations
complete_locations = np.sum(valid_days == total_days)

print("\nCompletely observed locations:", complete_locations)

print("\nANALYSIS COMPLETED")

ds.close()
