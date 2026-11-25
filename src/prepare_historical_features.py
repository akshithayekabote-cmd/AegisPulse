
import xarray as xr
import numpy as np
import pandas as pd
import os

# File paths
input_path = "data/processed/india_rainfall_cleaned_2015_2024.nc"
output_folder = "data/processed/features"

os.makedirs(output_folder, exist_ok=True)

print("AEGISPULSE - HISTORICAL FEATURE ENGINEERING")
print("=" * 60)

# Load cleaned rainfall dataset
ds = xr.open_dataset(input_path, engine="scipy")

rainfall = ds["RAINFALL"].values
dates = pd.to_datetime(ds["TIME"].values)

latitudes = ds["LATITUDE"].values
longitudes = ds["LONGITUDE"].values

total_days, total_locations = rainfall.shape

print("Total days:", total_days)
print("Total locations:", total_locations)

# Define chronological datasets
splits = {
    "train": (2015, 2021),
    "validation": (2022, 2023),
    "test": (2024, 2024)
}

output_files = {
    name: os.path.join(output_folder, f"{name}.csv.gz")
    for name in splits
}

# Remove previous generated feature files
for path in output_files.values():
    if os.path.exists(path):
        os.remove(path)

# Input days: exclude first 7 days and last day
time_indices = np.arange(7, total_days - 1)

# Target is rainfall on the following day
target_indices = time_indices + 1
target_dates = dates[target_indices]

target_years = target_dates.year

batch_size = 50

print("\nGenerating features...")

for start in range(0, total_locations, batch_size):

    end = min(start + batch_size, total_locations)

    rain = rainfall[:, start:end]

    current = rain[time_indices]
    lag_1 = rain[time_indices - 1]
    lag_3 = rain[time_indices - 3]
    lag_7 = rain[time_indices - 7]

    rolling_3 = (
        current
        + rain[time_indices - 1]
        + rain[time_indices - 2]
    ) / 3

    rolling_7 = np.zeros_like(current)

    for offset in range(7):
        rolling_7 += rain[time_indices - offset]

    rolling_7 /= 7

    next_day = rain[target_indices]

    # Rainfall occurrence target
    target = (next_day > 0).astype(np.int8)

    batch_locations = end - start

    # Create feature table
    df = pd.DataFrame({
        "LATITUDE": np.tile(latitudes[start:end], len(time_indices)),
        "LONGITUDE": np.tile(longitudes[start:end], len(time_indices)),
        "RAINFALL": current.flatten(),
        "RAIN_LAG_1": lag_1.flatten(),
        "RAIN_LAG_3": lag_3.flatten(),
        "RAIN_LAG_7": lag_7.flatten(),
        "RAIN_ROLLING_3": rolling_3.flatten(),
        "RAIN_ROLLING_7": rolling_7.flatten(),
        "MONTH": np.repeat(target_dates.month, batch_locations),
        "DAY_OF_YEAR": np.repeat(target_dates.dayofyear, batch_locations),
        "RAIN_OCCURRED": target.flatten()
    })

    # Save chronological splits
    for name, (first_year, last_year) in splits.items():

        mask = (
            (target_years >= first_year)
            & (target_years <= last_year)
        )

        if not np.any(mask):
            continue

        selected = df.loc[
            np.repeat(mask, batch_locations)
        ]

        selected.to_csv(
            output_files[name],
            mode="a",
            index=False,
            header=(start == 0),
            compression="gzip"
        )

    print(
        f"Processed locations: {end}/{total_locations}"
    )

ds.close()

print("\nFEATURE ENGINEERING COMPLETED")
print("=" * 60)

for name, path in output_files.items():

    print(f"{name.upper()} dataset:")
    print("Location:", path)
    print("Size (MB):", round(os.path.getsize(path) / (1024 ** 2), 2))

print("\nAll feature datasets saved successfully!")
