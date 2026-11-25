
import xarray as xr
import numpy as np
import pandas as pd
import os

INPUT_PATH = (
    "data/processed/"
    "india_rainfall_cleaned_2015_2024.nc"
)

MAPPING_PATH = (
    "data/processed/"
    "spatial_neighbor_mapping.npz"
)

OUTPUT_FOLDER = (
    "data/processed/"
    "spatial_features"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("AEGISPULSE - SPATIAL RAINFALL FEATURE ENGINEERING")
print("=" * 65)

ds = xr.open_dataset(
    INPUT_PATH,
    engine="scipy"
)

rainfall = ds["RAINFALL"].values
dates = pd.to_datetime(ds["TIME"].values)

total_days, total_locations = rainfall.shape

print("Total days:", total_days)
print("Total locations:", total_locations)

mapping = np.load(MAPPING_PATH)

neighbor_indices = mapping["neighbor_indices"]

if neighbor_indices.shape != (total_locations, 8):
    raise ValueError(
        "Neighbour mapping does not match rainfall dataset."
    )

if not np.allclose(
    mapping["latitudes"],
    ds["LATITUDE"].values
):
    raise ValueError("Latitude ordering mismatch.")

if not np.allclose(
    mapping["longitudes"],
    ds["LONGITUDE"].values
):
    raise ValueError("Longitude ordering mismatch.")

splits = {
    "train": (2015, 2021),
    "validation": (2022, 2023),
    "test": (2024, 2024)
}

output_files = {
    name: os.path.join(
        OUTPUT_FOLDER,
        f"{name}_spatial.csv.gz"
    )
    for name in splits
}

for path in output_files.values():
    if os.path.exists(path):
        os.remove(path)

time_indices = np.arange(7, total_days - 1)

target_indices = time_indices + 1

target_dates = dates[target_indices]
target_years = target_dates.year

batch_size = 50

print("\nGenerating spatial features...")

for start in range(0, total_locations, batch_size):

    end = min(
        start + batch_size,
        total_locations
    )

    current_indices = np.arange(start, end)

    current_rain = rainfall[
        time_indices[:, None],
        current_indices[None, :]
    ]

    batch_neighbors = neighbor_indices[start:end]

    neighbor_mean = np.zeros_like(
        current_rain,
        dtype=np.float32
    )

    neighbor_max = np.zeros_like(
        current_rain,
        dtype=np.float32
    )

    neighbor_rainy_count = np.zeros_like(
        current_rain,
        dtype=np.int8
    )

    neighbor_valid_count = np.zeros(
        end - start,
        dtype=np.int8
    )

    for location_index in range(end - start):

        valid_neighbors = batch_neighbors[
            location_index
        ]

        valid_neighbors = valid_neighbors[
            valid_neighbors >= 0
        ]

        if len(valid_neighbors) == 0:
            raise ValueError(
                f"No neighbours found for location {start + location_index}"
            )

        neighbor_valid_count[location_index] = len(
            valid_neighbors
        )

        neighbor_rain = rainfall[
            time_indices[:, None],
            valid_neighbors[None, :]
        ]

        neighbor_mean[:, location_index] = (
            neighbor_rain.mean(axis=1)
        )

        neighbor_max[:, location_index] = (
            neighbor_rain.max(axis=1)
        )

        neighbor_rainy_count[:, location_index] = (
            np.sum(neighbor_rain > 0, axis=1)
        )

    batch_locations = end - start

    df = pd.DataFrame({
        "LATITUDE": np.tile(
            ds["LATITUDE"].values[start:end],
            len(time_indices)
        ),
        "LONGITUDE": np.tile(
            ds["LONGITUDE"].values[start:end],
            len(time_indices)
        ),

        # Label with the target date for correct alignment
        "DATE": np.repeat(
            target_dates.strftime("%Y-%m-%d"),
            batch_locations
        ),

        "NEIGHBOR_RAIN_MEAN": neighbor_mean.flatten(),
        "NEIGHBOR_RAIN_MAX": neighbor_max.flatten(),
        "NEIGHBOR_RAINY_COUNT": neighbor_rainy_count.flatten(),
        "NEIGHBOR_VALID_COUNT": np.tile(
            neighbor_valid_count,
            len(time_indices)
        )
    })

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

print("\nSPATIAL FEATURE GENERATION COMPLETED")
print("=" * 65)

for name, path in output_files.items():

    print(f"\n{name.upper()} dataset:")
    print("Location:", path)
    print(
        "Size (MB):",
        round(os.path.getsize(path) / (1024 ** 2), 2)
    )

print("\nAll spatial feature datasets saved successfully!")
