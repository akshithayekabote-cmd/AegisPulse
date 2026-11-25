
import numpy as np
import pandas as pd
import xarray as xr

# Load historical rainfall dataset
ds = xr.open_dataset(
    "data/processed/india_rainfall_cleaned_2015_2024.nc",
    engine="scipy"
)

rainfall_data = ds["RAINFALL"].values

time_values = pd.to_datetime(ds["TIME"].values)

latitude_values = ds["LATITUDE"].values
longitude_values = ds["LONGITUDE"].values

location_lookup = {
    (float(lat), float(lon)): i
    for i, (lat, lon) in enumerate(
        zip(latitude_values, longitude_values)
    )
}

start_2024 = np.where(
    time_values == pd.Timestamp("2024-01-01")
)[0][0]

# Test three possible date alignments
offsets = [-1, 0, 1]

results = {
    offset: {
        "compared": 0,
        "mismatches": 0,
        "invalid": 0
    }
    for offset in offsets
}

# Read test dataset
for chunk in pd.read_csv(
    "data/processed/features/test.csv.gz",
    chunksize=50000
):

    actual = chunk["RAIN_OCCURRED"].to_numpy(dtype=int)

    day_of_year = chunk["DAY_OF_YEAR"].to_numpy(dtype=int)

    location_indices = np.array([
        location_lookup.get(
            (float(lat), float(lon)), -1
        )
        for lat, lon in zip(
            chunk["LATITUDE"],
            chunk["LONGITUDE"]
        )
    ])

    for offset in offsets:

        # Candidate date alignment
        time_indices = (
            start_2024 + day_of_year - 1 + offset
        )

        valid = (
            (location_indices >= 0)
            & (time_indices >= 0)
            & (time_indices < len(time_values))
        )

        results[offset]["invalid"] += int(
            (~valid).sum()
        )

        if not valid.any():
            continue

        recovered_rain = rainfall_data[
            time_indices[valid],
            location_indices[valid]
        ]

        recovered_actual = (
            recovered_rain > 0
        ).astype(int)

        mismatches = int(
            (recovered_actual != actual[valid]).sum()
        )

        results[offset]["compared"] += int(valid.sum())
        results[offset]["mismatches"] += mismatches

ds.close()

# Display comparison
print("\nAEGISPULSE - DATE ALIGNMENT DIAGNOSTICS")
print("=" * 85)

for offset in offsets:

    r = results[offset]

    compared = r["compared"]
    mismatches = r["mismatches"]

    accuracy = (
        (compared - mismatches) / compared * 100
        if compared else 0
    )

    print(f"\nDATE OFFSET: {offset:+d} day(s)")
    print("-" * 50)
    print("Compared records:", compared)
    print("Invalid records:", r["invalid"])
    print("Target mismatches:", mismatches)
    print(f"Target agreement: {accuracy:.4f}%")

print("\nDIAGNOSTIC COMPLETED")
