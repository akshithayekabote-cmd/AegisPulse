
import numpy as np
import pandas as pd
import xarray as xr

# --------------------------------------------------
# 1. LOAD HISTORICAL RAINFALL DATA
# --------------------------------------------------

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

print("Historical rainfall data loaded.")

# --------------------------------------------------
# 2. LOAD SUBDIVISION MAPPING
# --------------------------------------------------

mapping = pd.read_csv(
    "data/processed/india_grid_subdivision_mapping.csv"
)

mapping = mapping[mapping["STATUS"] == "MAPPED"]

subdivision_lookup = {
    (float(row.LATITUDE), float(row.LONGITUDE)): row.SUBDIVISION
    for row in mapping.itertuples()
}

target_regions = [
    "PUNJAB",
    "JAMMU & KASHMIR"
]

# --------------------------------------------------
# 3. LOAD FROZEN MODEL V2
# --------------------------------------------------

model = np.load(
    "models/historical_rainfall_model_v2.npz"
)

weights = model["weights"]
bias = float(model["bias"])
mean = model["mean"]
std = model["std"]
threshold = float(model["threshold"])

features = [
    "LATITUDE",
    "LONGITUDE",
    "RAINFALL",
    "RAIN_LAG_1",
    "RAIN_LAG_3",
    "RAIN_LAG_7",
    "RAIN_ROLLING_3",
    "RAIN_ROLLING_7",
    "MONTH",
    "DAY_OF_YEAR",
    "RAIN_TODAY"
]

print("Frozen Model V2 loaded.")
print("Threshold:", threshold)

# --------------------------------------------------
# 4. DEFINE PROBABILITY BANDS
# --------------------------------------------------

probability_bands = [
    (0.00, 0.10, "0.00-0.10"),
    (0.10, 0.20, "0.10-0.20"),
    (0.20, 0.30, "0.20-0.30"),
    (0.30, 0.40, "0.30-0.40"),
    (0.40, 0.45, "0.40-0.45"),
    (0.45, 0.50, "0.45-0.50"),
    (0.50, 0.60, "0.50-0.60"),
    (0.60, 0.70, "0.60-0.70"),
    (0.70, 0.80, "0.70-0.80"),
    (0.80, 0.90, "0.80-0.90"),
    (0.90, 1.01, "0.90-1.00")
]

probability_records = {
    region: {
        "TP": [],
        "FN": []
    }
    for region in target_regions
}

# --------------------------------------------------
# 5. PROCESS 2024 TEST DATA
# --------------------------------------------------

total_records = 0
unmatched_locations = 0
invalid_dates = 0
target_mismatches = 0

for chunk in pd.read_csv(
    "data/processed/features/test.csv.gz",
    chunksize=50000
):

    chunk["RAIN_TODAY"] = (
        chunk["RAINFALL"] > 0
    ).astype(int)

    actual = chunk["RAIN_OCCURRED"].to_numpy(
        dtype=int
    )

    # Calculate unchanged Model V2 probabilities
    X = chunk[features].to_numpy(dtype=float)
    X = (X - mean) / std

    probabilities = 1 / (
        1 + np.exp(-np.clip(X @ weights + bias, -500, 500))
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # Confirmed date alignment: offset 0
    day_of_year = chunk[
        "DAY_OF_YEAR"
    ].to_numpy(dtype=int)

    target_time_indices = (
        start_2024 + day_of_year - 1
    )

    # Match coordinates
    location_indices = np.array([
        location_lookup.get(
            (float(lat), float(lon)), -1
        )
        for lat, lon in zip(
            chunk["LATITUDE"],
            chunk["LONGITUDE"]
        )
    ])

    valid_dates = (
        (target_time_indices >= 0)
        & (target_time_indices < len(time_values))
    )

    valid_locations = location_indices >= 0

    valid = valid_dates & valid_locations

    unmatched_locations += int(
        (~valid_locations).sum()
    )

    invalid_dates += int(
        (~valid_dates).sum()
    )

    # Recover actual rainfall
    target_rainfall = np.full(
        len(chunk), np.nan, dtype=float
    )

    target_rainfall[valid] = rainfall_data[
        target_time_indices[valid],
        location_indices[valid]
    ]

    # Verify recovered rainfall against binary target
    recovered_actual = (
        target_rainfall[valid] > 0
    ).astype(int)

    target_mismatches += int(
        (recovered_actual != actual[valid]).sum()
    )

    # Identify subdivisions
    assigned = np.array([
        subdivision_lookup.get(
            (float(lat), float(lon)), ""
        )
        for lat, lon in zip(
            chunk["LATITUDE"],
            chunk["LONGITUDE"]
        )
    ])

    # Select actual very-light rainfall events
    very_light = (
        valid
        & (target_rainfall > 0)
        & (target_rainfall <= 2.4)
    )

    total_records += len(chunk)

    for region in target_regions:

        region_mask = (
            very_light
            & (assigned == region)
        )

        tp_mask = (
            region_mask
            & (predictions == 1)
        )

        fn_mask = (
            region_mask
            & (predictions == 0)
        )

        probability_records[region]["TP"].extend(
            probabilities[tp_mask].tolist()
        )

        probability_records[region]["FN"].extend(
            probabilities[fn_mask].tolist()
        )

# --------------------------------------------------
# 6. VALIDATE DATA
# --------------------------------------------------

print("\nDATA VALIDATION")
print("=" * 80)

print("Total test records:", total_records)
print("Unmatched locations:", unmatched_locations)
print("Invalid dates:", invalid_dates)
print("Target rainfall mismatches:", target_mismatches)

if (
    unmatched_locations > 0
    or invalid_dates > 0
    or target_mismatches > 0
):
    raise ValueError(
        "Data validation failed. Do not interpret the results."
    )

print("Rainfall alignment verified.")

# --------------------------------------------------
# 7. PROBABILITY SUMMARY
# --------------------------------------------------

summary_rows = []
band_rows = []

for region in target_regions:

    print("\n" + "=" * 100)
    print(region)
    print("=" * 100)

    for outcome in ["TP", "FN"]:

        values = np.array(
            probability_records[region][outcome],
            dtype=float
        )

        if len(values) == 0:
            print(f"\n{outcome}: No records")
            continue

        print(f"\n{outcome} - PROBABILITY SUMMARY")
        print("-" * 80)

        print("Records:", len(values))
        print("Mean probability:", round(values.mean(), 4))
        print("Median probability:", round(np.median(values), 4))
        print("Minimum probability:", round(values.min(), 4))
        print("Maximum probability:", round(values.max(), 4))
        print("25th percentile:", round(np.percentile(values, 25), 4))
        print("75th percentile:", round(np.percentile(values, 75), 4))

        summary_rows.append({
            "SUBDIVISION": region,
            "OUTCOME": outcome,
            "RECORDS": len(values),
            "MEAN_PROBABILITY": values.mean(),
            "MEDIAN_PROBABILITY": np.median(values),
            "MIN_PROBABILITY": values.min(),
            "MAX_PROBABILITY": values.max(),
            "P25": np.percentile(values, 25),
            "P75": np.percentile(values, 75)
        })

        print("\nPROBABILITY BAND DISTRIBUTION")
        print("-" * 80)

        for lower, upper, label in probability_bands:

            count = int(
                (
                    (values >= lower)
                    & (values < upper)
                ).sum()
            )

            percentage = count / len(values) * 100

            print(
                f"{label:12s} "
                f"Records: {count:6d} "
                f"Percentage: {percentage:6.2f}%"
            )

            band_rows.append({
                "SUBDIVISION": region,
                "OUTCOME": outcome,
                "PROBABILITY_BAND": label,
                "RECORDS": count,
                "PERCENTAGE": percentage
            })

# --------------------------------------------------
# 8. SAVE REPORTS
# --------------------------------------------------

summary_df = pd.DataFrame(summary_rows)
bands_df = pd.DataFrame(band_rows)

summary_path = (
    "data/processed/weak_region_probability_summary_2024.csv"
)

bands_path = (
    "data/processed/weak_region_probability_bands_2024.csv"
)

summary_df.to_csv(summary_path, index=False)
bands_df.to_csv(bands_path, index=False)

print("\nSaved probability summary:")
print(summary_path)

print("\nSaved probability band report:")
print(bands_path)

print("\nWEAK REGION PROBABILITY ANALYSIS COMPLETED")

ds.close()
