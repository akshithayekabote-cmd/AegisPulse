
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
# 2. LOAD IMD SUBDIVISION MAPPING
# --------------------------------------------------

mapping = pd.read_csv(
    "data/processed/india_grid_subdivision_mapping.csv"
)

mapping = mapping[
    mapping["STATUS"] == "MAPPED"
]

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
print("Model threshold:", threshold)
print("Model bias:", bias)

# --------------------------------------------------
# 4. INITIALIZE MONTHLY RESULTS
# --------------------------------------------------

results = {}

for region in target_regions:

    for month in range(1, 13):

        results[(region, month)] = {
            "TP": 0,
            "FN": 0
        }

total_records = 0
unmatched_locations = 0
invalid_dates = 0
target_mismatches = 0

# --------------------------------------------------
# 5. PROCESS 2024 TEST DATA
# --------------------------------------------------

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

    months = chunk["MONTH"].to_numpy(
        dtype=int
    )

    # Model V2 predictions
    X = chunk[features].to_numpy(dtype=float)
    X = (X - mean) / std

    # Correct probability calculation includes saved bias
    probabilities = 1 / (
        1 + np.exp(
            -np.clip(X @ weights + bias, -500, 500)
        )
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

    # Match geographical coordinates
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

    # Verify binary target alignment
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

    total_records += len(chunk)

    # --------------------------------------------------
    # 6. ANALYZE VERY-LIGHT RAINFALL
    # --------------------------------------------------

    very_light_mask = (
        valid
        & (target_rainfall > 0)
        & (target_rainfall <= 2.4)
    )

    for region in target_regions:

        region_mask = (
            very_light_mask
            & (assigned == region)
        )

        for month in range(1, 13):

            month_mask = (
                region_mask
                & (months == month)
            )

            if not month_mask.any():
                continue

            tp = (
                month_mask
                & (predictions == 1)
            )

            fn = (
                month_mask
                & (predictions == 0)
            )

            results[(region, month)]["TP"] += int(
                tp.sum()
            )

            results[(region, month)]["FN"] += int(
                fn.sum()
            )

# --------------------------------------------------
# 7. DATA VALIDATION
# --------------------------------------------------

print("\nDATA VALIDATION")
print("=" * 80)

print("Total test records:", total_records)
print("Unmatched locations:", unmatched_locations)
print("Invalid dates:", invalid_dates)
print("Target rainfall mismatches:", target_mismatches)

if unmatched_locations > 0 or invalid_dates > 0:
    raise ValueError(
        "Some test records could not be matched."
    )

if target_mismatches > 0:
    raise ValueError(
        "Recovered rainfall does not match the binary target."
    )

print("Rainfall alignment verified.")

# --------------------------------------------------
# 8. BUILD MONTHLY PERFORMANCE REPORT
# --------------------------------------------------

rows = []

for region in target_regions:

    for month in range(1, 13):

        r = results[(region, month)]

        tp = r["TP"]
        fn = r["FN"]

        total_events = tp + fn

        detection_rate = (
            tp / total_events * 100
            if total_events > 0
            else np.nan
        )

        rows.append({
            "SUBDIVISION": region,
            "MONTH": month,
            "TRUE_POSITIVES": tp,
            "FALSE_NEGATIVES": fn,
            "TOTAL_VERY_LIGHT_EVENTS": total_events,
            "DETECTION_RATE_PERCENT": detection_rate
        })

summary = pd.DataFrame(rows)

summary["MONTH_NAME"] = pd.to_datetime(
    summary["MONTH"],
    format="%m"
).dt.month_name()

# --------------------------------------------------
# 9. SAVE RESULTS
# --------------------------------------------------

output_path = (
    "data/processed/weak_region_monthly_intensity_2024.csv"
)

summary.to_csv(output_path, index=False)

# --------------------------------------------------
# 10. DISPLAY RESULTS
# --------------------------------------------------

print("\nAEGISPULSE - MONTHLY VERY-LIGHT RAINFALL DETECTION")
print("=" * 110)

print("Intensity range: > 0 to 2.4 mm/day")
print("Model threshold:", threshold)
print("Model bias:", bias)

for region in target_regions:

    regional = summary[
        summary["SUBDIVISION"] == region
    ]

    print(f"\n{region}")
    print("-" * 110)

    print(
        regional[
            [
                "MONTH_NAME",
                "TRUE_POSITIVES",
                "FALSE_NEGATIVES",
                "TOTAL_VERY_LIGHT_EVENTS",
                "DETECTION_RATE_PERCENT"
            ]
        ].to_string(
            index=False,
            formatters={
                "DETECTION_RATE_PERCENT": lambda x:
                    f"{x:.2f}%" if pd.notna(x) else "N/A"
            }
        )
    )

print("\nSaved results to:")
print(output_path)

print("\nMONTHLY VERY-LIGHT RAINFALL ANALYSIS COMPLETED")

ds.close()
