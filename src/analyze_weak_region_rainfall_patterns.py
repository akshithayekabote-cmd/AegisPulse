
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
# 4. PROCESS TEST DATA
# --------------------------------------------------

all_records = []

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

    actual = chunk[
        "RAIN_OCCURRED"
    ].to_numpy(dtype=int)

    # Model V2 prediction with saved bias
    X = chunk[features].to_numpy(dtype=float)
    X = (X - mean) / std

    probabilities = 1 / (
        1 + np.exp(-np.clip(X @ weights + bias, -500, 500))
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # Recover actual rainfall using verified offset 0
    day_of_year = chunk[
        "DAY_OF_YEAR"
    ].to_numpy(dtype=int)

    target_time_indices = (
        start_2024 + day_of_year - 1
    )

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

    target_rainfall = np.full(
        len(chunk), np.nan
    )

    target_rainfall[valid] = rainfall_data[
        target_time_indices[valid],
        location_indices[valid]
    ]

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

    # Select very-light rainfall events
    very_light = (
        valid
        & (target_rainfall > 0)
        & (target_rainfall <= 2.4)
    )

    # Keep only Punjab and Jammu & Kashmir
    selected = (
        very_light
        & np.isin(assigned, target_regions)
    )

    if selected.any():

        result = chunk.loc[
            selected,
            features
        ].copy()

        result["SUBDIVISION"] = assigned[selected]
        result["OUTCOME"] = np.where(
            predictions[selected] == 1,
            "TP",
            "FN"
        )

        result["ACTUAL_RAINFALL"] = (
            target_rainfall[selected]
        )

        all_records.append(result)

    total_records += len(chunk)

# --------------------------------------------------
# 5. DATA VALIDATION
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
        "Data validation failed."
    )

print("Rainfall alignment verified.")

# --------------------------------------------------
# 6. COMBINE SELECTED EVENTS
# --------------------------------------------------

if not all_records:
    raise ValueError("No matching rainfall events found.")

data = pd.concat(
    all_records,
    ignore_index=True
)

print("\nSELECTED EVENT COUNTS")
print("=" * 80)

print(
    data.groupby(
        ["SUBDIVISION", "OUTCOME"]
    ).size().to_string()
)

# --------------------------------------------------
# 7. RAINFALL HISTORY FEATURE ANALYSIS
# --------------------------------------------------

history_features = [
    "RAIN_TODAY",
    "RAIN_LAG_1",
    "RAIN_LAG_3",
    "RAIN_LAG_7",
    "RAIN_ROLLING_3",
    "RAIN_ROLLING_7"
]

summary_rows = []

print("\nRAINFALL HISTORY COMPARISON")
print("=" * 100)

for region in target_regions:

    print("\n" + "=" * 100)
    print(region)
    print("=" * 100)

    for outcome in ["TP", "FN"]:

        subset = data[
            (data["SUBDIVISION"] == region)
            & (data["OUTCOME"] == outcome)
        ]

        print(f"\n{outcome} - Records: {len(subset)}")
        print("-" * 80)

        for feature in history_features:

            values = subset[feature].to_numpy(
                dtype=float
            )

            if len(values) == 0:
                continue

            mean_value = values.mean()
            median_value = np.median(values)
            zero_percentage = (
                (values == 0).mean() * 100
            )

            print(
                f"{feature:18s} "
                f"Mean: {mean_value:10.3f} "
                f"Median: {median_value:10.3f} "
                f"Zero: {zero_percentage:7.2f}%"
            )

            summary_rows.append({
                "SUBDIVISION": region,
                "OUTCOME": outcome,
                "RECORDS": len(values),
                "FEATURE": feature,
                "MEAN": mean_value,
                "MEDIAN": median_value,
                "ZERO_PERCENTAGE": zero_percentage
            })

# --------------------------------------------------
# 8. MONTHLY TP AND FN COUNTS
# --------------------------------------------------

print("\nMONTHLY EVENT COMPARISON")
print("=" * 80)

monthly = (
    data.groupby(
        ["SUBDIVISION", "MONTH", "OUTCOME"]
    )
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

if "TP" not in monthly.columns:
    monthly["TP"] = 0

if "FN" not in monthly.columns:
    monthly["FN"] = 0

monthly["TOTAL"] = (
    monthly["TP"] + monthly["FN"]
)

monthly["DETECTION_RATE"] = np.where(
    monthly["TOTAL"] > 0,
    monthly["TP"] / monthly["TOTAL"] * 100,
    np.nan
)

for region in target_regions:

    print("\n" + region)
    print("-" * 70)

    region_monthly = monthly[
        monthly["SUBDIVISION"] == region
    ].sort_values("MONTH")

    print(
        region_monthly[
            [
                "MONTH",
                "TP",
                "FN",
                "TOTAL",
                "DETECTION_RATE"
            ]
        ].to_string(
            index=False,
            formatters={
                "DETECTION_RATE": lambda x: (
                    f"{x:.2f}%" if pd.notna(x) else "N/A"
                )
            }
        )
    )

# --------------------------------------------------
# 9. SAVE REPORTS
# --------------------------------------------------

summary_df = pd.DataFrame(summary_rows)

summary_path = (
    "data/processed/weak_region_rainfall_history_2024.csv"
)

monthly_path = (
    "data/processed/weak_region_rainfall_monthly_2024.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

monthly.to_csv(
    monthly_path,
    index=False
)

print("\nSaved rainfall history report:")
print(summary_path)

print("\nSaved monthly comparison report:")
print(monthly_path)

print("\nWEAK REGION RAINFALL PATTERN ANALYSIS COMPLETED")

ds.close()
