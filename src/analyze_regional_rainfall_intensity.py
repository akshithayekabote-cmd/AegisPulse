
import numpy as np
import pandas as pd
import xarray as xr

# --------------------------------------------------
# 1. LOAD CLEANED HISTORICAL RAINFALL DATA
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

print("Historical rainfall dataset loaded.")
print("Rainfall shape:", rainfall_data.shape)
print("2024 starting index:", start_2024)

# --------------------------------------------------
# 2. LOAD OFFICIAL IMD SUBDIVISION MAPPING
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

target_subdivisions = [
    "JAMMU & KASHMIR",
    "PUNJAB",
    "GANGETIC WEST BENGAL",
    "CHHATTISGARH",
    "ORISSA",
    "HIMACHAL PRADESH",
    "JHARKHAND",
    "MARATHWADA"
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

# --------------------------------------------------
# 4. RAINFALL INTENSITY CLASSIFICATION
# --------------------------------------------------

def classify_intensity(rainfall):

    if rainfall <= 0:
        return "NO_RAIN"

    elif rainfall <= 2.4:
        return "VERY_LIGHT"

    elif rainfall <= 15.5:
        return "LIGHT"

    elif rainfall <= 64.4:
        return "MODERATE"

    elif rainfall <= 115.5:
        return "HEAVY"

    elif rainfall <= 204.4:
        return "VERY_HEAVY"

    else:
        return "EXTREMELY_HEAVY"


categories = [
    "NO_RAIN",
    "VERY_LIGHT",
    "LIGHT",
    "MODERATE",
    "HEAVY",
    "VERY_HEAVY",
    "EXTREMELY_HEAVY"
]

# --------------------------------------------------
# 5. INITIALIZE RESULTS
# --------------------------------------------------

results = {}

for subdivision in target_subdivisions:

    for error_type in ["TP", "FP", "FN"]:

        for category in categories:

            results[
                (subdivision, error_type, category)
            ] = {
                "records": 0,
                "rainfall_sum": 0.0
            }

total_records = 0
unmatched_locations = 0
invalid_dates = 0
target_mismatches = 0
compared_records = 0

# --------------------------------------------------
# 6. PROCESS 2024 TEST DATA
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

    # Prepare Model V2 features
    X = chunk[features].to_numpy(dtype=float)

    # Standardize using saved training statistics
    X = (X - mean) / std

    # Calculate probabilities using weights AND bias
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

    # Match geographical locations
    location_indices = np.array([
        location_lookup.get(
            (float(lat), float(lon)), -1
        )
        for lat, lon in zip(
            chunk["LATITUDE"],
            chunk["LONGITUDE"]
        )
    ])

    valid = (
        (location_indices >= 0)
        & (target_time_indices >= 0)
        & (target_time_indices < len(time_values))
    )

    unmatched_locations += int(
        (location_indices < 0).sum()
    )

    invalid_dates += int(
        (
            (target_time_indices < 0)
            | (target_time_indices >= len(time_values))
        ).sum()
    )

    # Recover actual rainfall for the target date
    target_rainfall = np.full(
        len(chunk),
        np.nan,
        dtype=float
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

    compared_records += int(valid.sum())

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

    # Analyze selected subdivisions
    for subdivision in target_subdivisions:

        region_mask = (
            (assigned == subdivision) & valid
        )

        if not region_mask.any():
            continue

        tp_mask = (
            region_mask
            & (actual == 1)
            & (predictions == 1)
        )

        fp_mask = (
            region_mask
            & (actual == 0)
            & (predictions == 1)
        )

        fn_mask = (
            region_mask
            & (actual == 1)
            & (predictions == 0)
        )

        for error_type, mask in [
            ("TP", tp_mask),
            ("FP", fp_mask),
            ("FN", fn_mask)
        ]:

            if not mask.any():
                continue

            values = target_rainfall[mask]

            for value in values:

                category = classify_intensity(value)

                r = results[
                    (subdivision, error_type, category)
                ]

                r["records"] += 1
                r["rainfall_sum"] += float(value)

# --------------------------------------------------
# 7. DATA VALIDATION
# --------------------------------------------------

print("\nDATA VALIDATION")
print("=" * 80)

print("Total test records:", total_records)
print("Compared records:", compared_records)
print("Unmatched locations:", unmatched_locations)
print("Invalid target dates:", invalid_dates)
print("Target rainfall mismatches:", target_mismatches)

if unmatched_locations > 0 or invalid_dates > 0:
    raise ValueError(
        "Some test records could not be matched to historical rainfall."
    )

if target_mismatches > 0:
    raise ValueError(
        "Recovered rainfall does not match RAIN_OCCURRED."
    )

print("Rainfall date alignment verified: 100%.")

# --------------------------------------------------
# 8. BUILD INTENSITY REPORT
# --------------------------------------------------

rows = []

for (subdivision, error_type, category), r in results.items():

    count = r["records"]

    average_rainfall = (
        r["rainfall_sum"] / count
        if count else np.nan
    )

    rows.append({
        "SUBDIVISION": subdivision,
        "ERROR_TYPE": error_type,
        "INTENSITY": category,
        "RECORDS": count,
        "AVERAGE_RAINFALL_MM": average_rainfall
    })

summary = pd.DataFrame(rows)

output_path = (
    "data/processed/regional_rainfall_intensity_2024.csv"
)

summary.to_csv(output_path, index=False)

# --------------------------------------------------
# 9. DISPLAY RESULTS
# --------------------------------------------------

print("\nAEGISPULSE - REGIONAL RAINFALL INTENSITY")
print("=" * 100)

print("Model threshold:", threshold)
print("Model bias:", bias)

for subdivision in target_subdivisions:

    regional = summary[
        summary["SUBDIVISION"] == subdivision
    ]

    print(f"\n{subdivision}")
    print("-" * 100)

    count_pivot = regional.pivot(
        index="INTENSITY",
        columns="ERROR_TYPE",
        values="RECORDS"
    ).reindex(categories)

    print("\nRAINFALL INTENSITY EVENT COUNTS")
    print(count_pivot.fillna(0).astype(int).to_string())

    average_pivot = regional.pivot(
        index="INTENSITY",
        columns="ERROR_TYPE",
        values="AVERAGE_RAINFALL_MM"
    ).reindex(categories)

    print("\nAVERAGE ACTUAL RAINFALL (MM/DAY)")
    print(average_pivot.round(2).to_string())

print("\nSaved results to:")
print(output_path)

print("\nREGIONAL RAINFALL INTENSITY ANALYSIS COMPLETED")

ds.close()
