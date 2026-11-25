
import numpy as np
import pandas as pd
import xarray as xr

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATA_PATH = (
    "data/processed/"
    "india_rainfall_cleaned_2015_2024.nc"
)

MAPPING_PATH = (
    "data/processed/"
    "india_grid_subdivision_mapping.csv"
)

TEST_PATH = (
    "data/processed/features/test.csv.gz"
)

OUTPUT_PATH = (
    "data/processed/"
    "weak_region_error_characteristics_2024.csv"
)

TARGET_CASES = {
    ("JAMMU & KASHMIR", 10),
    ("JAMMU & KASHMIR", 11),
    ("PUNJAB", 5),
    ("PUNJAB", 10)
}

FEATURES = [
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
# 2. LOAD SUBDIVISION MAPPING
# --------------------------------------------------

print("Loading subdivision mapping...")

mapping = pd.read_csv(MAPPING_PATH)

mapping = mapping[
    mapping["STATUS"] == "MAPPED"
]

subdivision_lookup = {
    (
        round(float(row.LATITUDE), 2),
        round(float(row.LONGITUDE), 2)
    ): row.SUBDIVISION
    for row in mapping.itertuples()
}

# --------------------------------------------------
# 3. LOAD FROZEN MODEL V2
# --------------------------------------------------

print("Loading frozen Model V2...")

model = np.load(
    "models/historical_rainfall_model_v2.npz"
)

weights = model["weights"]
bias = float(model["bias"])
mean = model["mean"]
std = model["std"]
threshold = float(model["threshold"])

print("Threshold:", threshold)
print("Bias:", bias)

# --------------------------------------------------
# 4. LOAD HISTORICAL RAINFALL
# --------------------------------------------------

print("Loading historical rainfall dataset...")

ds = xr.open_dataset(
    DATA_PATH,
    engine="scipy"
)

rainfall = ds["RAINFALL"].values

dates = pd.to_datetime(
    ds["TIME"].values
)

latitudes = ds["LATITUDE"].values
longitudes = ds["LONGITUDE"].values

date_lookup = {
    pd.Timestamp(date): index
    for index, date in enumerate(dates)
}

location_lookup = {
    (
        round(float(lat), 2),
        round(float(lon), 2)
    ): index
    for index, (lat, lon) in enumerate(
        zip(latitudes, longitudes)
    )
}

print("Historical rainfall shape:", rainfall.shape)

# --------------------------------------------------
# 5. PROCESS TEST DATA
# --------------------------------------------------

print("\nAnalyzing selected regional cases...")

collected_records = []

total_test_records = 0
invalid_dates = 0
rainfall_mismatches = 0

for chunk in pd.read_csv(
    TEST_PATH,
    chunksize=50000
):

    total_test_records += len(chunk)

    chunk["RAIN_TODAY"] = (
        chunk["RAINFALL"] > 0
    ).astype(int)

    X = chunk[FEATURES].to_numpy(dtype=float)

    X = (X - mean) / std

    probabilities = 1 / (
        1 + np.exp(
            -np.clip(
                X @ weights + bias,
                -500,
                500
            )
        )
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    actual = chunk[
        "RAIN_OCCURRED"
    ].to_numpy(dtype=int)

    assigned = np.array([
        subdivision_lookup.get(
            (
                round(float(lat), 2),
                round(float(lon), 2)
            )
        )
        for lat, lon in zip(
            chunk["LATITUDE"],
            chunk["LONGITUDE"]
        )
    ], dtype=object)

    months = chunk["MONTH"].to_numpy(dtype=int)

    selected_mask = np.array([
        (subdivision, month) in TARGET_CASES
        for subdivision, month in zip(
            assigned,
            months
        )
    ])

    if not selected_mask.any():
        continue

    selected = chunk.loc[selected_mask].copy()

    selected_subdivisions = assigned[selected_mask]
    selected_actual = actual[selected_mask]
    selected_predictions = predictions[selected_mask]
    selected_probabilities = probabilities[selected_mask]

    # --------------------------------------------------
    # 6. RECOVER TARGET DATE
    # --------------------------------------------------

    target_dates = (
        pd.Timestamp("2024-01-01") +
        pd.to_timedelta(
            selected["DAY_OF_YEAR"].to_numpy(dtype=int) - 1,
            unit="D"
        )
    )

    date_indices = np.array([
        date_lookup.get(pd.Timestamp(date), -1)
        for date in target_dates
    ])

    # --------------------------------------------------
    # 7. RECOVER GRID LOCATION
    # --------------------------------------------------

    location_indices = np.array([
        location_lookup.get(
            (
                round(float(lat), 2),
                round(float(lon), 2)
            ),
            -1
        )
        for lat, lon in zip(
            selected["LATITUDE"],
            selected["LONGITUDE"]
        )
    ])

    valid_mask = (
        (date_indices >= 0) &
        (location_indices >= 0)
    )

    invalid_dates += int(
        (~valid_mask).sum()
    )

    target_rainfall = np.full(
        len(selected),
        np.nan
    )

    target_rainfall[valid_mask] = rainfall[
        date_indices[valid_mask],
        location_indices[valid_mask]
    ]

    # --------------------------------------------------
    # 8. VALIDATE RAINFALL LABELS
    # --------------------------------------------------

    valid_positions = np.where(valid_mask)[0]

    mismatch_mask = (
        (target_rainfall[valid_positions] > 0) !=
        (selected_actual[valid_positions] == 1)
    )

    rainfall_mismatches += int(
        mismatch_mask.sum()
    )

    verified_mask = np.zeros(
        len(selected),
        dtype=bool
    )

    verified_mask[valid_positions] = ~mismatch_mask

    if not verified_mask.any():
        continue

    # --------------------------------------------------
    # 9. CLASSIFY TP, FP, FN AND TN CORRECTLY
    # --------------------------------------------------

    y_true = selected_actual[verified_mask]
    y_pred = selected_predictions[verified_mask]

    categories = np.select(
        [
            (y_true == 1) & (y_pred == 1),
            (y_true == 0) & (y_pred == 1),
            (y_true == 1) & (y_pred == 0)
        ],
        [
            "TP",
            "FP",
            "FN"
        ],
        default="TN"
    )

    # Keep only TP, FP and FN.
    # True negatives are excluded from this analysis.
    error_mask = categories != "TN"

    if not error_mask.any():
        continue

    selected_verified = selected.iloc[
        np.where(verified_mask)[0]
    ]

    result = pd.DataFrame({

        "SUBDIVISION":
            selected_subdivisions[verified_mask][error_mask],

        "MONTH":
            months[selected_mask][verified_mask][error_mask],

        "CATEGORY":
            categories[error_mask],

        "PROBABILITY":
            selected_probabilities[verified_mask][error_mask],

        "ACTUAL_RAINFALL_MM":
            target_rainfall[verified_mask][error_mask],

        "RAIN_TODAY":
            selected_verified["RAINFALL"].to_numpy(dtype=float)[error_mask],

        "RAIN_LAG_1":
            selected_verified["RAIN_LAG_1"].to_numpy(dtype=float)[error_mask],

        "RAIN_LAG_3":
            selected_verified["RAIN_LAG_3"].to_numpy(dtype=float)[error_mask],

        "RAIN_LAG_7":
            selected_verified["RAIN_LAG_7"].to_numpy(dtype=float)[error_mask],

        "RAIN_ROLLING_3":
            selected_verified["RAIN_ROLLING_3"].to_numpy(dtype=float)[error_mask],

        "RAIN_ROLLING_7":
            selected_verified["RAIN_ROLLING_7"].to_numpy(dtype=float)[error_mask]

    })

    collected_records.append(result)

# --------------------------------------------------
# 10. DATA VALIDATION
# --------------------------------------------------

print("\nDATA VALIDATION")
print("=" * 65)

print("Total test records:", total_test_records)
print("Invalid dates or locations:", invalid_dates)
print("Target rainfall mismatches:", rainfall_mismatches)

if invalid_dates > 0:
    raise ValueError(
        "Invalid target dates or grid locations found."
    )

if rainfall_mismatches > 0:
    raise ValueError(
        "Target rainfall does not match RAIN_OCCURRED."
    )

if not collected_records:
    raise ValueError(
        "No verified error records found."
    )

df = pd.concat(
    collected_records,
    ignore_index=True
)

# --------------------------------------------------
# 11. SUMMARIZE ERROR CHARACTERISTICS
# --------------------------------------------------

summary = (
    df.groupby(
        ["SUBDIVISION", "MONTH", "CATEGORY"]
    )
    .agg(
        RECORDS=("CATEGORY", "size"),

        MEAN_PROBABILITY=("PROBABILITY", "mean"),

        MEAN_ACTUAL_RAINFALL=(
            "ACTUAL_RAINFALL_MM", "mean"
        ),

        MEDIAN_ACTUAL_RAINFALL=(
            "ACTUAL_RAINFALL_MM", "median"
        ),

        MEAN_RAIN_TODAY=("RAIN_TODAY", "mean"),

        MEAN_RAIN_LAG_1=("RAIN_LAG_1", "mean"),

        MEAN_RAIN_LAG_3=("RAIN_LAG_3", "mean"),

        MEAN_RAIN_LAG_7=("RAIN_LAG_7", "mean"),

        MEAN_ROLLING_3=("RAIN_ROLLING_3", "mean"),

        MEAN_ROLLING_7=("RAIN_ROLLING_7", "mean"),

        DRY_PREVIOUS_DAY_PERCENT=(
            "RAIN_LAG_1",
            lambda x: (x == 0).mean() * 100
        ),

        DRY_PREVIOUS_3_DAYS_PERCENT=(
            "RAIN_ROLLING_3",
            lambda x: (x == 0).mean() * 100
        ),

        DRY_PREVIOUS_7_DAYS_PERCENT=(
            "RAIN_ROLLING_7",
            lambda x: (x == 0).mean() * 100
        )
    )
    .reset_index()
)

# --------------------------------------------------
# 12. DISPLAY RESULTS
# --------------------------------------------------

for subdivision, month in sorted(TARGET_CASES):

    regional = summary[
        (summary["SUBDIVISION"] == subdivision) &
        (summary["MONTH"] == month)
    ]

    print("\n" + "=" * 100)
    print(subdivision, "| MONTH:", month)
    print("=" * 100)

    print(
        regional.to_string(
            index=False,
            formatters={
                column: lambda x: (
                    f"{x:.3f}"
                    if pd.notna(x)
                    else "N/A"
                )
                for column in regional.select_dtypes(
                    include="number"
                ).columns
                if column not in ["MONTH", "RECORDS"]
            }
        )
    )

# --------------------------------------------------
# 13. SAVE RESULTS
# --------------------------------------------------

summary.to_csv(
    OUTPUT_PATH,
    index=False
)

ds.close()

print("\nSaved results to:")
print(OUTPUT_PATH)

print("\nERROR CHARACTERISTICS ANALYSIS COMPLETED")
