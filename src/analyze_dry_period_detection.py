
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

TEST_PATH = (
    "data/processed/features/test.csv.gz"
)

MAPPING_PATH = (
    "data/processed/"
    "india_grid_subdivision_mapping.csv"
)

OUTPUT_PATH = (
    "data/processed/"
    "dry_period_detection_analysis_2024.csv"
)

TARGET_SUBDIVISIONS = [
    "JAMMU & KASHMIR",
    "PUNJAB"
]

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

CONDITIONS = {
    "PREVIOUS_DAY": "RAIN_TODAY",
    "PREVIOUS_3_DAYS": "RAIN_ROLLING_3",
    "PREVIOUS_7_DAYS": "RAIN_ROLLING_7"
}

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
# 4. INITIALIZE COUNTERS
# --------------------------------------------------

# Key:
# (subdivision, month, condition, state)
#
# Values:
# actual rainy events, detected events, missed events

results = {}

for subdivision in TARGET_SUBDIVISIONS:
    for month in range(1, 13):
        for condition in CONDITIONS:
            for state in ["DRY", "WET"]:

                results[
                    (subdivision, month, condition, state)
                ] = {
                    "ACTUAL_RAINY": 0,
                    "DETECTED": 0,
                    "MISSED": 0
                }

# --------------------------------------------------
# 5. PROCESS TEST DATA
# --------------------------------------------------

print("\nProcessing 2024 test data...")

total_records = 0
unmapped_records = 0

for chunk in pd.read_csv(
    TEST_PATH,
    chunksize=50000
):

    total_records += len(chunk)

    # Previous day relative to the target date
    chunk["RAIN_TODAY"] = (
        chunk["RAINFALL"] > 0
    ).astype(int)

    # Frozen model prediction
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

    # Map grid locations to subdivisions
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

    unmapped_records += int(
        pd.isna(assigned).sum()
    )

    months = chunk["MONTH"].to_numpy(dtype=int)

    # Only actual rainy events are relevant
    # for calculating event detection rates.
    rainy_mask = actual == 1

    for subdivision in TARGET_SUBDIVISIONS:

        region_mask = (
            assigned == subdivision
        )

        selected_mask = (
            region_mask & rainy_mask
        )

        if not selected_mask.any():
            continue

        selected_months = months[selected_mask]
        selected_predictions = predictions[selected_mask]

        selected_chunk = chunk.loc[
            selected_mask
        ]

        # Evaluate each rainfall-history condition
        for condition, column in CONDITIONS.items():

            previous_rainfall = (
                selected_chunk[column].to_numpy(
                    dtype=float
                )
            )

            states = np.where(
                previous_rainfall == 0,
                "DRY",
                "WET"
            )

            for state in ["DRY", "WET"]:

                state_mask = states == state

                if not state_mask.any():
                    continue

                for month in range(1, 13):

                    mask = (
                        state_mask &
                        (selected_months == month)
                    )

                    if not mask.any():
                        continue

                    key = (
                        subdivision,
                        month,
                        condition,
                        state
                    )

                    detected = int(
                        selected_predictions[mask].sum()
                    )

                    actual_count = int(mask.sum())

                    results[key]["ACTUAL_RAINY"] += (
                        actual_count
                    )

                    results[key]["DETECTED"] += (
                        detected
                    )

                    results[key]["MISSED"] += (
                        actual_count - detected
                    )

# --------------------------------------------------
# 6. CREATE SUMMARY
# --------------------------------------------------

rows = []

for key, values in results.items():

    subdivision, month, condition, state = key

    actual_count = values["ACTUAL_RAINY"]
    detected = values["DETECTED"]
    missed = values["MISSED"]

    detection_rate = (
        detected / actual_count * 100
        if actual_count > 0
        else np.nan
    )

    rows.append({

        "SUBDIVISION": subdivision,
        "MONTH": month,
        "CONDITION": condition,
        "PREVIOUS_STATE": state,
        "ACTUAL_RAINY_EVENTS": actual_count,
        "DETECTED_EVENTS": detected,
        "MISSED_EVENTS": missed,
        "DETECTION_RATE_PERCENT": detection_rate

    })

summary = pd.DataFrame(rows)

# --------------------------------------------------
# 7. DISPLAY MONTHLY RESULTS
# --------------------------------------------------

for subdivision in TARGET_SUBDIVISIONS:

    for condition in CONDITIONS:

        regional = summary[
            (summary["SUBDIVISION"] == subdivision) &
            (summary["CONDITION"] == condition)
        ]

        print("\n" + "=" * 100)
        print(subdivision, "|", condition)
        print("=" * 100)

        print(
            regional[
                [
                    "MONTH",
                    "PREVIOUS_STATE",
                    "ACTUAL_RAINY_EVENTS",
                    "DETECTED_EVENTS",
                    "MISSED_EVENTS",
                    "DETECTION_RATE_PERCENT"
                ]
            ].to_string(
                index=False,
                formatters={
                    "DETECTION_RATE_PERCENT":
                        lambda x: (
                            f"{x:.2f}%"
                            if pd.notna(x)
                            else "N/A"
                        )
                }
            )
        )

# --------------------------------------------------
# 8. CALCULATE OVERALL REGIONAL RESULTS
# --------------------------------------------------

overall = (
    summary
    .groupby(
        [
            "SUBDIVISION",
            "CONDITION",
            "PREVIOUS_STATE"
        ]
    )
    .agg(
        ACTUAL_RAINY_EVENTS=(
            "ACTUAL_RAINY_EVENTS", "sum"
        ),
        DETECTED_EVENTS=(
            "DETECTED_EVENTS", "sum"
        ),
        MISSED_EVENTS=(
            "MISSED_EVENTS", "sum"
        )
    )
    .reset_index()
)

overall["DETECTION_RATE_PERCENT"] = (
    overall["DETECTED_EVENTS"] /
    overall["ACTUAL_RAINY_EVENTS"] * 100
).replace([np.inf, -np.inf], np.nan)

print("\n" + "=" * 110)
print("OVERALL REGIONAL DETECTION AFTER DRY AND WET PERIODS")
print("=" * 110)

print(
    overall.to_string(
        index=False,
        formatters={
            "DETECTION_RATE_PERCENT":
                lambda x: (
                    f"{x:.2f}%"
                    if pd.notna(x)
                    else "N/A"
                )
        }
    )
)

# --------------------------------------------------
# 9. SAVE RESULTS
# --------------------------------------------------

summary.to_csv(
    OUTPUT_PATH,
    index=False
)

ds_output = OUTPUT_PATH.replace(
    ".csv",
    "_overall.csv"
)

overall.to_csv(
    ds_output,
    index=False
)

print("\nTotal test records:", total_records)
print("Unmapped records:", unmapped_records)

print("\nMonthly results saved to:")
print(OUTPUT_PATH)

print("\nOverall results saved to:")
print(ds_output)

print("\nDRY PERIOD DETECTION ANALYSIS COMPLETED")
