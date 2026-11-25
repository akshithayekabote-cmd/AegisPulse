
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

VALIDATION_PATH = (
    "data/processed/features/validation.csv.gz"
)

MAPPING_PATH = (
    "data/processed/india_grid_subdivision_mapping.csv"
)

REGIONAL_OUTPUT = (
    "data/processed/"
    "dry_period_detection_validation_subdivisions.csv"
)

OVERALL_OUTPUT = (
    "data/processed/"
    "dry_period_detection_validation_all_india.csv"
)

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

subdivisions = sorted(
    mapping["SUBDIVISION"].unique()
)

print("Mapped subdivisions:", len(subdivisions))

# --------------------------------------------------
# 3. LOAD FROZEN MODEL V2
# --------------------------------------------------

print("\nLoading frozen Model V2...")

model = np.load(
    "models/historical_rainfall_model_v2.npz"
)

weights = model["weights"]
bias = float(model["bias"])
mean = model["mean"]
std = model["std"]
threshold = float(model["threshold"])

print("Features:", model["features"].astype(str).tolist())
print("Threshold:", threshold)
print("Bias:", bias)

# --------------------------------------------------
# 4. INITIALIZE RESULT STORAGE
# --------------------------------------------------

results = {}

for subdivision in subdivisions:

    for condition in CONDITIONS:

        for state in ["DRY", "WET"]:

            results[
                (subdivision, condition, state)
            ] = {
                "ACTUAL_RAINY_EVENTS": 0,
                "DETECTED_EVENTS": 0
            }

total_records = 0
unmapped_records = 0

# --------------------------------------------------
# 5. PROCESS VALIDATION DATA
# --------------------------------------------------

print("\nProcessing 2022-2023 validation records...")

for chunk in pd.read_csv(
    VALIDATION_PATH,
    chunksize=50000
):

    total_records += len(chunk)

    # Reconstruct the model's binary feature
    chunk["RAIN_TODAY"] = (
        chunk["RAINFALL"] > 0
    ).astype(int)

    # Calculate predictions using the frozen model
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

    # Map grid locations to IMD subdivisions
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

    mapped_mask = pd.notna(assigned)

    unmapped_records += int(
        (~mapped_mask).sum()
    )

    # Evaluate actual rainy events only
    rainy_mask = (
        mapped_mask & (actual == 1)
    )

    if not rainy_mask.any():
        continue

    # Analyze each rainfall-history condition
    for condition, column in CONDITIONS.items():

        rainfall_history = chunk[
            column
        ].to_numpy(dtype=float)

        states = np.where(
            rainfall_history == 0,
            "DRY",
            "WET"
        )

        selected = pd.DataFrame({

            "SUBDIVISION": assigned[rainy_mask],

            "PREVIOUS_STATE": states[rainy_mask],

            "DETECTED": predictions[rainy_mask]

        })

        grouped = (
            selected
            .groupby(
                ["SUBDIVISION", "PREVIOUS_STATE"]
            )
            .agg(
                ACTUAL_RAINY_EVENTS=(
                    "DETECTED", "size"
                ),
                DETECTED_EVENTS=(
                    "DETECTED", "sum"
                )
            )
        )

        for index, row in grouped.iterrows():

            subdivision, state = index

            key = (
                subdivision,
                condition,
                state
            )

            results[key][
                "ACTUAL_RAINY_EVENTS"
            ] += int(
                row["ACTUAL_RAINY_EVENTS"]
            )

            results[key][
                "DETECTED_EVENTS"
            ] += int(
                row["DETECTED_EVENTS"]
            )

# --------------------------------------------------
# 6. CREATE REGIONAL SUMMARY
# --------------------------------------------------

rows = []

for key, values in results.items():

    subdivision, condition, state = key

    actual_count = values[
        "ACTUAL_RAINY_EVENTS"
    ]

    detected = values[
        "DETECTED_EVENTS"
    ]

    missed = actual_count - detected

    rate = (
        detected / actual_count * 100
        if actual_count > 0
        else np.nan
    )

    rows.append({

        "SUBDIVISION": subdivision,
        "CONDITION": condition,
        "PREVIOUS_STATE": state,
        "ACTUAL_RAINY_EVENTS": actual_count,
        "DETECTED_EVENTS": detected,
        "MISSED_EVENTS": missed,
        "DETECTION_RATE_PERCENT": rate

    })

summary = pd.DataFrame(rows)

summary.to_csv(
    REGIONAL_OUTPUT,
    index=False
)

# --------------------------------------------------
# 7. CREATE ALL-INDIA SUMMARY
# --------------------------------------------------

national = (
    summary
    .groupby(
        ["CONDITION", "PREVIOUS_STATE"]
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

national["DETECTION_RATE_PERCENT"] = (
    national["DETECTED_EVENTS"] /
    national["ACTUAL_RAINY_EVENTS"] * 100
).replace(
    [np.inf, -np.inf],
    np.nan
)

national.to_csv(
    OVERALL_OUTPUT,
    index=False
)

# --------------------------------------------------
# 8. DISPLAY ALL-INDIA RESULTS
# --------------------------------------------------

print("\n" + "=" * 105)
print("VALIDATION ALL-INDIA DRY-PERIOD DETECTION")
print("=" * 105)

print(
    national.to_string(
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
# 9. DISPLAY REGIONAL PREVIOUS-DAY COMPARISON
# --------------------------------------------------

previous_day = summary[
    summary["CONDITION"] == "PREVIOUS_DAY"
].pivot(
    index="SUBDIVISION",
    columns="PREVIOUS_STATE",
    values="DETECTION_RATE_PERCENT"
).reset_index()

previous_day = previous_day.rename(
    columns={
        "DRY": "DRY_DAY_DETECTION_PERCENT",
        "WET": "WET_DAY_DETECTION_PERCENT"
    }
)

print("\n" + "=" * 105)
print("VALIDATION REGIONAL PREVIOUS-DAY COMPARISON")
print("=" * 105)

print(
    previous_day.to_string(
        index=False,
        formatters={
            "DRY_DAY_DETECTION_PERCENT":
                lambda x: (
                    f"{x:.2f}%"
                    if pd.notna(x)
                    else "N/A"
                ),
            "WET_DAY_DETECTION_PERCENT":
                lambda x: (
                    f"{x:.2f}%"
                    if pd.notna(x)
                    else "N/A"
                )
        }
    )
)

# --------------------------------------------------
# 10. PROCESSING CHECKS
# --------------------------------------------------

print("\n" + "=" * 80)
print("VALIDATION PROCESSING CHECKS")
print("=" * 80)

print("Total validation records:", total_records)
print("Unmapped records:", unmapped_records)
print("Mapped subdivisions:", len(subdivisions))

print("\nRegional results saved to:")
print(REGIONAL_OUTPUT)

print("\nAll-India results saved to:")
print(OVERALL_OUTPUT)

print("\nVALIDATION DRY-PERIOD ANALYSIS COMPLETED.")
