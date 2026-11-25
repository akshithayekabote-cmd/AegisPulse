
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

TEST_PATH = "data/processed/features/test.csv.gz"

MAPPING_PATH = (
    "data/processed/india_grid_subdivision_mapping.csv"
)

OUTPUT_PATH = (
    "data/processed/"
    "prediction_feature_contributions_2024.csv"
)

TARGET_SUBDIVISIONS = [
    "JAMMU & KASHMIR",
    "PUNJAB"
]

# --------------------------------------------------
# 2. LOAD MAPPING
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
# 3. LOAD FROZEN MODEL
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

features = model["features"].astype(str).tolist()

print("Features:", features)
print("Threshold:", threshold)
print("Bias:", bias)

# --------------------------------------------------
# 4. INITIALIZE AGGREGATION
# --------------------------------------------------

# Each group contains:
# region, previous-day state, prediction result

results = {}

# --------------------------------------------------
# 5. PROCESS TEST DATA
# --------------------------------------------------

print("\nProcessing test records...")

total_records = 0
unmapped_records = 0

for chunk in pd.read_csv(
    TEST_PATH,
    chunksize=50000
):

    total_records += len(chunk)

    # Derive current-day rainfall indicator
    chunk["RAIN_TODAY"] = (
        chunk["RAINFALL"] > 0
    ).astype(int)

    # Standardize features
    X_raw = chunk[features].to_numpy(
        dtype=float
    )

    X = (X_raw - mean) / std

    # Individual feature contributions to log-odds
    contributions = X * weights

    # Calculate probabilities
    logits = np.clip(
        contributions.sum(axis=1) + bias,
        -500,
        500
    )

    probabilities = 1 / (
        1 + np.exp(-logits)
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    actual = chunk[
        "RAIN_OCCURRED"
    ].to_numpy(dtype=int)

    # Map subdivisions
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

    # Only actual rainy events
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

        # Selected rows
        selected_indices = np.flatnonzero(
            selected_mask
        )

        selected_state = np.where(
            chunk["RAIN_TODAY"].to_numpy()[
                selected_indices
            ] == 0,
            "DRY",
            "WET"
        )

        selected_result = np.where(
            predictions[selected_indices] == 1,
            "DETECTED",
            "MISSED"
        )

        selected_probabilities = (
            probabilities[selected_indices]
        )

        selected_contributions = (
            contributions[selected_indices]
        )

        # Aggregate each group
        for state in ["DRY", "WET"]:

            for prediction_result in [
                "DETECTED",
                "MISSED"
            ]:

                group_mask = (
                    (selected_state == state) &
                    (selected_result == prediction_result)
                )

                if not group_mask.any():
                    continue

                key = (
                    subdivision,
                    state,
                    prediction_result
                )

                count = int(group_mask.sum())

                probability_sum = (
                    selected_probabilities[
                        group_mask
                    ].sum()
                )

                contribution_sum = (
                    selected_contributions[
                        group_mask
                    ].sum(axis=0)
                )

                if key not in results:

                    results[key] = {
                        "RECORDS": 0,
                        "PROBABILITY_SUM": 0.0,
                        "CONTRIBUTION_SUM": np.zeros(
                            len(features)
                        )
                    }

                results[key]["RECORDS"] += count

                results[key]["PROBABILITY_SUM"] += (
                    probability_sum
                )

                results[key]["CONTRIBUTION_SUM"] += (
                    contribution_sum
                )

# --------------------------------------------------
# 6. CREATE SUMMARY
# --------------------------------------------------

rows = []

for key, values in results.items():

    subdivision, state, prediction_result = key

    count = values["RECORDS"]

    mean_probability = (
        values["PROBABILITY_SUM"] / count
    )

    mean_contributions = (
        values["CONTRIBUTION_SUM"] / count
    )

    row = {
        "SUBDIVISION": subdivision,
        "PREVIOUS_DAY_STATE": state,
        "PREDICTION_RESULT": prediction_result,
        "RECORDS": count,
        "MEAN_PROBABILITY": mean_probability,
        "BIAS": bias
    }

    for i, feature in enumerate(features):

        row[feature + "_CONTRIBUTION"] = (
            mean_contributions[i]
        )

    row["MEAN_LOGIT"] = (
        mean_contributions.sum() + bias
    )

    rows.append(row)

summary = pd.DataFrame(rows)

summary = summary.sort_values(
    [
        "SUBDIVISION",
        "PREVIOUS_DAY_STATE",
        "PREDICTION_RESULT"
    ]
)

# --------------------------------------------------
# 7. DISPLAY RESULTS
# --------------------------------------------------

for subdivision in TARGET_SUBDIVISIONS:

    print("\n" + "=" * 110)
    print("FEATURE CONTRIBUTIONS:", subdivision)
    print("=" * 110)

    regional = summary[
        summary["SUBDIVISION"] == subdivision
    ]

    for _, row in regional.iterrows():

        print("\nPrevious day:", row["PREVIOUS_DAY_STATE"])
        print("Prediction:", row["PREDICTION_RESULT"])
        print("Records:", int(row["RECORDS"]))
        print(
            "Mean probability:",
            round(row["MEAN_PROBABILITY"], 4)
        )
        print(
            "Mean logit:",
            round(row["MEAN_LOGIT"], 4)
        )

        print("\nMean feature contributions:")

        for feature in features:

            column = feature + "_CONTRIBUTION"

            print(
                f"{feature:20s}",
                f"{row[column]: .6f}"
            )

# --------------------------------------------------
# 8. SAVE RESULTS
# --------------------------------------------------

summary.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nTotal test records:", total_records)
print("Unmapped records:", unmapped_records)

print("\nSaved results to:")
print(OUTPUT_PATH)

print("\nFEATURE CONTRIBUTION ANALYSIS COMPLETED.")
