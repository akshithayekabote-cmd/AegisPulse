
import numpy as np
import pandas as pd

# Load official IMD subdivision mapping
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

# Load frozen Model V2
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

results = {}

for subdivision in target_subdivisions:
    for month in range(1, 13):
        results[(subdivision, month)] = {
            "records": 0,
            "actual_rainy": 0,
            "predicted_rainy": 0,
            "correct_rainy": 0
        }

total_records = 0
unmapped_records = 0

# Process 2024 test dataset
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

    assigned = [
        subdivision_lookup.get(
            (float(lat), float(lon))
        )
        for lat, lon in zip(
            chunk["LATITUDE"],
            chunk["LONGITUDE"]
        )
    ]

    months = chunk["MONTH"].to_numpy(dtype=int)

    total_records += len(chunk)
    unmapped_records += sum(
        value is None for value in assigned
    )

    for subdivision in target_subdivisions:

        region_mask = np.array([
            value == subdivision for value in assigned
        ])

        if not region_mask.any():
            continue

        for month in range(1, 13):

            mask = region_mask & (months == month)

            if not mask.any():
                continue

            y_true = actual[mask]
            y_pred = predictions[mask]

            r = results[(subdivision, month)]

            r["records"] += len(y_true)
            r["actual_rainy"] += int(y_true.sum())
            r["predicted_rainy"] += int(y_pred.sum())
            r["correct_rainy"] += int(
                ((y_true == 1) & (y_pred == 1)).sum()
            )

# Create summary
rows = []

for (subdivision, month), r in results.items():

    records = r["records"]
    actual_rainy = r["actual_rainy"]
    predicted_rainy = r["predicted_rainy"]
    correct_rainy = r["correct_rainy"]

    actual_rate = (
        actual_rainy / records * 100
        if records else np.nan
    )

    predicted_rate = (
        predicted_rainy / records * 100
        if records else np.nan
    )

    rows.append({
        "SUBDIVISION": subdivision,
        "MONTH": month,
        "RECORDS": records,
        "ACTUAL_RAINY_RECORDS": actual_rainy,
        "PREDICTED_RAINY_RECORDS": predicted_rainy,
        "CORRECT_RAINY_PREDICTIONS": correct_rainy,
        "ACTUAL_RAINFALL_RATE": actual_rate,
        "PREDICTED_RAINFALL_RATE": predicted_rate
    })

summary = pd.DataFrame(rows)

summary = summary.sort_values(
    ["SUBDIVISION", "MONTH"]
)

print("\nAEGISPULSE - 2024 REGIONAL RAINFALL OCCURRENCE")
print("=" * 100)

print("Model threshold:", threshold)
print("Model bias:", bias)
print("Total test records:", total_records)
print("Unmapped records:", unmapped_records)
print("Subdivisions analyzed:", len(target_subdivisions))

for subdivision in target_subdivisions:

    regional = summary[
        summary["SUBDIVISION"] == subdivision
    ]

    print(f"\n{subdivision}")
    print("-" * 100)

    print(
        regional[
            [
                "MONTH",
                "ACTUAL_RAINY_RECORDS",
                "PREDICTED_RAINY_RECORDS",
                "CORRECT_RAINY_PREDICTIONS",
                "ACTUAL_RAINFALL_RATE",
                "PREDICTED_RAINFALL_RATE"
            ]
        ].to_string(
            index=False,
            formatters={
                "ACTUAL_RAINFALL_RATE": lambda x:
                    f"{x:.2f}%" if pd.notna(x) else "N/A",
                "PREDICTED_RAINFALL_RATE": lambda x:
                    f"{x:.2f}%" if pd.notna(x) else "N/A"
            }
        )
    )

output_path = (
    "data/processed/regional_rainfall_occurrence_2024.csv"
)

summary.to_csv(output_path, index=False)

print("\nSaved results to:")
print(output_path)

print("\nREGIONAL RAINFALL OCCURRENCE ANALYSIS COMPLETED")
