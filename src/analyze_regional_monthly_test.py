import numpy as np
import pandas as pd

# Load IMD subdivision mapping
mapping = pd.read_csv(
    "data/processed/india_grid_subdivision_mapping.csv"
)

mapping = mapping[mapping["STATUS"] == "MAPPED"]

subdivision_lookup = {
    (float(row.LATITUDE), float(row.LONGITUDE)): row.SUBDIVISION
    for row in mapping.itertuples()
}

# Load Model V2
model = np.load("models/historical_rainfall_model_v2.npz")

weights = model["weights"]
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

# Subdivisions that declined in 2024
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
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0
        }

# Process 2024 test data
for chunk in pd.read_csv(
    "data/processed/features/test.csv.gz",
    chunksize=50000
):

    chunk["RAIN_TODAY"] = (
        chunk["RAINFALL"] > 0
    ).astype(int)

    actual = chunk["RAIN_OCCURRED"].to_numpy(dtype=int)

    X = chunk[features].to_numpy(dtype=float)
    X = (X - mean) / std

    probabilities = 1 / (
        1 + np.exp(-np.clip(X @ weights, -500, 500))
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

    for subdivision in target_subdivisions:

        mask = np.array([
            value == subdivision for value in assigned
        ])

        if not mask.any():
            continue

        for month in range(1, 13):

            month_mask = mask & (months == month)

            if not month_mask.any():
                continue

            y_true = actual[month_mask]
            y_pred = predictions[month_mask]

            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())

            r = results[(subdivision, month)]

            r["records"] += len(y_true)
            r["tp"] += tp
            r["fp"] += fp
            r["tn"] += tn
            r["fn"] += fn

# Calculate monthly F1 scores
rows = []

for (subdivision, month), r in results.items():

    tp = r["tp"]
    fp = r["fp"]
    fn = r["fn"]

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    rows.append({
        "SUBDIVISION": subdivision,
        "MONTH": month,
        "RECORDS": r["records"],
        "PRECISION": precision,
        "RECALL": recall,
        "F1_SCORE": f1
    })

summary = pd.DataFrame(rows)

summary = summary.sort_values(
    ["SUBDIVISION", "MONTH"]
)

print("\nAEGISPULSE - 2024 MONTHLY REGIONAL TEST")
print("=" * 85)

for subdivision in target_subdivisions:

    regional = summary[
        summary["SUBDIVISION"] == subdivision
    ]

    print(f"\n{subdivision}")
    print("-" * 85)

    print(
        regional[
            ["MONTH", "RECORDS", "PRECISION", "RECALL", "F1_SCORE"]
        ].to_string(
            index=False,
            formatters={
                "PRECISION": "{:.4f}".format,
                "RECALL": "{:.4f}".format,
                "F1_SCORE": "{:.4f}".format
            }
        )
    )

summary.to_csv(
    "data/processed/regional_monthly_test_performance.csv",
    index=False
)

print("\nSaved results to:")
print("data/processed/regional_monthly_test_performance.csv")

print("\nMONTHLY REGIONAL TEST ANALYSIS COMPLETED")