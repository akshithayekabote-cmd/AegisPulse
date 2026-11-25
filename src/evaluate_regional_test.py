
import numpy as np
import pandas as pd

# Load official IMD subdivision mapping
mapping = pd.read_csv(
    "data/processed/india_grid_subdivision_mapping.csv"
)

mapping = mapping[mapping["STATUS"] == "MAPPED"]

subdivision_lookup = {
    (float(row.LATITUDE), float(row.LONGITUDE)): row.SUBDIVISION
    for row in mapping.itertuples()
}

# Load trained Model V2
model = np.load("models/historical_rainfall_model_v2.npz")

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

# Initialize regional results
results = {}

for subdivision in sorted(set(subdivision_lookup.values())):
    results[subdivision] = {
        "records": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0
    }

unmapped_records = 0

# Load untouched 2024 test data
for chunk in pd.read_csv(
    "data/processed/features/test.csv.gz",
    chunksize=50000
):

    actual = chunk["RAIN_OCCURRED"].to_numpy(dtype=int)

    chunk["RAIN_TODAY"] = (
        chunk["RAINFALL"] > 0
    ).astype(int)

    X = chunk[features].to_numpy(dtype=float)

    # Standardize features using saved training statistics
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

    # Identify official IMD subdivisions
    assigned = [
        subdivision_lookup.get(
            (float(lat), float(lon))
        )
        for lat, lon in zip(
            chunk["LATITUDE"],
            chunk["LONGITUDE"]
        )
    ]

    for subdivision in set(assigned):

        if subdivision is None:
            continue

        mask = np.array([
            value == subdivision for value in assigned
        ])

        y_true = actual[mask]
        y_pred = predictions[mask]

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        results[subdivision]["records"] += len(y_true)
        results[subdivision]["tp"] += tp
        results[subdivision]["fp"] += fp
        results[subdivision]["tn"] += tn
        results[subdivision]["fn"] += fn

    unmapped_records += assigned.count(None)

# Calculate regional metrics
rows = []

for subdivision, r in results.items():

    tp = r["tp"]
    fp = r["fp"]
    tn = r["tn"]
    fn = r["fn"]

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    accuracy = (
        (tp + tn) / r["records"]
        if r["records"] else 0
    )

    rows.append({
        "SUBDIVISION": subdivision,
        "RECORDS": r["records"],
        "PRECISION": precision,
        "RECALL": recall,
        "F1_SCORE": f1,
        "ACCURACY": accuracy
    })

summary = pd.DataFrame(rows)

summary = summary.sort_values(
    "F1_SCORE",
    ascending=False
)

print("\nAEGISPULSE - OFFICIAL IMD 2024 REGIONAL TEST")
print("=" * 75)
print("Model threshold:", threshold)
print("Model bias:", bias)

print("\nREGIONAL TEST PERFORMANCE")
print("=" * 75)

print(
    summary.to_string(
        index=False,
        formatters={
            "PRECISION": "{:.4f}".format,
            "RECALL": "{:.4f}".format,
            "F1_SCORE": "{:.4f}".format,
            "ACCURACY": "{:.4f}".format
        }
    )
)

print("\nOVERALL SUMMARY")
print("=" * 75)

print("Mapped test records:", summary["RECORDS"].sum())
print("Unmapped test records:", unmapped_records)
print("Subdivisions analyzed:", len(summary))

summary.to_csv(
    "data/processed/imd_regional_test_performance.csv",
    index=False
)

print("\nSaved results to:")
print("data/processed/imd_regional_test_performance.csv")

print("\n2024 REGIONAL TEST EVALUATION COMPLETED")
