
import pandas as pd
import numpy as np

validation_path = "data/processed/features/validation.csv.gz"
mapping_path = "data/processed/india_grid_subdivision_mapping.csv"
model_path = "models/historical_rainfall_model_v2.npz"

# Load trained Model V2
model = np.load(model_path)

weights = model["weights"]
bias = float(model["bias"])
mean = model["mean"]
std = model["std"]
threshold = float(model["threshold"])

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
    "DAY_OF_YEAR"
]

# Load official IMD subdivision mapping
mapping = pd.read_csv(mapping_path)

mapping = mapping[
    mapping["STATUS"] == "MAPPED"
].copy()

subdivision_lookup = {
    (float(row.LATITUDE), float(row.LONGITUDE)): row.SUBDIVISION
    for row in mapping.itertuples()
}

subdivisions = sorted(mapping["SUBDIVISION"].unique())

results = {
    subdivision: {
        "baseline": np.zeros(4, dtype=np.int64),
        "v2": np.zeros(4, dtype=np.int64),
        "records": 0
    }
    for subdivision in subdivisions
}

unmapped_records = 0


def calculate_counts(actual, predicted):

    tp = np.sum((actual == 1) & predicted)
    tn = np.sum((actual == 0) & ~predicted)
    fp = np.sum((actual == 0) & predicted)
    fn = np.sum((actual == 1) & ~predicted)

    return np.array([tp, tn, fp, fn], dtype=np.int64)


def calculate_metrics(counts):

    tp, tn, fp, fn = counts

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-12)
    )

    return accuracy, precision, recall, f1


print("\nAEGISPULSE - OFFICIAL IMD REGIONAL PERFORMANCE")
print("=" * 75)

print("Mapped grid locations:", len(mapping))
print("IMD subdivisions:", len(subdivisions))
print("Model threshold:", threshold)
print("Model bias:", bias)

# Process validation data in chunks
for chunk in pd.read_csv(
    validation_path,
    chunksize=50000
):

    actual = chunk["RAIN_OCCURRED"].to_numpy(dtype=int)

    lat = chunk["LATITUDE"].to_numpy()
    lon = chunk["LONGITUDE"].to_numpy()

    # Assign official IMD subdivisions
    assigned = [
        subdivision_lookup.get(
            (float(a), float(b)),
            "UNMAPPED"
        )
        for a, b in zip(lat, lon)
    ]

    assigned = np.asarray(assigned, dtype=object)

    unmapped_mask = assigned == "UNMAPPED"
    unmapped_records += int(np.sum(unmapped_mask))

    # Persistence baseline
    baseline_pred = (
        chunk["RAINFALL"].to_numpy() > 0
    )

    # Prepare Model V2 features
    x = chunk[FEATURES].to_numpy(dtype=np.float64)

    rain_today = (
        chunk["RAINFALL"].to_numpy() > 0
    ).astype(np.float64)

    x = np.column_stack((x, rain_today))

    # Standardize using saved training statistics
    x = (x - mean) / std

    # Calculate probabilities using weights AND bias
    z = np.clip(x @ weights + bias, -500, 500)

    probabilities = 1 / (1 + np.exp(-z))

    v2_pred = probabilities >= threshold

    # Aggregate regional performance
    for subdivision in np.unique(assigned):

        if subdivision == "UNMAPPED":
            continue

        mask = assigned == subdivision

        y = actual[mask]

        results[subdivision]["baseline"] += calculate_counts(
            y,
            baseline_pred[mask]
        )

        results[subdivision]["v2"] += calculate_counts(
            y,
            v2_pred[mask]
        )

        results[subdivision]["records"] += int(np.sum(mask))


# Calculate regional metrics
print("\nOFFICIAL IMD SUBDIVISION PERFORMANCE")
print("=" * 75)

summary = []

for subdivision in subdivisions:

    baseline = calculate_metrics(
        results[subdivision]["baseline"]
    )

    v2 = calculate_metrics(
        results[subdivision]["v2"]
    )

    improvement = v2[3] - baseline[3]

    summary.append({
        "SUBDIVISION": subdivision,
        "RECORDS": results[subdivision]["records"],
        "BASELINE_F1": baseline[3],
        "V2_F1": v2[3],
        "F1_IMPROVEMENT": improvement,
        "BASELINE_PRECISION": baseline[1],
        "V2_PRECISION": v2[1],
        "BASELINE_RECALL": baseline[2],
        "V2_RECALL": v2[2],
        "BASELINE_ACCURACY": baseline[0],
        "V2_ACCURACY": v2[0]
    })

summary_df = pd.DataFrame(summary)

summary_df = summary_df.sort_values(
    "V2_F1",
    ascending=False
)

print(
    summary_df.to_string(
        index=False,
        formatters={
            column: "{:.4f}".format
            for column in summary_df.select_dtypes(
                include="number"
            ).columns
            if column != "RECORDS"
        }
    )
)

# Overall analysis
print("\nOVERALL SUMMARY")
print("=" * 75)

total_records = sum(
    item["records"] for item in results.values()
)

print("Mapped validation records:", total_records)
print("Unmapped validation records:", unmapped_records)

print("Subdivisions analyzed:", len(subdivisions))

print(
    "Subdivisions where V2 improved:",
    int((summary_df["F1_IMPROVEMENT"] > 0).sum()),
    "out of",
    len(summary_df)
)

# Save regional results
output_path = "data/processed/imd_regional_performance.csv"

summary_df.to_csv(output_path, index=False)

print("\nSaved regional results to:")
print(output_path)

print("\nOFFICIAL IMD REGIONAL ANALYSIS COMPLETED")
print("=" * 75)
