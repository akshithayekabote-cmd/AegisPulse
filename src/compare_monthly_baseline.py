
import pandas as pd
import numpy as np
import os

validation_path = "data/processed/features/validation.csv.gz"
model_path = "models/historical_rainfall_model_v2.npz"

model = np.load(model_path)

weights = model["weights"]
mean = model["mean"]
std = model["std"]
bias = float(model["bias"])
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

# Each month stores:
# Baseline: TP, TN, FP, FN
# V2:       TP, TN, FP, FN

results = {
    month: {
        "baseline": np.zeros(4, dtype=np.int64),
        "v2": np.zeros(4, dtype=np.int64)
    }
    for month in range(1, 13)
}


def calculate_counts(actual, predicted):
    tp = np.sum((actual == 1) & predicted)
    tn = np.sum((actual == 0) & ~predicted)
    fp = np.sum((actual == 0) & predicted)
    fn = np.sum((actual == 1) & ~predicted)

    return np.array([tp, tn, fp, fn])


print("AEGISPULSE - MONTHLY BASELINE COMPARISON")
print("=" * 70)

for chunk in pd.read_csv(
    validation_path,
    chunksize=50000
):

    actual = chunk["RAIN_OCCURRED"].to_numpy()
    months = chunk["MONTH"].to_numpy()

    # Persistence baseline
    baseline_pred = (
        chunk["RAINFALL"].to_numpy() > 0
    )

    # AegisPulse V2
    x = chunk[FEATURES].to_numpy(dtype=np.float64)

    rain_today = (
        chunk["RAINFALL"].to_numpy() > 0
    ).astype(np.float64)

    x = np.column_stack((x, rain_today))

    x = (x - mean) / std

    z = np.clip(x @ weights + bias, -35, 35)
    probabilities = 1 / (1 + np.exp(-z))

    v2_pred = probabilities >= threshold

    for month in range(1, 13):

        mask = months == month

        if not np.any(mask):
            continue

        y = actual[mask]

        results[month]["baseline"] += calculate_counts(
            y, baseline_pred[mask]
        )

        results[month]["v2"] += calculate_counts(
            y, v2_pred[mask]
        )


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


print("\nMONTHLY F1-SCORE COMPARISON")
print("=" * 70)

for month in range(1, 13):

    baseline = calculate_metrics(
        results[month]["baseline"]
    )

    v2 = calculate_metrics(
        results[month]["v2"]
    )

    improvement = v2[3] - baseline[3]

    print(f"\nMONTH: {month}")
    print("-" * 45)

    print("Baseline F1:", round(baseline[3], 4))
    print("AegisPulse V2 F1:", round(v2[3], 4))
    print("F1 Improvement:", round(improvement, 4))

    print("\nBaseline Accuracy:", round(baseline[0], 4))
    print("V2 Accuracy:", round(v2[0], 4))

    print("Baseline Recall:", round(baseline[2], 4))
    print("V2 Recall:", round(v2[2], 4))

print("\nMONTHLY COMPARISON COMPLETED")
print("=" * 70)
