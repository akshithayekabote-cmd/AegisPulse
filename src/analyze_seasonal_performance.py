
import pandas as pd
import numpy as np
import os

folder = "data/processed/features"
model_path = "models/historical_rainfall_model_v2.npz"
validation_path = os.path.join(folder, "validation.csv.gz")

model = np.load(model_path)

weights = model["weights"]
mean = model["mean"]
std = model["std"]
bias = float(model["bias"])
threshold = float(model["threshold"])

BASE_FEATURES = [
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

def prepare_features(chunk):

    x = chunk[BASE_FEATURES].to_numpy(dtype=np.float64)

    rain_today = (
        chunk["RAINFALL"].to_numpy() > 0
    ).astype(np.float64)

    return np.column_stack((x, rain_today))


monthly_results = {
    month: np.zeros(5, dtype=np.int64)
    for month in range(1, 13)
}

print("AEGISPULSE - SEASONAL PERFORMANCE ANALYSIS")
print("=" * 65)

for chunk in pd.read_csv(
    validation_path,
    chunksize=50000
):

    x = prepare_features(chunk)

    actual = chunk["RAIN_OCCURRED"].to_numpy()
    months = chunk["MONTH"].to_numpy()

    x = (x - mean) / std

    z = np.clip(x @ weights + bias, -35, 35)
    probabilities = 1 / (1 + np.exp(-z))

    predicted = probabilities >= threshold

    for month in range(1, 13):

        mask = months == month

        if not np.any(mask):
            continue

        y = actual[mask]
        p = predicted[mask]

        tp = np.sum((y == 1) & p)
        tn = np.sum((y == 0) & ~p)
        fp = np.sum((y == 0) & p)
        fn = np.sum((y == 1) & ~p)

        positive = np.sum(y == 1)

        monthly_results[month] += [
            tp, tn, fp, fn, positive
        ]

print("\nMONTHLY MODEL PERFORMANCE")
print("=" * 65)

for month, values in monthly_results.items():

    tp, tn, fp, fn, positive = values

    total = tp + tn + fp + fn

    if total == 0:
        continue

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-12)
    )

    accuracy = (tp + tn) / total
    occurrence_rate = positive / total

    print(f"\nMONTH: {month}")
    print("-" * 40)
    print("Records:", total)
    print("Rain occurrence:", round(occurrence_rate * 100, 2), "%")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))

print("\nSEASONAL ANALYSIS COMPLETED")
print("=" * 65)
