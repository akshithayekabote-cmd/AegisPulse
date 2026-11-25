
import numpy as np
import pandas as pd
import os

folder = "data/processed/features"
model_path = "models/historical_rainfall_model.npz"

model = np.load(model_path)

weights = model["weights"]
mean = model["mean"]
std = model["std"]
bias = float(model["bias"])
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
    "DAY_OF_YEAR"
]

results = {
    "WET TODAY": np.zeros(4, dtype=np.int64),
    "DRY TODAY": np.zeros(4, dtype=np.int64)
}

print("AEGISPULSE - WET AND DRY DAY ANALYSIS")
print("=" * 60)

for chunk in pd.read_csv(
    os.path.join(folder, "validation.csv.gz"),
    usecols=features + ["RAIN_OCCURRED"],
    chunksize=50000
):

    x = chunk[features].to_numpy(dtype=np.float64)
    actual = chunk["RAIN_OCCURRED"].to_numpy()

    today_wet = chunk["RAINFALL"].to_numpy() > 0

    x = (x - mean) / std

    z = np.clip(x @ weights + bias, -35, 35)
    probability = 1 / (1 + np.exp(-z))

    model_prediction = probability >= threshold
    baseline_prediction = today_wet

    for name, mask in [
        ("WET TODAY", today_wet),
        ("DRY TODAY", ~today_wet)
    ]:

        y = actual[mask]
        predicted = model_prediction[mask]

        tp = np.sum((y == 1) & predicted)
        tn = np.sum((y == 0) & ~predicted)
        fp = np.sum((y == 0) & predicted)
        fn = np.sum((y == 1) & ~predicted)

        results[name] += [tp, tn, fp, fn]

print("\nMODEL PERFORMANCE BY TODAY'S RAINFALL")
print("=" * 60)

for name, values in results.items():

    tp, tn, fp, fn = values

    total = tp + tn + fp + fn

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-12)
    )

    accuracy = (tp + tn) / total

    print(f"\n{name}")
    print("-" * 40)
    print("Records:", total)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("False Positives:", fp)
    print("False Negatives:", fn)

print("\nANALYSIS COMPLETED")
print("=" * 60)
