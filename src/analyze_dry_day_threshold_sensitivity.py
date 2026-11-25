
import numpy as np
import pandas as pd

VALIDATION_PATH = (
    "data/processed/features/validation.csv.gz"
)

OUTPUT_PATH = (
    "data/processed/dry_day_threshold_sensitivity.csv"
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

THRESHOLDS = np.arange(0.05, 0.50, 0.05)

print("Loading frozen Model V2...")

model = np.load(
    "models/historical_rainfall_model_v2.npz"
)

weights = model["weights"]
bias = float(model["bias"])
mean = model["mean"]
std = model["std"]

print("Original threshold:", float(model["threshold"]))

# Store confusion counts for each threshold
results = {
    round(float(t), 2): {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN": 0
    }
    for t in THRESHOLDS
}

total_dry_records = 0

print("\nAnalyzing validation records after a dry previous day...")

for chunk in pd.read_csv(
    VALIDATION_PATH,
    chunksize=50000
):

    chunk["RAIN_TODAY"] = (
        chunk["RAINFALL"] > 0
    ).astype(int)

    # Previous-day dry records only
    dry_mask = (
        chunk["RAIN_TODAY"].to_numpy() == 0
    )

    if not dry_mask.any():
        continue

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

    actual = chunk[
        "RAIN_OCCURRED"
    ].to_numpy(dtype=int)

    probabilities = probabilities[dry_mask]
    actual = actual[dry_mask]

    total_dry_records += len(actual)

    for threshold in THRESHOLDS:

        predicted = (
            probabilities >= threshold
        ).astype(int)

        tp = int(np.sum(
            (actual == 1) & (predicted == 1)
        ))

        fp = int(np.sum(
            (actual == 0) & (predicted == 1)
        ))

        fn = int(np.sum(
            (actual == 1) & (predicted == 0)
        ))

        tn = int(np.sum(
            (actual == 0) & (predicted == 0)
        ))

        key = round(float(threshold), 2)

        results[key]["TP"] += tp
        results[key]["FP"] += fp
        results[key]["FN"] += fn
        results[key]["TN"] += tn

# Calculate metrics
rows = []

for threshold, counts in results.items():

    tp = counts["TP"]
    fp = counts["FP"]
    fn = counts["FN"]
    tn = counts["TN"]

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else np.nan
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else np.nan
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (
            pd.notna(precision)
            and pd.notna(recall)
            and precision + recall > 0
        )
        else np.nan
    )

    rows.append({
        "THRESHOLD": threshold,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "PRECISION": precision,
        "RECALL": recall,
        "F1_SCORE": f1,
        "PREDICTED_RAINY_EVENTS": tp + fp
    })

summary = pd.DataFrame(rows)

summary.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 115)
print("DRY PREVIOUS-DAY THRESHOLD SENSITIVITY")
print("=" * 115)

print("Total dry previous-day records:", total_dry_records)

print(
    summary.to_string(
        index=False,
        formatters={
            "PRECISION": lambda x: f"{x:.4f}",
            "RECALL": lambda x: f"{x:.4f}",
            "F1_SCORE": lambda x: f"{x:.4f}"
        }
    )
)

print("\nResults saved to:")
print(OUTPUT_PATH)

print("\nTHRESHOLD SENSITIVITY ANALYSIS COMPLETED.")
