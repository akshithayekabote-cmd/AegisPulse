
import numpy as np
import pandas as pd
import os

print("AEGISPULSE - V2 vs V3 DRY-DAY DETECTION ANALYSIS")
print("=" * 70)

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

v2_test_path = "data/processed/features/test.csv.gz"
v3_test_path = "data/processed/features_v3/test_v3.csv.gz"

v2_model_path = "models/historical_rainfall_model_v2.npz"
v3_model_path = "models/historical_rainfall_model_v3.npz"

CHUNK_SIZE = 50000

# --------------------------------------------------
# FEATURES
# --------------------------------------------------

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

SPATIAL_FEATURES = [
    "NEIGHBOR_RAIN_MEAN",
    "NEIGHBOR_RAIN_MAX",
    "NEIGHBOR_RAINY_COUNT",
    "NEIGHBOR_VALID_COUNT"
]

V2_FEATURES = BASE_FEATURES + ["RAIN_TODAY"]

V3_FEATURES = (
    BASE_FEATURES
    + ["RAIN_TODAY"]
    + SPATIAL_FEATURES
)

# --------------------------------------------------
# LOAD FROZEN MODELS
# --------------------------------------------------

v2 = np.load(v2_model_path)
v3 = np.load(v3_model_path)

v2_weights = v2["weights"]
v2_bias = float(v2["bias"])
v2_mean = v2["mean"]
v2_std = v2["std"]
v2_threshold = float(v2["threshold"])

v3_weights = v3["weights"]
v3_bias = float(v3["bias"])
v3_mean = v3["mean"]
v3_std = v3["std"]
v3_threshold = float(v3["threshold"])

print("\nFrozen models loaded.")
print("V2 threshold:", v2_threshold)
print("V3 threshold:", v3_threshold)

# --------------------------------------------------
# CONDITION METRIC STORAGE
# Each entry: TP, FP, FN, TN
# --------------------------------------------------

conditions = {
    "Previous Day Dry": {
        "Model V2": [0, 0, 0, 0],
        "Model V3": [0, 0, 0, 0]
    },
    "Previous Day Wet": {
        "Model V2": [0, 0, 0, 0],
        "Model V3": [0, 0, 0, 0]
    }
}

condition_records = {
    "Previous Day Dry": 0,
    "Previous Day Wet": 0
}

# --------------------------------------------------
# LOAD TEST DATA
# --------------------------------------------------

v2_reader = pd.read_csv(
    v2_test_path,
    chunksize=CHUNK_SIZE
)

v3_reader = pd.read_csv(
    v3_test_path,
    chunksize=CHUNK_SIZE
)

print("\nAnalyzing 2024 test records...")
print("=" * 70)

# --------------------------------------------------
# PROCESS DATA
# --------------------------------------------------

from itertools import zip_longest

for chunk_number, (chunk_v2, chunk_v3) in enumerate(
    zip_longest(v2_reader, v3_reader),
    start=1
):

    if chunk_v2 is None or chunk_v3 is None:
        raise ValueError("V2 and V3 test datasets have different lengths.")

    if len(chunk_v2) != len(chunk_v3):
        raise ValueError("Chunk record counts do not match.")

    for column in ["LATITUDE", "LONGITUDE", "RAIN_OCCURRED"]:
        if not np.array_equal(
            chunk_v2[column].to_numpy(),
            chunk_v3[column].to_numpy()
        ):
            raise ValueError(
                f"Dataset alignment failed: {column}"
            )

    # --------------------------------------------------
    # CURRENT DAY RAINFALL CONDITION
    # --------------------------------------------------

    rain_today = (
        chunk_v2["RAINFALL"].to_numpy() > 0
    )

    y_true = chunk_v2["RAIN_OCCURRED"].to_numpy(
        dtype=np.int8
    )

    chunk_v2["RAIN_TODAY"] = rain_today.astype(np.float64)
    chunk_v3["RAIN_TODAY"] = rain_today.astype(np.float64)

    # --------------------------------------------------
    # V2 PREDICTIONS
    # --------------------------------------------------

    X_v2 = chunk_v2[V2_FEATURES].to_numpy(dtype=np.float64)

    X_v2 = (
        X_v2 - v2_mean
    ) / (v2_std + 1e-8)

    logits_v2 = X_v2 @ v2_weights + v2_bias

    probabilities_v2 = 1 / (
        1 + np.exp(-np.clip(logits_v2, -500, 500))
    )

    predictions_v2 = probabilities_v2 >= v2_threshold

    # --------------------------------------------------
    # V3 PREDICTIONS
    # --------------------------------------------------

    X_v3 = chunk_v3[V3_FEATURES].to_numpy(dtype=np.float64)

    X_v3 = (
        X_v3 - v3_mean
    ) / (v3_std + 1e-8)

    logits_v3 = X_v3 @ v3_weights + v3_bias

    probabilities_v3 = 1 / (
        1 + np.exp(-np.clip(logits_v3, -500, 500))
    )

    predictions_v3 = probabilities_v3 >= v3_threshold

    # --------------------------------------------------
    # GROUP-WISE METRICS
    # --------------------------------------------------

    groups = {
        "Previous Day Dry": ~rain_today,
        "Previous Day Wet": rain_today
    }

    model_predictions = {
        "Model V2": predictions_v2,
        "Model V3": predictions_v3
    }

    for group_name, mask in groups.items():

        condition_records[group_name] += int(np.sum(mask))

        actual = y_true[mask]

        for model_name, predictions in model_predictions.items():

            predicted = predictions[mask]

            tp = int(np.sum((actual == 1) & predicted))
            fp = int(np.sum((actual == 0) & predicted))
            fn = int(np.sum((actual == 1) & (~predicted)))
            tn = int(np.sum((actual == 0) & (~predicted)))

            counts = conditions[group_name][model_name]

            counts[0] += tp
            counts[1] += fp
            counts[2] += fn
            counts[3] += tn

    if chunk_number % 10 == 0:
        print(
            f"Processed {chunk_number * CHUNK_SIZE:,} records..."
        )

# --------------------------------------------------
# FINAL REPORT
# --------------------------------------------------

print("\nCONDITION-WISE 2024 TEST RESULTS")
print("=" * 70)

report = []

for group_name, models in conditions.items():

    print(f"\n{group_name.upper()}")
    print("-" * 70)

    print(
        "Records:",
        f"{condition_records[group_name]:,}"
    )

    for model_name, counts in models.items():

        tp, fp, fn, tn = counts

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        report.append({
            "CONDITION": group_name,
            "MODEL": model_name,
            "RECORDS": condition_records[group_name],
            "PRECISION": precision,
            "RECALL": recall,
            "F1_SCORE": f1,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn
        })

        print(f"\n{model_name}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        print(f"  TP: {tp:,}")
        print(f"  FP: {fp:,}")
        print(f"  FN: {fn:,}")
        print(f"  TN: {tn:,}")

# --------------------------------------------------
# V3 IMPROVEMENT
# --------------------------------------------------

report_df = pd.DataFrame(report)

print("\nV3 IMPROVEMENT BY CONDITION")
print("=" * 70)

for group_name in conditions:

    group = report_df[
        report_df["CONDITION"] == group_name
    ]

    v2_f1 = group.loc[
        group["MODEL"] == "Model V2",
        "F1_SCORE"
    ].iloc[0]

    v3_f1 = group.loc[
        group["MODEL"] == "Model V3",
        "F1_SCORE"
    ].iloc[0]

    print(f"\n{group_name}")
    print(f"V2 F1: {v2_f1:.4f}")
    print(f"V3 F1: {v3_f1:.4f}")
    print(f"Improvement: {(v3_f1 - v2_f1) * 100:+.2f} percentage points")

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

output_folder = "data/processed/model_comparison"

os.makedirs(output_folder, exist_ok=True)

output_path = os.path.join(
    output_folder,
    "v2_v3_dry_day_detection.csv"
)

report_df.to_csv(output_path, index=False)

print("\nResults saved:")
print(output_path)

print("\nDRY-DAY DETECTION ANALYSIS COMPLETED")
print("=" * 70)
