
import numpy as np
import pandas as pd
import os
from itertools import zip_longest

print("AEGISPULSE - MODEL V2 vs V3 VALIDATION COMPARISON")
print("=" * 65)

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

v2_validation_path = "data/processed/features/validation.csv.gz"
v3_validation_path = "data/processed/features_v3/validation_v3.csv.gz"

v2_model_path = "models/historical_rainfall_model_v2.npz"
v3_model_path = "models/historical_rainfall_model_v3.npz"

CHUNK_SIZE = 50000

# --------------------------------------------------
# FEATURE CONFIGURATION
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
# LOAD MODELS
# --------------------------------------------------

print("\nLoading Model V2...")
v2 = np.load(v2_model_path)

v2_weights = v2["weights"]
v2_bias = float(v2["bias"])
v2_mean = v2["mean"]
v2_std = v2["std"]
v2_threshold = float(v2["threshold"])

print("Model V2 loaded.")
print("Features:", len(v2_weights))
print("Threshold:", v2_threshold)

print("\nLoading Model V3...")
v3 = np.load(v3_model_path)

v3_weights = v3["weights"]
v3_bias = float(v3["bias"])
v3_mean = v3["mean"]
v3_std = v3["std"]
v3_threshold = float(v3["threshold"])

print("Model V3 loaded.")
print("Features:", len(v3_weights))
print("Threshold:", v3_threshold)

if len(v2_weights) != len(V2_FEATURES):
    raise ValueError("Model V2 feature count mismatch.")

if len(v3_weights) != len(V3_FEATURES):
    raise ValueError("Model V3 feature count mismatch.")

# --------------------------------------------------
# METRIC STORAGE
# --------------------------------------------------

results = {
    "Persistence Baseline": [0, 0, 0, 0],
    "Model V2": [0, 0, 0, 0],
    "Model V3": [0, 0, 0, 0]
}

total_records = 0

# --------------------------------------------------
# LOAD VALIDATION DATASETS
# --------------------------------------------------

print("\nLoading validation datasets...")

v2_reader = pd.read_csv(
    v2_validation_path,
    chunksize=CHUNK_SIZE
)

v3_reader = pd.read_csv(
    v3_validation_path,
    chunksize=CHUNK_SIZE
)

# --------------------------------------------------
# PROCESS BOTH DATASETS TOGETHER
# --------------------------------------------------

print("\nProcessing validation records...")
print("=" * 65)

for chunk_number, pair in enumerate(
    zip_longest(v2_reader, v3_reader),
    start=1
):

    chunk_v2, chunk_v3 = pair

    if chunk_v2 is None or chunk_v3 is None:
        raise ValueError(
            "V2 and V3 validation datasets have different record counts."
        )

    if len(chunk_v2) != len(chunk_v3):
        raise ValueError(
            f"Chunk {chunk_number}: record count mismatch."
        )

    # --------------------------------------------------
    # VERIFY RECORD ALIGNMENT
    # --------------------------------------------------

    alignment_columns = [
        "LATITUDE",
        "LONGITUDE",
        "RAIN_OCCURRED"
    ]

    for column in alignment_columns:
        if not np.array_equal(
            chunk_v2[column].to_numpy(),
            chunk_v3[column].to_numpy()
        ):
            raise ValueError(
                f"Validation alignment failed for {column} "
                f"in chunk {chunk_number}."
            )

    # Verify target dates when available
    if "DATE" in chunk_v2.columns and "DATE" in chunk_v3.columns:

        if not np.array_equal(
            chunk_v2["DATE"].astype(str).to_numpy(),
            chunk_v3["DATE"].astype(str).to_numpy()
        ):
            raise ValueError(
                f"Date alignment failed in chunk {chunk_number}."
            )

    # --------------------------------------------------
    # DERIVE RAIN_TODAY
    # --------------------------------------------------

    chunk_v2["RAIN_TODAY"] = (
        chunk_v2["RAINFALL"] > 0
    ).astype(np.float64)

    chunk_v3["RAIN_TODAY"] = (
        chunk_v3["RAINFALL"] > 0
    ).astype(np.float64)

    y_true = chunk_v2["RAIN_OCCURRED"].to_numpy(
        dtype=np.int8
    )

    # --------------------------------------------------
    # PERSISTENCE BASELINE
    # --------------------------------------------------

    baseline_predictions = (
        chunk_v2["RAIN_TODAY"].to_numpy() > 0.5
    )

    # --------------------------------------------------
    # MODEL V2 PREDICTIONS
    # --------------------------------------------------

    X_v2 = chunk_v2[V2_FEATURES].to_numpy(
        dtype=np.float64
    )

    X_v2 = (
        X_v2 - v2_mean
    ) / (v2_std + 1e-8)

    v2_logits = X_v2 @ v2_weights + v2_bias

    v2_probabilities = 1 / (
        1 + np.exp(-np.clip(v2_logits, -500, 500))
    )

    v2_predictions = (
        v2_probabilities >= v2_threshold
    )

    # --------------------------------------------------
    # MODEL V3 PREDICTIONS
    # --------------------------------------------------

    X_v3 = chunk_v3[V3_FEATURES].to_numpy(
        dtype=np.float64
    )

    X_v3 = (
        X_v3 - v3_mean
    ) / (v3_std + 1e-8)

    v3_logits = X_v3 @ v3_weights + v3_bias

    v3_probabilities = 1 / (
        1 + np.exp(-np.clip(v3_logits, -500, 500))
    )

    v3_predictions = (
        v3_probabilities >= v3_threshold
    )

    # --------------------------------------------------
    # ACCUMULATE CONFUSION COUNTS
    # --------------------------------------------------

    predictions = {
        "Persistence Baseline": baseline_predictions,
        "Model V2": v2_predictions,
        "Model V3": v3_predictions
    }

    for name, predicted in predictions.items():

        tp = int(np.sum((y_true == 1) & predicted))
        fp = int(np.sum((y_true == 0) & predicted))
        fn = int(np.sum((y_true == 1) & (~predicted)))
        tn = int(np.sum((y_true == 0) & (~predicted)))

        results[name][0] += tp
        results[name][1] += fp
        results[name][2] += fn
        results[name][3] += tn

    total_records += len(chunk_v2)

    if chunk_number % 10 == 0:
        print(
            f"Processed {total_records:,} validation records..."
        )

# --------------------------------------------------
# FINAL METRICS
# --------------------------------------------------

print("\nVALIDATION COMPARISON RESULTS")
print("=" * 65)

print("Total validation records:", f"{total_records:,}")

comparison = []

for name, counts in results.items():

    tp, fp, fn, tn = counts

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    accuracy = (
        (tp + tn) / (tp + fp + fn + tn)
    )

    comparison.append({
        "MODEL": name,
        "ACCURACY": accuracy,
        "PRECISION": precision,
        "RECALL": recall,
        "F1_SCORE": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn
    })

comparison_df = pd.DataFrame(comparison)

print(
    comparison_df[
        [
            "MODEL",
            "ACCURACY",
            "PRECISION",
            "RECALL",
            "F1_SCORE"
        ]
    ].to_string(
        index=False,
        formatters={
            "ACCURACY": "{:.4f}".format,
            "PRECISION": "{:.4f}".format,
            "RECALL": "{:.4f}".format,
            "F1_SCORE": "{:.4f}".format
        }
    )
)

# --------------------------------------------------
# IMPROVEMENT ANALYSIS
# --------------------------------------------------

v2_f1 = comparison_df.loc[
    comparison_df["MODEL"] == "Model V2",
    "F1_SCORE"
].iloc[0]

v3_f1 = comparison_df.loc[
    comparison_df["MODEL"] == "Model V3",
    "F1_SCORE"
].iloc[0]

baseline_f1 = comparison_df.loc[
    comparison_df["MODEL"] == "Persistence Baseline",
    "F1_SCORE"
].iloc[0]

print("\nMODEL V3 IMPROVEMENT")
print("-" * 65)

print(f"V2 F1-score:       {v2_f1:.4f}")
print(f"V3 F1-score:       {v3_f1:.4f}")
print(f"Baseline F1-score: {baseline_f1:.4f}")

print(
    f"V3 vs V2 improvement: "
    f"{(v3_f1 - v2_f1) * 100:.2f} percentage points"
)

print(
    f"V3 vs baseline improvement: "
    f"{(v3_f1 - baseline_f1) * 100:.2f} percentage points"
)

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

os.makedirs(
    "data/processed/model_comparison",
    exist_ok=True
)

output_path = (
    "data/processed/model_comparison/"
    "v2_v3_validation_comparison.csv"
)

comparison_df.to_csv(output_path, index=False)

print("\nComparison saved:")
print(output_path)

print("\nVALIDATION COMPARISON COMPLETED")
print("=" * 65)
