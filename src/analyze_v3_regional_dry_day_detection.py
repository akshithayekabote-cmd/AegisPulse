
import numpy as np
import pandas as pd
import os
from itertools import zip_longest

print("AEGISPULSE - REGIONAL DRY-DAY DETECTION ANALYSIS")
print("=" * 75)

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

mapping_path = "data/processed/india_grid_subdivision_mapping.csv"

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
# LOAD MAPPING
# --------------------------------------------------

print("\nLoading subdivision mapping...")

mapping = pd.read_csv(mapping_path)

mapping = mapping[
    mapping["STATUS"] == "MAPPED"
].copy()

subdivision_lookup = {
    (
        round(float(row.LATITUDE), 4),
        round(float(row.LONGITUDE), 4)
    ): row.SUBDIVISION
    for row in mapping.itertuples()
}

print("Mapped grid locations:", len(subdivision_lookup))

# --------------------------------------------------
# LOAD FROZEN MODELS
# --------------------------------------------------

print("\nLoading frozen models...")

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

print("V2 threshold:", v2_threshold)
print("V3 threshold:", v3_threshold)

# --------------------------------------------------
# REGIONAL METRIC STORAGE
# --------------------------------------------------

results = {}

mapped_records = 0
unmapped_records = 0
dry_records = 0
total_records = 0

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

print("\nProcessing 2024 test data...")
print("=" * 75)

# --------------------------------------------------
# PROCESS DATA
# --------------------------------------------------

for chunk_number, pair in enumerate(
    zip_longest(v2_reader, v3_reader),
    start=1
):

    chunk_v2, chunk_v3 = pair

    if chunk_v2 is None or chunk_v3 is None:
        raise ValueError("Dataset lengths do not match.")

    if len(chunk_v2) != len(chunk_v3):
        raise ValueError("Chunk lengths do not match.")

    for column in [
        "LATITUDE",
        "LONGITUDE",
        "RAIN_OCCURRED"
    ]:

        if not np.array_equal(
            chunk_v2[column].to_numpy(),
            chunk_v3[column].to_numpy()
        ):
            raise ValueError(
                f"Dataset alignment failed: {column}"
            )

    # --------------------------------------------------
    # IDENTIFY PREVIOUS-DAY DRY RECORDS
    # --------------------------------------------------

    rain_today = (
        chunk_v2["RAINFALL"].to_numpy() > 0
    )

    dry_mask = ~rain_today

    y_true = chunk_v2["RAIN_OCCURRED"].to_numpy(
        dtype=np.int8
    )

    chunk_v2["RAIN_TODAY"] = rain_today.astype(np.float64)
    chunk_v3["RAIN_TODAY"] = rain_today.astype(np.float64)

    # --------------------------------------------------
    # V2 PREDICTIONS
    # --------------------------------------------------

    X_v2 = chunk_v2[V2_FEATURES].to_numpy(
        dtype=np.float64
    )

    X_v2 = (
        X_v2 - v2_mean
    ) / (v2_std + 1e-8)

    logits_v2 = X_v2 @ v2_weights + v2_bias

    probabilities_v2 = 1 / (
        1 + np.exp(-np.clip(logits_v2, -500, 500))
    )

    predictions_v2 = (
        probabilities_v2 >= v2_threshold
    )

    # --------------------------------------------------
    # V3 PREDICTIONS
    # --------------------------------------------------

    X_v3 = chunk_v3[V3_FEATURES].to_numpy(
        dtype=np.float64
    )

    X_v3 = (
        X_v3 - v3_mean
    ) / (v3_std + 1e-8)

    logits_v3 = X_v3 @ v3_weights + v3_bias

    probabilities_v3 = 1 / (
        1 + np.exp(-np.clip(logits_v3, -500, 500))
    )

    predictions_v3 = (
        probabilities_v3 >= v3_threshold
    )

    # --------------------------------------------------
    # MAP SUBDIVISIONS
    # --------------------------------------------------

    latitudes = chunk_v2["LATITUDE"].to_numpy()
    longitudes = chunk_v2["LONGITUDE"].to_numpy()

    subdivisions = np.array([
        subdivision_lookup.get(
            (round(float(lat), 4), round(float(lon), 4)),
            "UNMAPPED"
        )
        for lat, lon in zip(latitudes, longitudes)
    ])

    mapped_mask = subdivisions != "UNMAPPED"

    mapped_records += int(np.sum(mapped_mask))
    unmapped_records += int(np.sum(~mapped_mask))
    dry_records += int(np.sum(dry_mask))
    total_records += len(chunk_v2)

    # --------------------------------------------------
    # ACCUMULATE DRY-DAY METRICS
    # --------------------------------------------------

    for region in np.unique(subdivisions[mapped_mask]):

        mask = (subdivisions == region) & dry_mask

        if not np.any(mask):
            continue

        actual = y_true[mask]

        if region not in results:
            results[region] = {
                "RECORDS": 0,
                "Model V2": [0, 0, 0, 0],
                "Model V3": [0, 0, 0, 0]
            }

        results[region]["RECORDS"] += int(np.sum(mask))

        model_predictions = {
            "Model V2": predictions_v2[mask],
            "Model V3": predictions_v3[mask]
        }

        for model_name, predicted in model_predictions.items():

            tp = int(np.sum((actual == 1) & predicted))
            fp = int(np.sum((actual == 0) & predicted))
            fn = int(np.sum((actual == 1) & (~predicted)))
            tn = int(np.sum((actual == 0) & (~predicted)))

            counts = results[region][model_name]

            counts[0] += tp
            counts[1] += fp
            counts[2] += fn
            counts[3] += tn

    if chunk_number % 10 == 0:
        print(f"Processed {total_records:,} records...")

# --------------------------------------------------
# CALCULATE REGIONAL METRICS
# --------------------------------------------------

print("\nREGIONAL DRY-DAY DETECTION RESULTS")
print("=" * 75)

report = []

for region, region_data in results.items():

    for model_name in ["Model V2", "Model V3"]:

        tp, fp, fn, tn = region_data[model_name]

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        report.append({
            "SUBDIVISION": region,
            "MODEL": model_name,
            "DRY_RECORDS": region_data["RECORDS"],
            "PRECISION": precision,
            "RECALL": recall,
            "F1_SCORE": f1,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn
        })

report_df = pd.DataFrame(report)

# --------------------------------------------------
# COMPARE V2 AND V3
# --------------------------------------------------

pivot = report_df.pivot(
    index="SUBDIVISION",
    columns="MODEL",
    values="F1_SCORE"
)

pivot["F1_IMPROVEMENT"] = (
    pivot["Model V3"] - pivot["Model V2"]
)

pivot = pivot.sort_values(
    "F1_IMPROVEMENT",
    ascending=False
)

print("\nREGIONAL DRY-DAY F1 COMPARISON")
print("-" * 75)

print(
    pivot.to_string(
        formatters={
            "Model V2": "{:.4f}".format,
            "Model V3": "{:.4f}".format,
            "F1_IMPROVEMENT": "{:+.4f}".format
        }
    )
)

# --------------------------------------------------
# RECALL COMPARISON
# --------------------------------------------------

recall_pivot = report_df.pivot(
    index="SUBDIVISION",
    columns="MODEL",
    values="RECALL"
)

recall_pivot["RECALL_IMPROVEMENT"] = (
    recall_pivot["Model V3"] - recall_pivot["Model V2"]
)

print("\nREGIONAL DRY-DAY RECALL IMPROVEMENT")
print("-" * 75)

print(
    recall_pivot.sort_values(
        "RECALL_IMPROVEMENT",
        ascending=False
    ).to_string(
        formatters={
            "Model V2": "{:.4f}".format,
            "Model V3": "{:.4f}".format,
            "RECALL_IMPROVEMENT": "{:+.4f}".format
        }
    )
)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

improved = int((pivot["F1_IMPROVEMENT"] > 0).sum())
declined = int((pivot["F1_IMPROVEMENT"] < 0).sum())
unchanged = int((pivot["F1_IMPROVEMENT"] == 0).sum())

print("\nREGIONAL SUMMARY")
print("-" * 75)

print("Subdivisions analyzed:", len(pivot))
print("Improved:", improved)
print("Declined:", declined)
print("Unchanged:", unchanged)

print("Total test records:", f"{total_records:,}")
print("Previous-day dry records:", f"{dry_records:,}")
print("Mapped records:", f"{mapped_records:,}")
print("Unmapped records:", f"{unmapped_records:,}")

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

output_folder = "data/processed/model_comparison"

os.makedirs(output_folder, exist_ok=True)

report_df.to_csv(
    os.path.join(
        output_folder,
        "v2_v3_regional_dry_day_metrics.csv"
    ),
    index=False
)

pivot.to_csv(
    os.path.join(
        output_folder,
        "v2_v3_regional_dry_day_f1_improvement.csv"
    )
)

recall_pivot.to_csv(
    os.path.join(
        output_folder,
        "v2_v3_regional_dry_day_recall_improvement.csv"
    )
)

print("\nResults saved in:")
print(output_folder)

print("\nREGIONAL DRY-DAY ANALYSIS COMPLETED")
print("=" * 75)
