
import numpy as np
import pandas as pd
import os
from itertools import zip_longest

print("AEGISPULSE - V2 vs V3 REGIONAL PERFORMANCE")
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
# LOAD REGIONAL MAPPING
# --------------------------------------------------

print("\nLoading IMD subdivision mapping...")

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
print(
    "Subdivisions:",
    mapping["SUBDIVISION"].nunique()
)

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

if len(v2_weights) != len(V2_FEATURES):
    raise ValueError("V2 feature count mismatch.")

if len(v3_weights) != len(V3_FEATURES):
    raise ValueError("V3 feature count mismatch.")

# --------------------------------------------------
# METRIC STORAGE
# --------------------------------------------------

results = {}

mapped_records = 0
unmapped_records = 0
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

print("\nAnalyzing 2024 test records...")
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
        raise ValueError("V2 and V3 datasets have different lengths.")

    if len(chunk_v2) != len(chunk_v3):
        raise ValueError("Chunk record counts do not match.")

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
    # DERIVE RAIN_TODAY
    # --------------------------------------------------

    rain_today = (
        chunk_v2["RAINFALL"].to_numpy() > 0
    )

    chunk_v2["RAIN_TODAY"] = rain_today.astype(np.float64)
    chunk_v3["RAIN_TODAY"] = rain_today.astype(np.float64)

    y_true = chunk_v2["RAIN_OCCURRED"].to_numpy(
        dtype=np.int8
    )

    # --------------------------------------------------
    # PREDICT WITH V2
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
    # PREDICT WITH V3
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
    # MAP GRID LOCATIONS TO SUBDIVISIONS
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
    total_records += len(chunk_v2)

    # --------------------------------------------------
    # ACCUMULATE REGIONAL COUNTS
    # --------------------------------------------------

    for region in np.unique(subdivisions[mapped_mask]):

        mask = subdivisions == region

        actual = y_true[mask]

        region_predictions = {
            "Model V2": predictions_v2[mask],
            "Model V3": predictions_v3[mask]
        }

        if region not in results:
            results[region] = {
                "Model V2": [0, 0, 0, 0],
                "Model V3": [0, 0, 0, 0],
                "RECORDS": 0
            }

        results[region]["RECORDS"] += int(np.sum(mask))

        for model_name, predicted in region_predictions.items():

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
# REGIONAL METRICS
# --------------------------------------------------

print("\nREGIONAL PERFORMANCE RESULTS")
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

        accuracy = (
            (tp + tn) / (tp + fp + fn + tn)
        )

        report.append({
            "SUBDIVISION": region,
            "MODEL": model_name,
            "RECORDS": region_data["RECORDS"],
            "ACCURACY": accuracy,
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
# V2 VS V3 REGIONAL COMPARISON
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

print("\nV3 REGIONAL F1 IMPROVEMENT")
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

improved = int(
    (pivot["F1_IMPROVEMENT"] > 0).sum()
)

declined = int(
    (pivot["F1_IMPROVEMENT"] < 0).sum()
)

unchanged = int(
    (pivot["F1_IMPROVEMENT"] == 0).sum()
)

print("\nREGIONAL SUMMARY")
print("-" * 75)

print("Subdivisions analyzed:", len(pivot))
print("Improved with V3:", improved)
print("Declined with V3:", declined)
print("Unchanged:", unchanged)

print("\nMapped records:", f"{mapped_records:,}")
print("Unmapped records:", f"{unmapped_records:,}")
print("Total records:", f"{total_records:,}")

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

output_folder = "data/processed/model_comparison"

os.makedirs(output_folder, exist_ok=True)

report_path = os.path.join(
    output_folder,
    "v2_v3_regional_test_metrics.csv"
)

improvement_path = os.path.join(
    output_folder,
    "v2_v3_regional_f1_improvement.csv"
)

report_df.to_csv(report_path, index=False)
pivot.to_csv(improvement_path)

print("\nFiles saved:")
print(report_path)
print(improvement_path)

print("\nREGIONAL ANALYSIS COMPLETED")
print("=" * 75)
