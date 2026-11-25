
import numpy as np
import pandas as pd

# Load official IMD subdivision mapping
mapping = pd.read_csv(
    "data/processed/india_grid_subdivision_mapping.csv"
)

mapping = mapping[
    mapping["STATUS"] == "MAPPED"
]

subdivision_lookup = {
    (float(row.LATITUDE), float(row.LONGITUDE)): row.SUBDIVISION
    for row in mapping.itertuples()
}

# Load frozen Model V2
model = np.load(
    "models/historical_rainfall_model_v2.npz"
)

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

target_subdivisions = [
    "JAMMU & KASHMIR",
    "PUNJAB",
    "GANGETIC WEST BENGAL",
    "CHHATTISGARH",
    "ORISSA",
    "HIMACHAL PRADESH",
    "JHARKHAND",
    "MARATHWADA"
]

# Initialize error counters
results = {}

for subdivision in target_subdivisions:
    for month in range(1, 13):
        results[(subdivision, month)] = {
            "records": 0,
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "TN": 0
        }

total_records = 0
unmapped_records = 0

# Process the 2024 test dataset
for chunk in pd.read_csv(
    "data/processed/features/test.csv.gz",
    chunksize=50000
):

    chunk["RAIN_TODAY"] = (
        chunk["RAINFALL"] > 0
    ).astype(int)

    actual = chunk["RAIN_OCCURRED"].to_numpy(dtype=int)

    # Prepare Model V2 features
    X = chunk[features].to_numpy(dtype=float)

    # Standardize using saved training statistics
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

    months = chunk["MONTH"].to_numpy(dtype=int)

    assigned = np.array([
        subdivision_lookup.get(
            (float(lat), float(lon)), ""
        )
        for lat, lon in zip(
            chunk["LATITUDE"],
            chunk["LONGITUDE"]
        )
    ])

    total_records += len(chunk)
    unmapped_records += int((assigned == "").sum())

    # Calculate errors for each subdivision and month
    for subdivision in target_subdivisions:

        region_mask = assigned == subdivision

        if not region_mask.any():
            continue

        for month in range(1, 13):

            mask = region_mask & (months == month)

            if not mask.any():
                continue

            y_true = actual[mask]
            y_pred = predictions[mask]

            tp = int(
                ((y_true == 1) & (y_pred == 1)).sum()
            )

            fp = int(
                ((y_true == 0) & (y_pred == 1)).sum()
            )

            fn = int(
                ((y_true == 1) & (y_pred == 0)).sum()
            )

            tn = int(
                ((y_true == 0) & (y_pred == 0)).sum()
            )

            r = results[(subdivision, month)]

            r["records"] += len(y_true)
            r["TP"] += tp
            r["FP"] += fp
            r["FN"] += fn
            r["TN"] += tn

# Prepare final report
rows = []

for (subdivision, month), r in results.items():

    records = r["records"]
    tp = r["TP"]
    fp = r["FP"]
    fn = r["FN"]
    tn = r["TN"]

    precision = (
        tp / (tp + fp)
        if tp + fp else np.nan
    )

    recall = (
        tp / (tp + fn)
        if tp + fn else np.nan
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if pd.notna(precision)
        and pd.notna(recall)
        and precision + recall > 0
        else 0 if pd.notna(precision)
        and pd.notna(recall)
        else np.nan
    )

    error_rate = (
        (fp + fn) / records * 100
        if records else np.nan
    )

    false_positive_rate = (
        fp / (fp + tn) * 100
        if fp + tn else np.nan
    )

    false_negative_rate = (
        fn / (fn + tp) * 100
        if fn + tp else np.nan
    )

    rows.append({
        "SUBDIVISION": subdivision,
        "MONTH": month,
        "RECORDS": records,
        "TRUE_POSITIVES": tp,
        "FALSE_POSITIVES": fp,
        "FALSE_NEGATIVES": fn,
        "TRUE_NEGATIVES": tn,
        "PRECISION": precision,
        "RECALL": recall,
        "F1_SCORE": f1,
        "ERROR_RATE_PERCENT": error_rate,
        "FALSE_POSITIVE_RATE_PERCENT": false_positive_rate,
        "FALSE_NEGATIVE_RATE_PERCENT": false_negative_rate
    })

summary = pd.DataFrame(rows)

summary = summary.sort_values(
    ["SUBDIVISION", "MONTH"]
)

# Save detailed results
output_path = (
    "data/processed/regional_monthly_error_patterns.csv"
)

summary.to_csv(output_path, index=False)

# Display results
print("\nAEGISPULSE - REGIONAL ERROR PATTERN ANALYSIS")
print("=" * 110)

print("Model threshold:", threshold)
print("Model bias:", bias)
print("Test year: 2024")
print("Subdivisions analyzed:", len(target_subdivisions))

print("\nDATA SUMMARY")
print("=" * 75)
print("Total test records:", total_records)
print("Unmapped records:", unmapped_records)

for subdivision in target_subdivisions:

    regional = summary[
        summary["SUBDIVISION"] == subdivision
    ]

    print(f"\n{subdivision}")
    print("-" * 110)

    display_columns = [
        "MONTH",
        "TRUE_POSITIVES",
        "FALSE_POSITIVES",
        "FALSE_NEGATIVES",
        "TRUE_NEGATIVES",
        "PRECISION",
        "RECALL",
        "ERROR_RATE_PERCENT"
    ]

    print(
        regional[display_columns].to_string(
            index=False,
            formatters={
                "PRECISION": lambda x: (
                    f"{x:.3f}" if pd.notna(x) else "N/A"
                ),
                "RECALL": lambda x: (
                    f"{x:.3f}" if pd.notna(x) else "N/A"
                ),
                "ERROR_RATE_PERCENT": lambda x: (
                    f"{x:.2f}%" if pd.notna(x) else "N/A"
                )
            }
        )
    )

# Overall error totals for each subdivision
print("\nREGIONAL ERROR SUMMARY")
print("=" * 110)

regional_totals = summary.groupby("SUBDIVISION")[
    [
        "RECORDS",
        "TRUE_POSITIVES",
        "FALSE_POSITIVES",
        "FALSE_NEGATIVES",
        "TRUE_NEGATIVES"
    ]
].sum()

regional_totals["TOTAL_ERRORS"] = (
    regional_totals["FALSE_POSITIVES"]
    + regional_totals["FALSE_NEGATIVES"]
)

regional_totals["ERROR_RATE_PERCENT"] = (
    regional_totals["TOTAL_ERRORS"]
    / regional_totals["RECORDS"] * 100
)

print(regional_totals.round(2).to_string())

print("\nSaved detailed results to:")
print(output_path)

print("\nREGIONAL ERROR PATTERN ANALYSIS COMPLETED")
