
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

VALIDATION_PATH = (
    "data/processed/features/validation.csv.gz"
)

OUTPUT_PATH = (
    "data/processed/"
    "validation_conditioned_performance.csv"
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

CONDITIONS = {
    "PREVIOUS_DAY": "RAIN_TODAY",
    "PREVIOUS_3_DAYS": "RAIN_ROLLING_3",
    "PREVIOUS_7_DAYS": "RAIN_ROLLING_7"
}

# --------------------------------------------------
# 2. LOAD FROZEN MODEL V2
# --------------------------------------------------

print("Loading frozen Model V2...")

model = np.load(
    "models/historical_rainfall_model_v2.npz"
)

weights = model["weights"]
bias = float(model["bias"])
mean = model["mean"]
std = model["std"]
threshold = float(model["threshold"])

print("Threshold:", threshold)
print("Bias:", bias)

# --------------------------------------------------
# 3. INITIALIZE CONFUSION COUNTS
# --------------------------------------------------

results = {}

for condition in CONDITIONS:

    for state in ["DRY", "WET"]:

        results[
            (condition, state)
        ] = {
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "TN": 0
        }

total_records = 0

# --------------------------------------------------
# 4. PROCESS VALIDATION DATA
# --------------------------------------------------

print("\nProcessing validation records...")

for chunk in pd.read_csv(
    VALIDATION_PATH,
    chunksize=50000
):

    total_records += len(chunk)

    # Reconstruct binary feature
    chunk["RAIN_TODAY"] = (
        chunk["RAINFALL"] > 0
    ).astype(int)

    # Generate frozen Model V2 predictions
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

    predictions = (
        probabilities >= threshold
    ).astype(int)

    actual = chunk[
        "RAIN_OCCURRED"
    ].to_numpy(dtype=int)

    # Analyze each rainfall-history condition
    for condition, column in CONDITIONS.items():

        rainfall_history = chunk[
            column
        ].to_numpy(dtype=float)

        states = np.where(
            rainfall_history == 0,
            "DRY",
            "WET"
        )

        for state in ["DRY", "WET"]:

            selected = (
                states == state
            )

            if not selected.any():
                continue

            actual_selected = actual[selected]
            predicted_selected = predictions[selected]

            tp = int(np.sum(
                (actual_selected == 1) &
                (predicted_selected == 1)
            ))

            fp = int(np.sum(
                (actual_selected == 0) &
                (predicted_selected == 1)
            ))

            fn = int(np.sum(
                (actual_selected == 1) &
                (predicted_selected == 0)
            ))

            tn = int(np.sum(
                (actual_selected == 0) &
                (predicted_selected == 0)
            ))

            key = (condition, state)

            results[key]["TP"] += tp
            results[key]["FP"] += fp
            results[key]["FN"] += fn
            results[key]["TN"] += tn

# --------------------------------------------------
# 5. CALCULATE PERFORMANCE METRICS
# --------------------------------------------------

rows = []

for key, counts in results.items():

    condition, state = key

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
        2 * precision * recall /
        (precision + recall)
        if (
            pd.notna(precision) and
            pd.notna(recall) and
            precision + recall > 0
        )
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else np.nan
    )

    accuracy = (
        (tp + tn) / (tp + fp + fn + tn)
        if tp + fp + fn + tn > 0
        else np.nan
    )

    rows.append({

        "CONDITION": condition,
        "PREVIOUS_STATE": state,

        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,

        "PRECISION": precision,
        "RECALL": recall,
        "F1_SCORE": f1,
        "SPECIFICITY": specificity,
        "ACCURACY": accuracy,

        "ACTUAL_RAINY_EVENTS": tp + fn,
        "PREDICTED_RAINY_EVENTS": tp + fp

    })

summary = pd.DataFrame(rows)

summary.to_csv(
    OUTPUT_PATH,
    index=False
)

# --------------------------------------------------
# 6. DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 120)
print("VALIDATION CONDITIONED MODEL PERFORMANCE")
print("=" * 120)

print(
    summary.to_string(
        index=False,
        formatters={
            "PRECISION": lambda x: f"{x:.4f}" if pd.notna(x) else "N/A",
            "RECALL": lambda x: f"{x:.4f}" if pd.notna(x) else "N/A",
            "F1_SCORE": lambda x: f"{x:.4f}" if pd.notna(x) else "N/A",
            "SPECIFICITY": lambda x: f"{x:.4f}" if pd.notna(x) else "N/A",
            "ACCURACY": lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
        }
    )
)

# --------------------------------------------------
# 7. PROCESSING SUMMARY
# --------------------------------------------------

print("\n" + "=" * 80)
print("PROCESSING SUMMARY")
print("=" * 80)

print("Total validation records:", total_records)

print("\nResults saved to:")
print(OUTPUT_PATH)

print("\nCONDITIONED PERFORMANCE ANALYSIS COMPLETED.")
