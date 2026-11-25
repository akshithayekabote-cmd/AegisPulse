
import numpy as np
import pandas as pd

VALIDATION_PATH = (
    "data/processed/features/validation.csv.gz"
)

OUTPUT_PATH = (
    "data/processed/overall_threshold_sensitivity.csv"
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
    "ALL_RECORDS": None,
    "PREVIOUS_DAY": "RAIN_TODAY",
    "PREVIOUS_3_DAYS": "RAIN_ROLLING_3",
    "PREVIOUS_7_DAYS": "RAIN_ROLLING_7"
}

THRESHOLDS = np.arange(0.05, 0.50, 0.05)

# --------------------------------------------------
# 1. LOAD FROZEN MODEL
# --------------------------------------------------

print("Loading frozen Model V2...")

model = np.load(
    "models/historical_rainfall_model_v2.npz"
)

weights = model["weights"]
bias = float(model["bias"])
mean = model["mean"]
std = model["std"]

print("Original threshold:", float(model["threshold"]))

# --------------------------------------------------
# 2. INITIALIZE COUNTS
# --------------------------------------------------

results = {}

for threshold in THRESHOLDS:

    threshold = round(float(threshold), 2)

    for condition in CONDITIONS:

        states = (
            ["ALL"]
            if condition == "ALL_RECORDS"
            else ["DRY", "WET"]
        )

        for state in states:

            results[
                (threshold, condition, state)
            ] = {
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "TN": 0
            }

total_records = 0

# --------------------------------------------------
# 3. PROCESS VALIDATION DATA
# --------------------------------------------------

print("\nProcessing validation records...")

for chunk in pd.read_csv(
    VALIDATION_PATH,
    chunksize=50000
):

    total_records += len(chunk)

    chunk["RAIN_TODAY"] = (
        chunk["RAINFALL"] > 0
    ).astype(int)

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

    condition_states = {}

    for condition, column in CONDITIONS.items():

        if column is None:
            condition_states[condition] = None
        else:
            values = chunk[
                column
            ].to_numpy(dtype=float)

            condition_states[condition] = np.where(
                values == 0,
                "DRY",
                "WET"
            )

    # Evaluate every threshold
    for threshold in THRESHOLDS:

        threshold = round(float(threshold), 2)

        predictions = (
            probabilities >= threshold
        ).astype(int)

        for condition in CONDITIONS:

            if condition == "ALL_RECORDS":

                groups = {
                    "ALL": np.ones(
                        len(chunk),
                        dtype=bool
                    )
                }

            else:

                states = condition_states[condition]

                groups = {
                    "DRY": states == "DRY",
                    "WET": states == "WET"
                }

            for state, mask in groups.items():

                if not mask.any():
                    continue

                y = actual[mask]
                pred = predictions[mask]

                tp = int(np.sum(
                    (y == 1) & (pred == 1)
                ))

                fp = int(np.sum(
                    (y == 0) & (pred == 1)
                ))

                fn = int(np.sum(
                    (y == 1) & (pred == 0)
                ))

                tn = int(np.sum(
                    (y == 0) & (pred == 0)
                ))

                key = (
                    threshold,
                    condition,
                    state
                )

                results[key]["TP"] += tp
                results[key]["FP"] += fp
                results[key]["FN"] += fn
                results[key]["TN"] += tn

# --------------------------------------------------
# 4. CALCULATE METRICS
# --------------------------------------------------

rows = []

for key, counts in results.items():

    threshold, condition, state = key

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
            pd.notna(precision)
            and pd.notna(recall)
            and precision + recall > 0
        )
        else np.nan
    )

    accuracy = (
        (tp + tn) / (tp + fp + fn + tn)
        if tp + fp + fn + tn > 0
        else np.nan
    )

    rows.append({

        "THRESHOLD": threshold,
        "CONDITION": condition,
        "STATE": state,

        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,

        "PRECISION": precision,
        "RECALL": recall,
        "F1_SCORE": f1,
        "ACCURACY": accuracy,

        "PREDICTED_RAINY_EVENTS": tp + fp

    })

summary = pd.DataFrame(rows)

summary.to_csv(
    OUTPUT_PATH,
    index=False
)

# --------------------------------------------------
# 5. DISPLAY OVERALL PERFORMANCE
# --------------------------------------------------

print("\n" + "=" * 110)
print("OVERALL VALIDATION PERFORMANCE")
print("=" * 110)

overall = summary[
    summary["CONDITION"] == "ALL_RECORDS"
].copy()

print(
    overall[
        [
            "THRESHOLD",
            "TP",
            "FP",
            "FN",
            "PRECISION",
            "RECALL",
            "F1_SCORE",
            "ACCURACY"
        ]
    ].to_string(
        index=False,
        formatters={
            "PRECISION": lambda x: f"{x:.4f}",
            "RECALL": lambda x: f"{x:.4f}",
            "F1_SCORE": lambda x: f"{x:.4f}",
            "ACCURACY": lambda x: f"{x:.4f}"
        }
    )
)

# --------------------------------------------------
# 6. DISPLAY PREVIOUS-DAY PERFORMANCE
# --------------------------------------------------

print("\n" + "=" * 110)
print("PREVIOUS-DAY DRY/WET PERFORMANCE")
print("=" * 110)

previous_day = summary[
    summary["CONDITION"] == "PREVIOUS_DAY"
]

print(
    previous_day[
        [
            "THRESHOLD",
            "STATE",
            "TP",
            "FP",
            "FN",
            "PRECISION",
            "RECALL",
            "F1_SCORE"
        ]
    ].to_string(
        index=False,
        formatters={
            "PRECISION": lambda x: f"{x:.4f}",
            "RECALL": lambda x: f"{x:.4f}",
            "F1_SCORE": lambda x: f"{x:.4f}"
        }
    )
)

# --------------------------------------------------
# 7. COMPLETION
# --------------------------------------------------

print("\nTotal validation records:", total_records)

print("\nResults saved to:")
print(OUTPUT_PATH)

print("\nOVERALL THRESHOLD ANALYSIS COMPLETED.")
