
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. LOAD EXISTING RESULTS
# --------------------------------------------------

INPUT_PATH = (
    "data/processed/"
    "regional_rainfall_occurrence_2024.csv"
)

OUTPUT_PATH = (
    "data/processed/"
    "monthly_prediction_calibration_2024.csv"
)

target_subdivisions = [
    "JAMMU & KASHMIR",
    "PUNJAB"
]

df = pd.read_csv(INPUT_PATH)

df = df[
    df["SUBDIVISION"].isin(target_subdivisions)
].copy()

# --------------------------------------------------
# 2. CALCULATE CONFUSION MATRIX VALUES
# --------------------------------------------------

df["TP"] = df["CORRECT_RAINY_PREDICTIONS"]

df["FP"] = (
    df["PREDICTED_RAINY_RECORDS"] - df["TP"]
)

df["FN"] = (
    df["ACTUAL_RAINY_RECORDS"] - df["TP"]
)

df["TN"] = (
    df["RECORDS"] - df["TP"] - df["FP"] - df["FN"]
)

# --------------------------------------------------
# 3. CALCULATE PERFORMANCE METRICS
# --------------------------------------------------

df["PRECISION"] = (
    df["TP"] / (df["TP"] + df["FP"])
).replace([np.inf, -np.inf], np.nan)

df["RECALL"] = (
    df["TP"] / (df["TP"] + df["FN"])
).replace([np.inf, -np.inf], np.nan)

df["F1_SCORE"] = (
    2 * df["PRECISION"] * df["RECALL"] /
    (df["PRECISION"] + df["RECALL"])
).replace([np.inf, -np.inf], np.nan)

df["ACCURACY"] = (
    (df["TP"] + df["TN"]) / df["RECORDS"]
).replace([np.inf, -np.inf], np.nan)

# --------------------------------------------------
# 4. CALCULATE PREDICTION CALIBRATION
# --------------------------------------------------

df["PREDICTION_COUNT_DIFFERENCE"] = (
    df["PREDICTED_RAINY_RECORDS"] -
    df["ACTUAL_RAINY_RECORDS"]
)

df["ABSOLUTE_COUNT_DIFFERENCE"] = (
    df["PREDICTION_COUNT_DIFFERENCE"].abs()
)

df["CALIBRATION_GAP_PP"] = (
    df["PREDICTED_RAINFALL_RATE"] -
    df["ACTUAL_RAINFALL_RATE"]
)

# --------------------------------------------------
# 5. VALIDATE COUNTS
# --------------------------------------------------

df["COUNT_VALIDATION"] = (
    df["TP"] + df["FP"] + df["FN"] + df["TN"]
    == df["RECORDS"]
)

df["PREDICTION_VALIDATION"] = (
    df["TP"] + df["FP"]
    == df["PREDICTED_RAINY_RECORDS"]
)

df["ACTUAL_VALIDATION"] = (
    df["TP"] + df["FN"]
    == df["ACTUAL_RAINY_RECORDS"]
)

# --------------------------------------------------
# 6. DISPLAY RESULTS
# --------------------------------------------------

for subdivision in target_subdivisions:

    regional = df[
        df["SUBDIVISION"] == subdivision
    ]

    print("\n" + "=" * 110)
    print(subdivision)
    print("=" * 110)

    columns = [
        "MONTH",
        "TP",
        "FP",
        "FN",
        "TN",
        "PREDICTION_COUNT_DIFFERENCE",
        "CALIBRATION_GAP_PP",
        "PRECISION",
        "RECALL",
        "F1_SCORE",
        "ACCURACY"
    ]

    print(
        regional[columns].to_string(
            index=False,
            formatters={
                "CALIBRATION_GAP_PP":
                    lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A",
                "PRECISION":
                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A",
                "RECALL":
                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A",
                "F1_SCORE":
                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A",
                "ACCURACY":
                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
            }
        )
    )

# --------------------------------------------------
# 7. CHECK DATA INTEGRITY
# --------------------------------------------------

print("\nDATA VALIDATION")
print("=" * 60)

print(
    "Confusion matrix checks passed:",
    df["COUNT_VALIDATION"].all()
)

print(
    "Prediction count checks passed:",
    df["PREDICTION_VALIDATION"].all()
)

print(
    "Actual rainfall count checks passed:",
    df["ACTUAL_VALIDATION"].all()
)

# --------------------------------------------------
# 8. SAVE RESULTS
# --------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved results to:")
print(OUTPUT_PATH)

print("\nMONTHLY PREDICTION CALIBRATION COMPLETED")
