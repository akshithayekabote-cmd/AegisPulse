
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. LOAD FROZEN MODEL V2
# --------------------------------------------------

print("Loading frozen AegisPulse Model V2...")

model = np.load(
    "models/historical_rainfall_model_v2.npz"
)

weights = model["weights"]
bias = float(model["bias"])
mean = model["mean"]
std = model["std"]
threshold = float(model["threshold"])
features = model["features"].astype(str)

print("Model loaded successfully.")
print("Number of features:", len(features))
print("Bias:", bias)
print("Threshold:", threshold)

# --------------------------------------------------
# 2. FEATURE WEIGHT ANALYSIS
# --------------------------------------------------

results = []

for i, feature in enumerate(features):

    results.append({

        "FEATURE": feature,
        "WEIGHT": weights[i],
        "ABSOLUTE_WEIGHT": abs(weights[i]),
        "MEAN": mean[i],
        "STANDARD_DEVIATION": std[i]

    })

summary = pd.DataFrame(results)

summary = summary.sort_values(
    "ABSOLUTE_WEIGHT",
    ascending=False
)

print("\n" + "=" * 90)
print("MODEL V2 FEATURE WEIGHTS")
print("=" * 90)

print(
    summary.to_string(
        index=False,
        formatters={
            "WEIGHT": "{:.6f}".format,
            "ABSOLUTE_WEIGHT": "{:.6f}".format,
            "MEAN": "{:.6f}".format,
            "STANDARD_DEVIATION": "{:.6f}".format
        }
    )
)

# --------------------------------------------------
# 3. RAINFALL FEATURE GROUP ANALYSIS
# --------------------------------------------------

rainfall_features = [
    "RAINFALL",
    "RAIN_LAG_1",
    "RAIN_LAG_3",
    "RAIN_LAG_7",
    "RAIN_ROLLING_3",
    "RAIN_ROLLING_7",
    "RAIN_TODAY"
]

rainfall_summary = summary[
    summary["FEATURE"].isin(rainfall_features)
].copy()

rainfall_weight_sum = (
    rainfall_summary["ABSOLUTE_WEIGHT"].sum()
)

total_weight_sum = (
    summary["ABSOLUTE_WEIGHT"].sum()
)

rainfall_weight_percentage = (
    rainfall_weight_sum / total_weight_sum * 100
    if total_weight_sum > 0
    else 0
)

print("\n" + "=" * 90)
print("RAINFALL FEATURE IMPORTANCE")
print("=" * 90)

print(
    rainfall_summary[
        ["FEATURE", "WEIGHT", "ABSOLUTE_WEIGHT"]
    ].to_string(
        index=False,
        formatters={
            "WEIGHT": "{:.6f}".format,
            "ABSOLUTE_WEIGHT": "{:.6f}".format
        }
    )
)

print(
    "\nRainfall feature absolute-weight share:",
    f"{rainfall_weight_percentage:.2f}%"
)

# --------------------------------------------------
# 4. POSITIVE AND NEGATIVE WEIGHTS
# --------------------------------------------------

print("\n" + "=" * 90)
print("WEIGHT DIRECTION")
print("=" * 90)

positive = summary[
    summary["WEIGHT"] > 0
]

negative = summary[
    summary["WEIGHT"] < 0
]

print("\nPositive-weight features:")

print(
    positive[
        ["FEATURE", "WEIGHT"]
    ].to_string(index=False)
)

print("\nNegative-weight features:")

print(
    negative[
        ["FEATURE", "WEIGHT"]
    ].to_string(index=False)
)

# --------------------------------------------------
# 5. STANDARDIZED CONTRIBUTION PER FEATURE
# --------------------------------------------------

contribution = pd.DataFrame({

    "FEATURE": features,
    "WEIGHT": weights,
    "MEAN": mean,
    "STANDARD_DEVIATION": std,

    "WEIGHT_TIMES_STD": (
        weights * std
    )

})

contribution["ABSOLUTE_WEIGHT_TIMES_STD"] = (
    contribution["WEIGHT_TIMES_STD"].abs()
)

contribution = contribution.sort_values(
    "ABSOLUTE_WEIGHT_TIMES_STD",
    ascending=False
)

print("\n" + "=" * 90)
print("WEIGHT SCALED BY FEATURE STANDARD DEVIATION")
print("=" * 90)

print(
    contribution.to_string(
        index=False,
        formatters={
            "WEIGHT": "{:.6f}".format,
            "MEAN": "{:.6f}".format,
            "STANDARD_DEVIATION": "{:.6f}".format,
            "WEIGHT_TIMES_STD": "{:.6f}".format,
            "ABSOLUTE_WEIGHT_TIMES_STD": "{:.6f}".format
        }
    )
)

# --------------------------------------------------
# 6. SAVE RESULTS
# --------------------------------------------------

summary.to_csv(
    "data/processed/model_v2_feature_weights.csv",
    index=False
)

contribution.to_csv(
    "data/processed/model_v2_scaled_feature_weights.csv",
    index=False
)

print("\nSaved feature-weight analysis files.")

print(
    "data/processed/model_v2_feature_weights.csv"
)

print(
    "data/processed/model_v2_scaled_feature_weights.csv"
)

print("\nFEATURE WEIGHT ANALYSIS COMPLETED.")
