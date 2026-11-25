
import numpy as np
import pandas as pd
import pickle

# Load testing data
test_data = pd.read_csv("data/processed/rainfall_test.csv")

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
    "DAY_OF_YEAR"
]

X_test = test_data[features].to_numpy(dtype=np.float64)
y_test = test_data["NEXT_DAY_RAINFALL"].to_numpy(dtype=np.float64)

# Load saved model
with open("models/rainfall_prediction_model.pkl", "rb") as file:
    model = pickle.load(file)

coefficients = model["coefficients"]
means = model["means"]
stds = model["stds"]

# Generate predictions
X_scaled = (X_test - means) / stds
X_scaled = np.column_stack([np.ones(len(X_scaled)), X_scaled])

predictions = X_scaled @ coefficients
predictions = np.maximum(predictions, 0)

# Baseline predictions
baseline = X_test[:, features.index("RAINFALL")]

# Calculate absolute errors
model_errors = np.abs(y_test - predictions)
baseline_errors = np.abs(y_test - baseline)

print("AEGISPULSE - PREDICTION ERROR ANALYSIS")
print("=" * 50)

print("\nActual Rainfall Statistics:")
print(pd.Series(y_test).describe())

print("\nModel Absolute Error:")
print(pd.Series(model_errors).describe())

print("\nBaseline Absolute Error:")
print(pd.Series(baseline_errors).describe())

# Compare performance across rainfall ranges
results = []

ranges = [
    (0, 1),
    (1, 10),
    (10, 50),
    (50, 100),
    (100, float("inf"))
]

for lower, upper in ranges:

    mask = (y_test >= lower) & (y_test < upper)

    if mask.sum() > 0:
        results.append({
            "RAINFALL_RANGE": f"{lower} to {upper}",
            "OBSERVATIONS": int(mask.sum()),
            "MODEL_MAE": model_errors[mask].mean(),
            "BASELINE_MAE": baseline_errors[mask].mean()
        })

print("\nERROR BY ACTUAL RAINFALL RANGE")

print(pd.DataFrame(results).round(4).to_string(index=False))
