
import numpy as np
import pandas as pd
import pickle

# Load validation dataset
df = pd.read_csv(
    "data/processed/rainfall_occurrence_validation.csv"
)

# Load improved model
with open(
    "models/rainfall_occurrence_model_numpy_v2.pkl",
    "rb"
) as file:
    model = pickle.load(file)

features = model["features"]

X = df[features].to_numpy(dtype=np.float64)
y = df["RAIN_OCCURRED"].to_numpy(dtype=int)

# Standardize features
X = (X - model["means"]) / model["stds"]

# Add intercept
X = np.column_stack([np.ones(len(X)), X])

# Generate model predictions
z = np.clip(X @ model["coefficients"], -35, 35)
probabilities = 1 / (1 + np.exp(-z))

threshold = model["threshold"]
model_predictions = (probabilities >= threshold).astype(int)

# Persistence baseline
baseline_predictions = (
    df["RAINFALL"].to_numpy() > 0
).astype(int)

df["MODEL_PREDICTION"] = model_predictions
df["BASELINE_PREDICTION"] = baseline_predictions

print("\nMONTHLY RAINFALL MODEL COMPARISON")
print("=" * 65)

results = []

for month, group in df.groupby("MONTH"):

    actual = group["RAIN_OCCURRED"].to_numpy()

    print(f"\nMONTH: {month}")

    for name, column in [
        ("Logistic Regression", "MODEL_PREDICTION"),
        ("Persistence Baseline", "BASELINE_PREDICTION")
    ]:

        predicted = group[column].to_numpy()

        tp = np.sum((predicted == 1) & (actual == 1))
        fp = np.sum((predicted == 1) & (actual == 0))
        fn = np.sum((predicted == 0) & (actual == 1))
        tn = np.sum((predicted == 0) & (actual == 0))

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        results.append({
            "MONTH": month,
            "MODEL": name,
            "PRECISION": precision,
            "RECALL": recall,
            "F1": f1,
            "ACCURACY": (tp + tn) / len(actual)
        })

        print(
            f"{name}: "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}, "
            f"F1={f1:.4f}, "
            f"Accuracy={(tp + tn) / len(actual):.4f}"
        )

# Save comparison
results_df = pd.DataFrame(results)

results_df.to_csv(
    "data/processed/monthly_occurrence_model_comparison.csv",
    index=False
)

print("\nMonthly comparison saved successfully!")

print("\nMONTHLY PROBABILITY ANALYSIS")
print("=" * 50)

df["PREDICTED_PROBABILITY"] = probabilities

for month, group in df.groupby("MONTH"):

    print(f"\nMonth: {month}")

    print(
        "Actual rainfall rate:",
        round(group["RAIN_OCCURRED"].mean(), 4)
    )

    print(
        "Average predicted probability:",
        round(group["PREDICTED_PROBABILITY"].mean(), 4)
    )

    print(
        "Average observed rainfall:",
        round(group["RAINFALL"].mean(), 4)
    )
