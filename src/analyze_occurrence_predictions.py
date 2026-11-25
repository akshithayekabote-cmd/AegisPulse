
import numpy as np
import pandas as pd
import pickle

# Load test data
test = pd.read_csv("data/processed/rainfall_occurrence_test.csv")

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

X = test[features].to_numpy(dtype=np.float64)
y = test["RAIN_OCCURRED"].to_numpy()

# Load trained model
with open("models/rainfall_occurrence_model.pkl", "rb") as file:
    model = pickle.load(file)

# Standardize test data
X = (X - model["means"]) / model["stds"]

# Add intercept
X = np.column_stack([np.ones(len(X)), X])

# Calculate probabilities
coefficients = model["coefficients"]

z = np.clip(X @ coefficients, -35, 35)
probabilities = 1 / (1 + np.exp(-z))
# Calculate natural class proportions from training data
train = pd.read_csv(
    "data/processed/rainfall_occurrence_train.csv",
    usecols=["RAIN_OCCURRED"]
)

positive_count = (train["RAIN_OCCURRED"] == 1).sum()
negative_count = (train["RAIN_OCCURRED"] == 0).sum()

positive_weight = len(train) / (2 * positive_count)
negative_weight = len(train) / (2 * negative_count)

# Correct probabilities for the class weighting
corrected_probabilities = (
    negative_weight * probabilities
) / (
    positive_weight * (1 - probabilities)
    + negative_weight * probabilities
)

print("\nCORRECTED PROBABILITY STATISTICS")
print("=" * 45)
print("Minimum:", corrected_probabilities.min())
print("Maximum:", corrected_probabilities.max())
print("Mean:", corrected_probabilities.mean())
print("Median:", np.median(corrected_probabilities))

print("\nPREDICTION PROBABILITY ANALYSIS")
print("=" * 45)

print("\nProbability Statistics:")
print("Minimum:", probabilities.min())
print("Maximum:", probabilities.max())
print("Mean:", probabilities.mean())
print("Median:", np.median(probabilities))

print("\nPredictions at Different Thresholds:")

for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    predictions = (corrected_probabilities >= threshold).astype(int)

    positive_predictions = predictions.sum()

    tp = np.sum((predictions == 1) & (y == 1))
    fp = np.sum((predictions == 1) & (y == 0))

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / np.sum(y == 1)

    print(
        f"Threshold: {threshold:.1f} | "
        f"Predicted Rain: {positive_predictions} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f}"
    )


print("\nPROBABILITY DISTRIBUTION BY ACTUAL CLASS")
print("=" * 50)

for label, name in [(0, "DRY DAYS"), (1, "RAINY DAYS")]:
    values = corrected_probabilities[y == label]

    print(f"\n{name}")
    print("Count:", len(values))
    print("Mean probability:", round(values.mean(), 4))
    print("Median probability:", round(np.median(values), 4))
    print("Minimum probability:", round(values.min(), 4))
    print("Maximum probability:", round(values.max(), 4))
