
import numpy as np
import pandas as pd

# Load datasets
train = pd.read_csv(
    "data/processed/rainfall_occurrence_train_v2.csv"
)

validation = pd.read_csv(
    "data/processed/rainfall_occurrence_validation.csv"
)

# Calculate monthly rainfall occurrence rates from training data
monthly_rates = train.groupby("MONTH")["RAIN_OCCURRED"].mean()

print("MONTHLY TRAINING RAINFALL RATES")
print("=" * 45)

print((monthly_rates * 100).round(2))

# Predict validation probabilities using training monthly rates
y = validation["RAIN_OCCURRED"].to_numpy(dtype=int)

seasonal_probabilities = (
    validation["MONTH"].map(monthly_rates).to_numpy()
)

# Evaluate seasonal baseline at threshold 0.5
predictions = (seasonal_probabilities >= 0.5).astype(int)

tp = np.sum((predictions == 1) & (y == 1))
tn = np.sum((predictions == 0) & (y == 0))
fp = np.sum((predictions == 1) & (y == 0))
fn = np.sum((predictions == 0) & (y == 1))

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

accuracy = (tp + tn) / len(y)
specificity = tn / (tn + fp) if tn + fp else 0
balanced_accuracy = (recall + specificity) / 2

print("\nSEASONAL BASELINE RESULTS")
print("=" * 45)

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("Balanced Accuracy:", round(balanced_accuracy, 4))

print("\nConfusion Matrix")
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

# Compare actual and predicted monthly rates
print("\nMONTHLY COMPARISON")
print("=" * 45)

monthly_results = validation.groupby("MONTH").agg(
    ACTUAL_RATE=("RAIN_OCCURRED", "mean"),
    OBSERVATIONS=("RAIN_OCCURRED", "count")
)

monthly_results["SEASONAL_PREDICTION"] = monthly_results.index.map(
    monthly_rates
)

print((monthly_results.assign(
    ACTUAL_RATE=monthly_results["ACTUAL_RATE"] * 100,
    SEASONAL_PREDICTION=monthly_results["SEASONAL_PREDICTION"] * 100
)).round(2))

# Save results
monthly_results.to_csv(
    "data/processed/seasonal_baseline_comparison.csv"
)

print("\nSeasonal baseline comparison saved successfully!")
