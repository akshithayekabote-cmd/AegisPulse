
import numpy as np
import pandas as pd
import pickle
import os

# Load datasets
train = pd.read_csv(
    "data/processed/rainfall_occurrence_train_v2.csv"
)

validation = pd.read_csv(
    "data/processed/rainfall_occurrence_validation.csv"
)

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

X_train = train[features].to_numpy(dtype=np.float64)
y_train = train["RAIN_OCCURRED"].to_numpy(dtype=np.float64)

X_val = validation[features].to_numpy(dtype=np.float64)
y_val = validation["RAIN_OCCURRED"].to_numpy(dtype=np.int32)

# Standardize using training data only
means = X_train.mean(axis=0)
stds = X_train.std(axis=0)
stds[stds == 0] = 1

X_train = (X_train - means) / stds
X_val = (X_val - means) / stds

# Add intercept
X_train = np.column_stack([np.ones(len(X_train)), X_train])
X_val = np.column_stack([np.ones(len(X_val)), X_val])

# Sigmoid function
def sigmoid(z):
    z = np.clip(z, -35, 35)
    return 1 / (1 + np.exp(-z))

# Train unweighted logistic regression
print("Training improved rainfall occurrence model...")

rng = np.random.default_rng(42)

coefficients = np.zeros(X_train.shape[1])

learning_rate = 0.01
epochs = 10
batch_size = 8192
regularization = 0.001

for epoch in range(epochs):

    indices = rng.permutation(len(y_train))

    for start in range(0, len(indices), batch_size):

        batch_indices = indices[start:start + batch_size]

        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]

        probabilities = sigmoid(X_batch @ coefficients)

        errors = probabilities - y_batch

        gradient = (X_batch.T @ errors) / len(batch_indices)

        gradient[1:] += regularization * coefficients[1:]

        coefficients -= learning_rate * gradient

    print(f"Epoch {epoch + 1}/{epochs} completed")

# Predict validation probabilities
val_probabilities = sigmoid(X_val @ coefficients)

# Evaluate thresholds using validation data
print("\nVALIDATION THRESHOLD ANALYSIS")
print("=" * 55)

best_threshold = 0.5
best_f1 = -1

for threshold in np.arange(0.05, 0.96, 0.05):

    predictions = (val_probabilities >= threshold).astype(int)

    tp = np.sum((predictions == 1) & (y_val == 1))
    tn = np.sum((predictions == 0) & (y_val == 0))
    fp = np.sum((predictions == 1) & (y_val == 0))
    fn = np.sum((predictions == 0) & (y_val == 1))

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\nBEST VALIDATION THRESHOLD:", round(best_threshold, 2))
print("BEST VALIDATION F1:", round(best_f1, 4))

# Evaluate selected threshold
predictions = (val_probabilities >= best_threshold).astype(int)

tp = np.sum((predictions == 1) & (y_val == 1))
tn = np.sum((predictions == 0) & (y_val == 0))
fp = np.sum((predictions == 1) & (y_val == 0))
fn = np.sum((predictions == 0) & (y_val == 1))

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
specificity = tn / (tn + fp) if tn + fp else 0
accuracy = (tp + tn) / len(y_val)
balanced_accuracy = (recall + specificity) / 2

print("\nIMPROVED RAINFALL OCCURRENCE MODEL")
print("=" * 50)

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("Specificity:", round(specificity, 4))
print("F1 Score:", round(best_f1, 4))
print("Balanced Accuracy:", round(balanced_accuracy, 4))

print("\nConfusion Matrix")
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

# Compare with persistence baseline
baseline_predictions = (
    validation["RAINFALL"].to_numpy() > 0
).astype(int)

baseline_tp = np.sum(
    (baseline_predictions == 1) & (y_val == 1)
)

baseline_tn = np.sum(
    (baseline_predictions == 0) & (y_val == 0)
)

baseline_fp = np.sum(
    (baseline_predictions == 1) & (y_val == 0)
)

baseline_fn = np.sum(
    (baseline_predictions == 0) & (y_val == 1)
)

baseline_precision = (
    baseline_tp / (baseline_tp + baseline_fp)
    if baseline_tp + baseline_fp else 0
)

baseline_recall = (
    baseline_tp / (baseline_tp + baseline_fn)
    if baseline_tp + baseline_fn else 0
)

baseline_f1 = (
    2 * baseline_precision * baseline_recall /
    (baseline_precision + baseline_recall)
    if baseline_precision + baseline_recall else 0
)

print("\nPERSISTENCE BASELINE")
print("=" * 50)
print("Precision:", round(baseline_precision, 4))
print("Recall:", round(baseline_recall, 4))
print("F1 Score:", round(baseline_f1, 4))

# Save model
os.makedirs("models", exist_ok=True)

model_data = {
    "coefficients": coefficients,
    "means": means,
    "stds": stds,
    "features": features,
    "threshold": best_threshold
}

with open(
    "models/rainfall_occurrence_model_numpy_v2.pkl",
    "wb"
) as file:
    pickle.dump(model_data, file)

print("\nImproved model saved successfully!")
print("models/rainfall_occurrence_model_numpy_v2.pkl")
