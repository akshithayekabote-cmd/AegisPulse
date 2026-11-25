
import numpy as np
import pandas as pd
import pickle
import os

# Load training and testing datasets
train = pd.read_csv("data/processed/rainfall_occurrence_train.csv")
test = pd.read_csv("data/processed/rainfall_occurrence_test.csv")

# Select input features
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

X_test = test[features].to_numpy(dtype=np.float64)
y_test = test["RAIN_OCCURRED"].to_numpy(dtype=np.int32)

# Standardize features using training data only
means = X_train.mean(axis=0)
stds = X_train.std(axis=0)
stds[stds == 0] = 1

X_train = (X_train - means) / stds
X_test = (X_test - means) / stds

# Add intercept
X_train = np.column_stack([np.ones(len(X_train)), X_train])
X_test = np.column_stack([np.ones(len(X_test)), X_test])

# Calculate class weights
positive_count = np.sum(y_train == 1)
negative_count = np.sum(y_train == 0)

positive_weight = len(y_train) / (2 * positive_count)
negative_weight = len(y_train) / (2 * negative_count)

weights = np.where(y_train == 1, positive_weight, negative_weight)

# Sigmoid function
def sigmoid(z):
    z = np.clip(z, -35, 35)
    return 1 / (1 + np.exp(-z))

# Train logistic regression using mini-batch gradient descent
print("Training rainfall occurrence model...")

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
        w_batch = weights[batch_indices]

        probabilities = sigmoid(X_batch @ coefficients)

        errors = (probabilities - y_batch) * w_batch

        gradient = (X_batch.T @ errors) / len(batch_indices)
        gradient[1:] += regularization * coefficients[1:]

        coefficients -= learning_rate * gradient

    print(f"Epoch {epoch + 1}/{epochs} completed")

# Predict rainfall occurrence
probabilities = sigmoid(X_test @ coefficients)
predictions = (probabilities >= 0.5).astype(int)

# Calculate evaluation metrics
tp = np.sum((predictions == 1) & (y_test == 1))
tn = np.sum((predictions == 0) & (y_test == 0))
fp = np.sum((predictions == 1) & (y_test == 0))
fn = np.sum((predictions == 0) & (y_test == 1))

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

accuracy = (tp + tn) / len(y_test)

print("\nAEGISPULSE - RAINFALL OCCURRENCE MODEL")
print("=" * 50)

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

print("\nConfusion Matrix")
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

# Save model parameters
os.makedirs("models", exist_ok=True)

model_data = {
    "coefficients": coefficients,
    "means": means,
    "stds": stds,
    "features": features
}

with open("models/rainfall_occurrence_model.pkl", "wb") as file:
    pickle.dump(model_data, file)

print("\nModel saved successfully!")
print("Location: models/rainfall_occurrence_model.pkl")
