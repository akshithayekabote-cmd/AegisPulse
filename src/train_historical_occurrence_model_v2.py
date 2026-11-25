
import pandas as pd
import numpy as np
import os

folder = "data/processed/features"
model_folder = "models"

train_path = os.path.join(folder, "train.csv.gz")
validation_path = os.path.join(folder, "validation.csv.gz")
test_path = os.path.join(folder, "test.csv.gz")

os.makedirs(model_folder, exist_ok=True)

BASE_FEATURES = [
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

FEATURES = BASE_FEATURES + ["RAIN_TODAY"]
TARGET = "RAIN_OCCURRED"

CHUNK_SIZE = 50000
EPOCHS = 8
LEARNING_RATE = 0.001
L2 = 0.0001


def prepare_features(chunk):

    x = chunk[BASE_FEATURES].to_numpy(dtype=np.float64)

    rain_today = (
        chunk["RAINFALL"].to_numpy() > 0
    ).astype(np.float64)

    return np.column_stack((x, rain_today))


print("AEGISPULSE - HISTORICAL RAINFALL MODEL V2")
print("=" * 60)

# Step 1: Calculate feature statistics
print("\nCalculating feature statistics...")

feature_sum = np.zeros(len(FEATURES))
feature_squared_sum = np.zeros(len(FEATURES))

total_rows = 0
positive_count = 0
negative_count = 0

for chunk in pd.read_csv(
    train_path,
    usecols=BASE_FEATURES + [TARGET],
    chunksize=CHUNK_SIZE
):

    x = prepare_features(chunk)
    y = chunk[TARGET].to_numpy()

    feature_sum += x.sum(axis=0)
    feature_squared_sum += (x ** 2).sum(axis=0)

    total_rows += len(chunk)
    positive_count += np.sum(y == 1)
    negative_count += np.sum(y == 0)

mean = feature_sum / total_rows

variance = (
    feature_squared_sum / total_rows
    - mean ** 2
)

std = np.sqrt(np.maximum(variance, 0))
std[std < 1e-8] = 1.0

weight_0 = total_rows / (2 * negative_count)
weight_1 = total_rows / (2 * positive_count)

print("Training records:", total_rows)
print("Features:", len(FEATURES))
print("Feature statistics calculated.")

# Step 2: Initialize model
weights = np.zeros(len(FEATURES))
bias = 0.0

m = np.zeros(len(FEATURES))
v = np.zeros(len(FEATURES))

m_bias = 0.0
v_bias = 0.0

beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
step = 0

# Step 3: Train model
print("\nStarting Version 2 training...")

for epoch in range(EPOCHS):

    total_loss = 0.0
    processed_rows = 0

    for batch_number, chunk in enumerate(
        pd.read_csv(
            train_path,
            usecols=BASE_FEATURES + [TARGET],
            chunksize=CHUNK_SIZE
        )
    ):

        chunk = chunk.sample(
            frac=1,
            random_state=epoch * 10000 + batch_number
        )

        x = prepare_features(chunk)
        y = chunk[TARGET].to_numpy(dtype=np.float64)

        x = (x - mean) / std

        sample_weights = np.where(
            y == 1,
            weight_1,
            weight_0
        )

        z = np.clip(x @ weights + bias, -35, 35)
        predictions = 1 / (1 + np.exp(-z))

        loss = np.mean(
            sample_weights
            * (np.logaddexp(0, z) - y * z)
        )

        total_loss += loss * len(y)
        processed_rows += len(y)

        error = sample_weights * (predictions - y)

        gradient_weights = (
            x.T @ error / len(y)
            + L2 * weights
        )

        gradient_bias = np.mean(error)

        step += 1

        m = beta1 * m + (1 - beta1) * gradient_weights
        v = beta2 * v + (1 - beta2) * gradient_weights ** 2

        m_bias = beta1 * m_bias + (1 - beta1) * gradient_bias
        v_bias = beta2 * v_bias + (1 - beta2) * gradient_bias ** 2

        m_corrected = m / (1 - beta1 ** step)
        v_corrected = v / (1 - beta2 ** step)

        m_bias_corrected = m_bias / (1 - beta1 ** step)
        v_bias_corrected = v_bias / (1 - beta2 ** step)

        weights -= (
            LEARNING_RATE
            * m_corrected
            / (np.sqrt(v_corrected) + epsilon)
        )

        bias -= (
            LEARNING_RATE
            * m_bias_corrected
            / (np.sqrt(v_bias_corrected) + epsilon)
        )

    average_loss = total_loss / processed_rows

    print(
        f"Epoch {epoch + 1}/{EPOCHS} completed - "
        f"Loss: {average_loss:.6f}"
    )

# Step 4: Select threshold using validation data
print("\nEvaluating validation dataset...")

thresholds = np.arange(0.20, 0.81, 0.05)

tp = np.zeros(len(thresholds), dtype=np.int64)
fp = np.zeros(len(thresholds), dtype=np.int64)
fn = np.zeros(len(thresholds), dtype=np.int64)

for chunk in pd.read_csv(
    validation_path,
    usecols=BASE_FEATURES + [TARGET],
    chunksize=CHUNK_SIZE
):

    x = prepare_features(chunk)
    y = chunk[TARGET].to_numpy()

    x = (x - mean) / std

    z = np.clip(x @ weights + bias, -35, 35)
    probabilities = 1 / (1 + np.exp(-z))

    for i, threshold in enumerate(thresholds):

        predicted = probabilities >= threshold

        tp[i] += np.sum((y == 1) & predicted)
        fp[i] += np.sum((y == 0) & predicted)
        fn[i] += np.sum((y == 1) & ~predicted)

precision = tp / np.maximum(tp + fp, 1)
recall = tp / np.maximum(tp + fn, 1)

f1_scores = (
    2 * precision * recall
    / np.maximum(precision + recall, 1e-12)
)

best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

print("\nBEST VALIDATION THRESHOLD")
print("Threshold:", round(best_threshold, 2))
print("Validation F1-score:", round(f1_scores[best_index], 4))

# Step 5: Evaluate test dataset
print("\nEvaluating test dataset...")

tp_test = 0
tn_test = 0
fp_test = 0
fn_test = 0

for chunk in pd.read_csv(
    test_path,
    usecols=BASE_FEATURES + [TARGET],
    chunksize=CHUNK_SIZE
):

    x = prepare_features(chunk)
    y = chunk[TARGET].to_numpy()

    x = (x - mean) / std

    z = np.clip(x @ weights + bias, -35, 35)
    probabilities = 1 / (1 + np.exp(-z))

    predicted = probabilities >= best_threshold

    tp_test += np.sum((y == 1) & predicted)
    tn_test += np.sum((y == 0) & ~predicted)
    fp_test += np.sum((y == 0) & predicted)
    fn_test += np.sum((y == 1) & ~predicted)

accuracy = (
    (tp_test + tn_test)
    / (tp_test + tn_test + fp_test + fn_test)
)

precision_test = tp_test / max(tp_test + fp_test, 1)
recall_test = tp_test / max(tp_test + fn_test, 1)

f1_test = (
    2 * precision_test * recall_test
    / max(precision_test + recall_test, 1e-12)
)

print("\nHISTORICAL MODEL V2 TEST RESULTS")
print("=" * 60)
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision_test, 4))
print("Recall:", round(recall_test, 4))
print("F1-score:", round(f1_test, 4))

print("\nConfusion Matrix:")
print("True Positives:", tp_test)
print("True Negatives:", tn_test)
print("False Positives:", fp_test)
print("False Negatives:", fn_test)

# Step 6: Save Version 2
model_path = os.path.join(
    model_folder,
    "historical_rainfall_model_v2.npz"
)

np.savez(
    model_path,
    weights=weights,
    bias=bias,
    mean=mean,
    std=std,
    threshold=best_threshold,
    features=np.array(FEATURES)
)

print("\nVersion 2 model saved:", model_path)
print("\nVERSION 2 TRAINING COMPLETED")
print("=" * 60)
