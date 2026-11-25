
import pandas as pd
import numpy as np
import os

folder = "data/processed/features_v3"
model_folder = "models"

train_path = os.path.join(
    folder,
    "train_v3.csv.gz"
)

validation_path = os.path.join(
    folder,
    "validation_v3.csv.gz"
)

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

SPATIAL_FEATURES = [
    "NEIGHBOR_RAIN_MEAN",
    "NEIGHBOR_RAIN_MAX",
    "NEIGHBOR_RAINY_COUNT",
    "NEIGHBOR_VALID_COUNT"
]

FEATURES = (
    BASE_FEATURES
    + ["RAIN_TODAY"]
    + SPATIAL_FEATURES
)

TARGET = "RAIN_OCCURRED"

CHUNK_SIZE = 50000
EPOCHS = 8
LEARNING_RATE = 0.001
L2 = 0.0001


def prepare_features(chunk):

    base = chunk[
        BASE_FEATURES
    ].to_numpy(dtype=np.float64)

    rain_today = (
        chunk["RAINFALL"].to_numpy() > 0
    ).astype(np.float64)

    spatial = chunk[
        SPATIAL_FEATURES
    ].to_numpy(dtype=np.float64)

    return np.column_stack((
        base,
        rain_today,
        spatial
    ))


print("AEGISPULSE - HISTORICAL RAINFALL MODEL V3")
print("=" * 65)

print("\nMODEL CONFIGURATION")
print("-" * 65)

print("Total features:", len(FEATURES))
print("Training epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("L2 regularization:", L2)

print("\nFeatures:")

for index, feature in enumerate(FEATURES, start=1):
    print(f"{index:2}. {feature}")


# Step 1: Calculate training feature statistics

print("\nCalculating training feature statistics...")

feature_sum = np.zeros(
    len(FEATURES),
    dtype=np.float64
)

feature_squared_sum = np.zeros(
    len(FEATURES),
    dtype=np.float64
)

total_rows = 0
positive_count = 0
negative_count = 0

training_columns = (
    BASE_FEATURES
    + SPATIAL_FEATURES
    + [TARGET]
)

for chunk in pd.read_csv(
    train_path,
    usecols=training_columns,
    chunksize=CHUNK_SIZE
):

    x = prepare_features(chunk)

    y = chunk[TARGET].to_numpy()

    feature_sum += x.sum(axis=0)

    feature_squared_sum += (
        x ** 2
    ).sum(axis=0)

    total_rows += len(chunk)

    positive_count += np.sum(y == 1)
    negative_count += np.sum(y == 0)


if positive_count == 0 or negative_count == 0:
    raise ValueError(
        "Training dataset must contain both target classes."
    )

mean = feature_sum / total_rows

variance = (
    feature_squared_sum / total_rows
    - mean ** 2
)

std = np.sqrt(
    np.maximum(variance, 0)
)

std[std < 1e-8] = 1.0

weight_0 = (
    total_rows / (2 * negative_count)
)

weight_1 = (
    total_rows / (2 * positive_count)
)

print("Training records:", f"{total_rows:,}")
print("Positive records:", f"{positive_count:,}")
print("Negative records:", f"{negative_count:,}")
print("Features:", len(FEATURES))
print("Feature statistics calculated.")


# Step 2: Initialize model

weights = np.zeros(
    len(FEATURES),
    dtype=np.float64
)

bias = 0.0

m = np.zeros(
    len(FEATURES),
    dtype=np.float64
)

v = np.zeros(
    len(FEATURES),
    dtype=np.float64
)

m_bias = 0.0
v_bias = 0.0

beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

step = 0


# Step 3: Train model

print("\nStarting Version 3 training...")
print("=" * 65)

for epoch in range(EPOCHS):

    total_loss = 0.0
    processed_rows = 0

    for batch_number, chunk in enumerate(
        pd.read_csv(
            train_path,
            usecols=training_columns,
            chunksize=CHUNK_SIZE
        )
    ):

        chunk = chunk.sample(
            frac=1,
            random_state=epoch * 10000 + batch_number
        )

        x = prepare_features(chunk)

        y = chunk[TARGET].to_numpy(
            dtype=np.float64
        )

        x = (x - mean) / std

        sample_weights = np.where(
            y == 1,
            weight_1,
            weight_0
        )

        z = np.clip(
            x @ weights + bias,
            -35,
            35
        )

        predictions = (
            1 / (1 + np.exp(-z))
        )

        loss = np.mean(
            sample_weights
            * (
                np.logaddexp(0, z)
                - y * z
            )
        )

        total_loss += loss * len(y)

        processed_rows += len(y)

        error = (
            sample_weights
            * (predictions - y)
        )

        gradient_weights = (
            x.T @ error / len(y)
            + L2 * weights
        )

        gradient_bias = np.mean(error)

        step += 1

        m = (
            beta1 * m
            + (1 - beta1) * gradient_weights
        )

        v = (
            beta2 * v
            + (1 - beta2) * gradient_weights ** 2
        )

        m_bias = (
            beta1 * m_bias
            + (1 - beta1) * gradient_bias
        )

        v_bias = (
            beta2 * v_bias
            + (1 - beta2) * gradient_bias ** 2
        )

        m_corrected = (
            m / (1 - beta1 ** step)
        )

        v_corrected = (
            v / (1 - beta2 ** step)
        )

        m_bias_corrected = (
            m_bias / (1 - beta1 ** step)
        )

        v_bias_corrected = (
            v_bias / (1 - beta2 ** step)
        )

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

    average_loss = (
        total_loss / processed_rows
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} completed - "
        f"Loss: {average_loss:.6f}"
    )


# Step 4: Select threshold using validation data

print("\nEvaluating validation dataset...")
print("=" * 65)

thresholds = np.arange(
    0.20,
    0.81,
    0.05
)

tp = np.zeros(
    len(thresholds),
    dtype=np.int64
)

fp = np.zeros(
    len(thresholds),
    dtype=np.int64
)

fn = np.zeros(
    len(thresholds),
    dtype=np.int64
)

validation_columns = (
    BASE_FEATURES
    + SPATIAL_FEATURES
    + [TARGET]
)

for chunk in pd.read_csv(
    validation_path,
    usecols=validation_columns,
    chunksize=CHUNK_SIZE
):

    x = prepare_features(chunk)

    y = chunk[TARGET].to_numpy()

    x = (x - mean) / std

    z = np.clip(
        x @ weights + bias,
        -35,
        35
    )

    probabilities = (
        1 / (1 + np.exp(-z))
    )

    for i, threshold in enumerate(thresholds):

        predicted = (
            probabilities >= threshold
        )

        tp[i] += np.sum(
            (y == 1) & predicted
        )

        fp[i] += np.sum(
            (y == 0) & predicted
        )

        fn[i] += np.sum(
            (y == 1) & ~predicted
        )


precision = (
    tp / np.maximum(tp + fp, 1)
)

recall = (
    tp / np.maximum(tp + fn, 1)
)

f1_scores = (
    2 * precision * recall
    / np.maximum(
        precision + recall,
        1e-12
    )
)

best_index = int(
    np.argmax(f1_scores)
)

best_threshold = float(
    thresholds[best_index]
)

print("\nVALIDATION THRESHOLD RESULTS")
print("-" * 65)

for i, threshold in enumerate(thresholds):

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision[i]:.4f} | "
        f"Recall: {recall[i]:.4f} | "
        f"F1: {f1_scores[i]:.4f}"
    )

print("\nBEST VALIDATION RESULT")
print("-" * 65)

print("Threshold:", round(best_threshold, 2))
print(
    "Validation precision:",
    round(precision[best_index], 4)
)

print(
    "Validation recall:",
    round(recall[best_index], 4)
)

print(
    "Validation F1-score:",
    round(f1_scores[best_index], 4)
)


# Step 5: Save Version 3 model
# Test data is deliberately not accessed here.

model_path = os.path.join(
    model_folder,
    "historical_rainfall_model_v3.npz"
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

print("\nMODEL V3 SAVED")
print("=" * 65)

print("Model path:", model_path)
print("Number of features:", len(FEATURES))
print("Selected threshold:", best_threshold)
print("Bias:", bias)

print("\nVERSION 3 TRAINING COMPLETED")
print("=" * 65)
