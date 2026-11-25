
import numpy as np
import pandas as pd
import os
import pickle

# Load datasets
train_data = pd.read_csv("data/processed/rainfall_train.csv")
test_data = pd.read_csv("data/processed/rainfall_test.csv")

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

target = "NEXT_DAY_RAINFALL"

X_train = train_data[features].to_numpy(dtype=np.float64)
y_train = train_data[target].to_numpy(dtype=np.float64)

X_test = test_data[features].to_numpy(dtype=np.float64)
y_test = test_data[target].to_numpy(dtype=np.float64)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Standardize input features using training data only
means = X_train.mean(axis=0)
stds = X_train.std(axis=0)
stds[stds == 0] = 1

X_train_scaled = (X_train - means) / stds
X_test_scaled = (X_test - means) / stds

# Add intercept column
X_train_scaled = np.column_stack([
    np.ones(len(X_train_scaled)),
    X_train_scaled
])

X_test_scaled = np.column_stack([
    np.ones(len(X_test_scaled)),
    X_test_scaled
])

# Train multiple linear regression model
print("\nTraining rainfall prediction model...")

coefficients = np.linalg.lstsq(
    X_train_scaled,
    y_train,
    rcond=None
)[0]

# Generate predictions
predictions = X_test_scaled @ coefficients
predictions = np.maximum(predictions, 0)

# Evaluate model
mae = np.mean(np.abs(y_test - predictions))
rmse = np.sqrt(np.mean((y_test - predictions) ** 2))

ss_total = np.sum((y_test - y_test.mean()) ** 2)
ss_residual = np.sum((y_test - predictions) ** 2)

r2 = 1 - (ss_residual / ss_total)

# Baseline: tomorrow's rainfall equals today's rainfall
baseline = X_test[:, features.index("RAINFALL")]

baseline_mae = np.mean(np.abs(y_test - baseline))
baseline_rmse = np.sqrt(np.mean((y_test - baseline) ** 2))

print("\nAEGISPULSE - MODEL EVALUATION")
print("=" * 45)

print("\nMultiple Linear Regression:")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

print("\nBaseline Model:")
print("MAE:", round(baseline_mae, 4))
print("RMSE:", round(baseline_rmse, 4))

# Save model parameters
os.makedirs("models", exist_ok=True)

model_data = {
    "coefficients": coefficients,
    "means": means,
    "stds": stds,
    "features": features
}

with open("models/rainfall_prediction_model.pkl", "wb") as file:
    pickle.dump(model_data, file)

print("\nModel saved successfully!")
print("Location: models/rainfall_prediction_model.pkl")
