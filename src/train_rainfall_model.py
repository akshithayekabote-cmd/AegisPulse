
import os
import joblib
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define input features and target
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

# Load training and testing data
train_data = pd.read_csv("data/processed/rainfall_train.csv")
test_data = pd.read_csv("data/processed/rainfall_test.csv")

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Check missing values
if X_train.isnull().any().any() or X_test.isnull().any().any():
    raise ValueError("Missing values found in input features!")

if y_train.isnull().any() or y_test.isnull().any():
    raise ValueError("Missing values found in prediction target!")

# Train the machine learning model
print("\nTraining rainfall prediction model...")

model = HistGradientBoostingRegressor(
    max_iter=50,
    max_leaf_nodes=31,
    learning_rate=0.1,
    random_state=42,
    early_stopping=False
)

model.fit(X_train, y_train)

print("Model training completed!")

# Generate predictions
predictions = model.predict(X_test)

# Rainfall cannot be negative
predictions = predictions.clip(min=0)

# Evaluate model performance
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

# Baseline: tomorrow's rainfall equals today's rainfall
baseline_predictions = X_test["RAINFALL"]

baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_rmse = mean_squared_error(y_test, baseline_predictions) ** 0.5

print("\nAEGISPULSE - MODEL EVALUATION")
print("=" * 45)

print("\nMachine Learning Model:")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

print("\nBaseline Model:")
print("MAE:", round(baseline_mae, 4))
print("RMSE:", round(baseline_rmse, 4))

# Save trained model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/rainfall_prediction_model.pkl")

print("\nModel saved successfully!")
print("Location: models/rainfall_prediction_model.pkl")
