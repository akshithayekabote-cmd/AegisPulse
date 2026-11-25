
import pandas as pd

# Load the feature-engineered dataset
file_path = "data/processed/rainfall_ml_features_2024.csv"
df = pd.read_csv(file_path, parse_dates=["TIME"])

# Sort records chronologically
df = df.sort_values(["TIME", "location"]).reset_index(drop=True)

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

# Define the prediction target
target = "NEXT_DAY_RAINFALL"

# Split by date, not randomly, to prevent future-data leakage
unique_dates = sorted(df["TIME"].unique())
split_index = int(len(unique_dates) * 0.80)
split_date = unique_dates[split_index]

train_data = df[
    (df["TIME"] + pd.Timedelta(days=1)) < split_date
].copy()

test_data = df[df["TIME"] >= split_date].copy()
# Save training and testing datasets
train_data.to_csv(
    "data/processed/rainfall_train.csv",
    index=False
)

test_data.to_csv(
    "data/processed/rainfall_test.csv",
    index=False
)

# Display dataset information
print("MACHINE LEARNING DATA PREPARATION COMPLETED!")

print("\nTraining records:", len(train_data))
print("Testing records:", len(test_data))

print("\nTraining date range:")
print(train_data["TIME"].min(), "to", train_data["TIME"].max())

print("\nTesting date range:")
print(test_data["TIME"].min(), "to", test_data["TIME"].max())

print("\nInput features:", features)
print("Prediction target:", target)

print("\nTraining and testing datasets saved successfully!")

