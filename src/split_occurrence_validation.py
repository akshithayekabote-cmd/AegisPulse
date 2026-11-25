
import pandas as pd

input_file = "data/processed/rainfall_occurrence_train.csv"

print("Loading occurrence training data...")

df = pd.read_csv(input_file, parse_dates=["TIME"])

df = df.sort_values(["TIME", "location"])

unique_dates = sorted(df["TIME"].unique())

split_index = int(len(unique_dates) * 0.8)
validation_start = unique_dates[split_index]

train_data = df[df["TIME"] < validation_start].copy()
validation_data = df[df["TIME"] >= validation_start].copy()

train_data.to_csv(
    "data/processed/rainfall_occurrence_train_v2.csv",
    index=False
)

validation_data.to_csv(
    "data/processed/rainfall_occurrence_validation.csv",
    index=False
)

print("\nVALIDATION SPLIT SUMMARY")
print("=" * 45)

print("Training records:", len(train_data))
print("Validation records:", len(validation_data))

print(
    "Training dates:",
    train_data["TIME"].min(),
    "to",
    train_data["TIME"].max()
)

print(
    "Validation dates:",
    validation_data["TIME"].min(),
    "to",
    validation_data["TIME"].max()
)

print("\nRainfall occurrence rates:")
print("Training:", round(train_data["RAIN_OCCURRED"].mean() * 100, 2), "%")
print("Validation:", round(validation_data["RAIN_OCCURRED"].mean() * 100, 2), "%")
