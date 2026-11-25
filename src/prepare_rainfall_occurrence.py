
import pandas as pd

# Load existing datasets
train = pd.read_csv("data/processed/rainfall_train.csv")
test = pd.read_csv("data/processed/rainfall_test.csv")

# Create binary rainfall occurrence target
train["RAIN_OCCURRED"] = (
    train["NEXT_DAY_RAINFALL"] > 0
).astype(int)

test["RAIN_OCCURRED"] = (
    test["NEXT_DAY_RAINFALL"] > 0
).astype(int)

# Save occurrence datasets
train.to_csv(
    "data/processed/rainfall_occurrence_train.csv",
    index=False
)

test.to_csv(
    "data/processed/rainfall_occurrence_test.csv",
    index=False
)

# Display distribution
print("AEGISPULSE - RAINFALL OCCURRENCE PREPARATION")
print("=" * 50)

print("\nTraining distribution:")
print(train["RAIN_OCCURRED"].value_counts())

print("\nTesting distribution:")
print(test["RAIN_OCCURRED"].value_counts())

print("\nTraining and testing datasets saved successfully!")
