
import pandas as pd

# Load only the required columns
train = pd.read_csv(
    "data/processed/rainfall_occurrence_train.csv",
    usecols=["MONTH", "RAIN_OCCURRED"]
)

test = pd.read_csv(
    "data/processed/rainfall_occurrence_test.csv",
    usecols=["MONTH", "RAIN_OCCURRED"]
)

# Calculate monthly rainfall occurrence
train_monthly = train.groupby("MONTH")["RAIN_OCCURRED"].mean() * 100
test_monthly = test.groupby("MONTH")["RAIN_OCCURRED"].mean() * 100

# Combine results
comparison = pd.DataFrame({
    "TRAIN_RAINFALL_PERCENT": train_monthly,
    "TEST_RAINFALL_PERCENT": test_monthly
})

print("AEGISPULSE - MONTHLY RAINFALL OCCURRENCE")
print("=" * 55)

print(comparison.round(2).to_string())

print("\nMonthly distribution analysis completed!")
