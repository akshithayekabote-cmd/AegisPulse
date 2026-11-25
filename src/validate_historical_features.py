
import pandas as pd
import os

folder = "data/processed/features"

files = {
    "TRAIN": os.path.join(folder, "train.csv.gz"),
    "VALIDATION": os.path.join(folder, "validation.csv.gz"),
    "TEST": os.path.join(folder, "test.csv.gz")
}

print("AEGISPULSE - FEATURE DATASET VALIDATION")
print("=" * 60)

for name, path in files.items():

    print(f"\n{name} DATASET")
    print("-" * 40)

    df = pd.read_csv(path)

    print("Total records:", len(df))
    print("Total columns:", len(df.columns))
    print("Columns:", list(df.columns))

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nRainfall occurrence distribution:")
    print(df["RAIN_OCCURRED"].value_counts())

    print("\nRainfall occurrence percentage:")
    print(
        df["RAIN_OCCURRED"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nDate feature range:")
    print("Month:", df["MONTH"].min(), "to", df["MONTH"].max())
    print(
        "Day of year:",
        df["DAY_OF_YEAR"].min(),
        "to",
        df["DAY_OF_YEAR"].max()
    )

    print("\nFirst five records:")
    print(df.head())

print("\nVALIDATION COMPLETED")
print("=" * 60)
