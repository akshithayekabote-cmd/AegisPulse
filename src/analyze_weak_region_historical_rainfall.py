
import numpy as np
import pandas as pd
import xarray as xr

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATA_PATH = (
    "data/processed/"
    "india_rainfall_cleaned_2015_2024.nc"
)

MAPPING_PATH = (
    "data/processed/"
    "india_grid_subdivision_mapping.csv"
)

OUTPUT_PATH = (
    "data/processed/"
    "weak_region_historical_rainfall_comparison.csv"
)

TARGET_SUBDIVISIONS = [
    "JAMMU & KASHMIR",
    "PUNJAB"
]

# --------------------------------------------------
# 2. LOAD SUBDIVISION MAPPING
# --------------------------------------------------

mapping = pd.read_csv(MAPPING_PATH)

mapping = mapping[
    mapping["STATUS"] == "MAPPED"
].copy()

subdivision_lookup = {
    (round(float(row.LATITUDE), 2),
     round(float(row.LONGITUDE), 2)): row.SUBDIVISION
    for row in mapping.itertuples()
}

# --------------------------------------------------
# 3. LOAD CLEANED HISTORICAL RAINFALL
# --------------------------------------------------

print("Loading historical rainfall dataset...")

ds = xr.open_dataset(
    DATA_PATH,
    engine="scipy"
)

rainfall = ds["RAINFALL"].values

dates = pd.to_datetime(
    ds["TIME"].values
)

latitudes = ds["LATITUDE"].values
longitudes = ds["LONGITUDE"].values

print("Rainfall data shape:", rainfall.shape)
print("Historical period:", dates.min(), "to", dates.max())

# --------------------------------------------------
# 4. VALIDATE DATA
# --------------------------------------------------

if rainfall.shape[0] != len(dates):
    raise ValueError("Rainfall dates do not match data.")

if rainfall.shape[1] != len(latitudes):
    raise ValueError("Latitude coordinates do not match data.")

if rainfall.shape[1] != len(longitudes):
    raise ValueError("Longitude coordinates do not match data.")

if not np.isfinite(rainfall).all():
    raise ValueError(
        "Missing or invalid rainfall values found."
    )

# --------------------------------------------------
# 5. MAP GRID LOCATIONS TO SUBDIVISIONS
# --------------------------------------------------

grid_subdivisions = []

for lat, lon in zip(latitudes, longitudes):

    key = (
        round(float(lat), 2),
        round(float(lon), 2)
    )

    grid_subdivisions.append(
        subdivision_lookup.get(key)
    )

grid_subdivisions = np.array(
    grid_subdivisions,
    dtype=object
)

print("\nREGIONAL MAPPING CHECK")
print("=" * 65)

for subdivision in TARGET_SUBDIVISIONS:

    count = np.sum(
        grid_subdivisions == subdivision
    )

    print(subdivision, "grid locations:", count)

    if count == 0:
        raise ValueError(
            f"No mapped locations found for {subdivision}"
        )

# --------------------------------------------------
# 6. PREPARE DATE INFORMATION
# --------------------------------------------------

years = dates.year.to_numpy()
months = dates.month.to_numpy()

historical_mask = (
    (years >= 2015) &
    (years <= 2023)
)

test_mask = years == 2024

if not historical_mask.any():
    raise ValueError("No historical records found.")

if not test_mask.any():
    raise ValueError("No 2024 test records found.")

# --------------------------------------------------
# 7. CALCULATE MONTHLY RAINFALL STATISTICS
# --------------------------------------------------

rows = []

for subdivision in TARGET_SUBDIVISIONS:

    location_indices = np.where(
        grid_subdivisions == subdivision
    )[0]

    regional_rainfall = rainfall[
        :,
        location_indices
    ]

    print(
        "\nAnalyzing:",
        subdivision,
        "| Grid locations:",
        len(location_indices)
    )

    for month in range(1, 13):

        historical_dates = (
            historical_mask &
            (months == month)
        )

        test_dates = (
            test_mask &
            (months == month)
        )

        historical_values = regional_rainfall[
            historical_dates
        ]

        test_values = regional_rainfall[
            test_dates
        ]

        # Historical 2015-2023 statistics
        historical_records = historical_values.size

        historical_rainy = np.sum(
            historical_values > 0
        )

        historical_rainfall_sum = np.sum(
            historical_values
        )

        historical_rate = (
            historical_rainy /
            historical_records * 100
        )

        historical_mean = (
            historical_rainfall_sum /
            historical_records
        )

        historical_rainy_mean = (
            historical_rainfall_sum /
            historical_rainy
            if historical_rainy > 0
            else np.nan
        )

        # 2024 test statistics
        test_records = test_values.size

        test_rainy = np.sum(
            test_values > 0
        )

        test_rainfall_sum = np.sum(
            test_values
        )

        test_rate = (
            test_rainy /
            test_records * 100
        )

        test_mean = (
            test_rainfall_sum /
            test_records
        )

        test_rainy_mean = (
            test_rainfall_sum /
            test_rainy
            if test_rainy > 0
            else np.nan
        )

        rows.append({

            "SUBDIVISION": subdivision,
            "MONTH": month,

            "HISTORICAL_RECORDS": historical_records,
            "HISTORICAL_RAINY_RECORDS": int(
                historical_rainy
            ),

            "HISTORICAL_RAINY_RATE_PERCENT":
                historical_rate,

            "HISTORICAL_MEAN_DAILY_RAINFALL_MM":
                historical_mean,

            "HISTORICAL_MEAN_RAINFALL_ON_RAINY_DAYS_MM":
                historical_rainy_mean,

            "TEST_2024_RECORDS": test_records,
            "TEST_2024_RAINY_RECORDS": int(
                test_rainy
            ),

            "TEST_2024_RAINY_RATE_PERCENT":
                test_rate,

            "TEST_2024_MEAN_DAILY_RAINFALL_MM":
                test_mean,

            "TEST_2024_MEAN_RAINFALL_ON_RAINY_DAYS_MM":
                test_rainy_mean,

            "RAINY_RATE_CHANGE_PP":
                test_rate - historical_rate,

            "MEAN_DAILY_RAINFALL_CHANGE_MM":
                test_mean - historical_mean

        })

# --------------------------------------------------
# 8. CREATE COMPARISON TABLE
# --------------------------------------------------

summary = pd.DataFrame(rows)

summary = summary.sort_values(
    ["SUBDIVISION", "MONTH"]
)

print("\nAEGISPULSE - HISTORICAL VS 2024 RAINFALL")
print("=" * 110)

for subdivision in TARGET_SUBDIVISIONS:

    regional = summary[
        summary["SUBDIVISION"] == subdivision
    ]

    print(f"\n{subdivision}")
    print("-" * 110)

    columns = [
        "MONTH",
        "HISTORICAL_RAINY_RATE_PERCENT",
        "TEST_2024_RAINY_RATE_PERCENT",
        "RAINY_RATE_CHANGE_PP",
        "HISTORICAL_MEAN_DAILY_RAINFALL_MM",
        "TEST_2024_MEAN_DAILY_RAINFALL_MM"
    ]

    print(
        regional[columns].to_string(
            index=False,
            formatters={
                "HISTORICAL_RAINY_RATE_PERCENT":
                    "{:.2f}".format,

                "TEST_2024_RAINY_RATE_PERCENT":
                    "{:.2f}".format,

                "RAINY_RATE_CHANGE_PP":
                    "{:+.2f}".format,

                "HISTORICAL_MEAN_DAILY_RAINFALL_MM":
                    "{:.3f}".format,

                "TEST_2024_MEAN_DAILY_RAINFALL_MM":
                    "{:.3f}".format
            }
        )
    )

# --------------------------------------------------
# 9. SAVE RESULTS
# --------------------------------------------------

summary.to_csv(
    OUTPUT_PATH,
    index=False
)

ds.close()

print("\nSaved comparison to:")
print(OUTPUT_PATH)

print("\nHISTORICAL RAINFALL COMPARISON COMPLETED")
