
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
    "weak_region_yearly_rainfall_variability.csv"
)

TARGET_SUBDIVISIONS = [
    "JAMMU & KASHMIR",
    "PUNJAB"
]

HISTORICAL_YEARS = range(2015, 2024)

# --------------------------------------------------
# 2. LOAD SUBDIVISION MAPPING
# --------------------------------------------------

mapping = pd.read_csv(MAPPING_PATH)

mapping = mapping[
    mapping["STATUS"] == "MAPPED"
].copy()

subdivision_lookup = {
    (
        round(float(row.LATITUDE), 2),
        round(float(row.LONGITUDE), 2)
    ): row.SUBDIVISION
    for row in mapping.itertuples()
}

# --------------------------------------------------
# 3. LOAD RAINFALL DATA
# --------------------------------------------------

print("Loading rainfall dataset...")

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

print("Rainfall shape:", rainfall.shape)

# --------------------------------------------------
# 4. MAP GRID LOCATIONS
# --------------------------------------------------

grid_subdivisions = np.array([
    subdivision_lookup.get(
        (
            round(float(lat), 2),
            round(float(lon), 2)
        )
    )
    for lat, lon in zip(latitudes, longitudes)
], dtype=object)

years = dates.year.to_numpy()
months = dates.month.to_numpy()

# --------------------------------------------------
# 5. CALCULATE YEARLY MONTHLY STATISTICS
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

    for year in range(2015, 2025):

        for month in range(1, 13):

            date_mask = (
                (years == year) &
                (months == month)
            )

            values = regional_rainfall[
                date_mask
            ]

            total_records = values.size

            rainy_records = np.sum(
                values > 0
            )

            rainy_rate = (
                rainy_records /
                total_records * 100
                if total_records > 0
                else np.nan
            )

            mean_rainfall = (
                np.mean(values)
                if total_records > 0
                else np.nan
            )

            rows.append({

                "SUBDIVISION": subdivision,
                "YEAR": year,
                "MONTH": month,
                "RECORDS": total_records,
                "RAINY_RECORDS": int(rainy_records),
                "RAINY_RATE_PERCENT": rainy_rate,
                "MEAN_DAILY_RAINFALL_MM": mean_rainfall

            })

yearly = pd.DataFrame(rows)

# --------------------------------------------------
# 6. CALCULATE HISTORICAL DISTRIBUTION
# --------------------------------------------------

historical = yearly[
    yearly["YEAR"].isin(HISTORICAL_YEARS)
].copy()

historical_summary = (
    historical
    .groupby(["SUBDIVISION", "MONTH"])
    .agg(
        HISTORICAL_MEAN_RATE=(
            "RAINY_RATE_PERCENT", "mean"
        ),
        HISTORICAL_MIN_RATE=(
            "RAINY_RATE_PERCENT", "min"
        ),
        HISTORICAL_MAX_RATE=(
            "RAINY_RATE_PERCENT", "max"
        ),
        HISTORICAL_STD_RATE=(
            "RAINY_RATE_PERCENT", "std"
        ),
        HISTORICAL_MEAN_RAINFALL=(
            "MEAN_DAILY_RAINFALL_MM", "mean"
        )
    )
    .reset_index()
)

# --------------------------------------------------
# 7. COMPARE 2024 WITH HISTORICAL YEARS
# --------------------------------------------------

test_2024 = yearly[
    yearly["YEAR"] == 2024
].copy()

comparison = pd.merge(
    test_2024,
    historical_summary,
    on=["SUBDIVISION", "MONTH"],
    how="inner",
    validate="one_to_one"
)

comparison["RATE_CHANGE_PP"] = (
    comparison["RAINY_RATE_PERCENT"] -
    comparison["HISTORICAL_MEAN_RATE"]
)

comparison["STANDARDIZED_RATE_DIFFERENCE"] = (
    comparison["RATE_CHANGE_PP"] /
    comparison["HISTORICAL_STD_RATE"]
)

comparison["OUTSIDE_HISTORICAL_RANGE"] = (
    (comparison["RAINY_RATE_PERCENT"] <
     comparison["HISTORICAL_MIN_RATE"]) |
    (comparison["RAINY_RATE_PERCENT"] >
     comparison["HISTORICAL_MAX_RATE"])
)

# --------------------------------------------------
# 8. DISPLAY RESULTS
# --------------------------------------------------

for subdivision in TARGET_SUBDIVISIONS:

    regional = comparison[
        comparison["SUBDIVISION"] == subdivision
    ]

    print(f"\n{subdivision}")
    print("=" * 120)

    columns = [
        "MONTH",
        "HISTORICAL_MEAN_RATE",
        "HISTORICAL_MIN_RATE",
        "HISTORICAL_MAX_RATE",
        "RAINY_RATE_PERCENT",
        "RATE_CHANGE_PP",
        "STANDARDIZED_RATE_DIFFERENCE",
        "OUTSIDE_HISTORICAL_RANGE"
    ]

    print(
        regional[columns].to_string(
            index=False,
            formatters={
                "HISTORICAL_MEAN_RATE": "{:.2f}".format,
                "HISTORICAL_MIN_RATE": "{:.2f}".format,
                "HISTORICAL_MAX_RATE": "{:.2f}".format,
                "RAINY_RATE_PERCENT": "{:.2f}".format,
                "RATE_CHANGE_PP": "{:+.2f}".format,
                "STANDARDIZED_RATE_DIFFERENCE":
                    lambda x: (
                        f"{x:+.2f}"
                        if pd.notna(x)
                        else "N/A"
                    )
            }
        )
    )

    outside_count = comparison[
        comparison["SUBDIVISION"] == subdivision
    ]["OUTSIDE_HISTORICAL_RANGE"].sum()

    print(
        "\nMonths outside historical range:",
        int(outside_count),
        "/ 12"
    )

# --------------------------------------------------
# 9. SAVE RESULTS
# --------------------------------------------------

comparison.to_csv(
    OUTPUT_PATH,
    index=False
)

ds.close()

print("\nSaved results to:")
print(OUTPUT_PATH)

print("\nYEARLY RAINFALL VARIABILITY ANALYSIS COMPLETED")
