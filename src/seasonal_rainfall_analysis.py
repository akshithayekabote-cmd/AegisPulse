
import pandas as pd
import matplotlib.pyplot as plt

# Load daily rainfall dataset
file_path = "data/processed/rainfall_ml_features_2024.csv"

monthly_data = []

for chunk in pd.read_csv(
    file_path,
    usecols=["MONTH", "RAINFALL"],
    chunksize=500000
):
    summary = chunk.groupby("MONTH")["RAINFALL"].agg(
        ["sum", "count"]
    )
    monthly_data.append(summary)

# Calculate monthly averages
monthly = pd.concat(monthly_data).groupby(level=0).sum()

monthly["AVERAGE"] = monthly["sum"] / monthly["count"]

# Define seasons
seasons = {
    "Winter": [1, 2],
    "Pre-Monsoon": [3, 4, 5],
    "Monsoon": [6, 7, 8, 9],
    "Post-Monsoon": [10, 11, 12]
}

# Calculate seasonal average daily rainfall
results = []

for season, months in seasons.items():
    data = monthly.loc[months]

    average = data["sum"].sum() / data["count"].sum()

    results.append({
        "SEASON": season,
        "AVERAGE_DAILY_RAINFALL": average
    })

seasonal = pd.DataFrame(results)

print("AEGISPULSE - SEASONAL RAINFALL ANALYSIS")
print("=" * 50)
print(seasonal.to_string(index=False))

# Save seasonal report
seasonal.to_csv(
    "data/processed/seasonal_rainfall_2024.csv",
    index=False
)

# Create seasonal chart
plt.figure(figsize=(10, 6))

plt.bar(
    seasonal["SEASON"],
    seasonal["AVERAGE_DAILY_RAINFALL"]
)

plt.title("Seasonal Average Daily Rainfall Across India - 2024")
plt.xlabel("Season")
plt.ylabel("Average Daily Rainfall (mm)")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "data/processed/seasonal_rainfall_2024.png",
    dpi=300
)

plt.show()

print("\nSeasonal analysis completed successfully!")
