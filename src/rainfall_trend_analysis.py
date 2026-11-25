
import pandas as pd
import matplotlib.pyplot as plt

# Load daily rainfall data in chunks
file_path = "data/processed/rainfall_ml_features_2024.csv"

monthly_data = []

for chunk in pd.read_csv(
    file_path,
    usecols=["TIME", "MONTH", "RAINFALL"],
    chunksize=500000
):
    summary = chunk.groupby("MONTH")["RAINFALL"].agg(
        ["sum", "count", "max"]
    )

    monthly_data.append(summary)

# Combine monthly summaries
monthly = pd.concat(monthly_data).groupby(level=0).sum()

monthly["AVERAGE_RAINFALL"] = (
    monthly["sum"] / monthly["count"]
)

months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

monthly["MONTH_NAME"] = months

# Display monthly analysis
print("AEGISPULSE - MONTHLY RAINFALL TREND ANALYSIS")
print("=" * 55)

print(
    monthly[
        ["MONTH_NAME", "AVERAGE_RAINFALL", "max", "count"]
    ].to_string(index=False)
)

# Save monthly report
monthly.to_csv(
    "data/processed/monthly_rainfall_trend_2024.csv"
)

# Create monthly rainfall chart
plt.figure(figsize=(12, 6))

plt.bar(
    monthly["MONTH_NAME"],
    monthly["AVERAGE_RAINFALL"]
)

plt.title("Monthly Average Daily Rainfall Across India - 2024")
plt.xlabel("Month")
plt.ylabel("Average Daily Rainfall (mm)")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "data/processed/monthly_rainfall_trend_2024.png",
    dpi=300
)

plt.show()

print("\nMonthly rainfall analysis completed!")
print("CSV report and chart saved successfully.")
