
import xarray as xr
import matplotlib.pyplot as plt

file_path = "data/processed/india_rainfall_2024.nc"

dataset = xr.open_dataset(file_path)

rainfall = dataset["RAINFALL"]

# Monthly rainfall analysis
daily_average = rainfall.mean(dim="location")

monthly_average = daily_average.groupby("TIME.month").mean(dim="TIME")

months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

values = monthly_average.values

# Monthly rainfall bar chart
plt.figure(figsize=(12, 6))

plt.bar(months, values)

plt.title("Monthly Average Daily Rainfall Across India - 2024")
plt.xlabel("Month")
plt.ylabel("Average Rainfall (mm)")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "data/processed/monthly_rainfall_2024.png",
    dpi=300
)

# Annual rainfall distribution
annual_rainfall = rainfall.sum(dim="TIME")

plt.figure(figsize=(12, 6))

plt.hist(
    annual_rainfall.values,
    bins=30,
    edgecolor="black"
)

plt.title("Annual Rainfall Distribution Across India - 2024")
plt.xlabel("Total Annual Rainfall (mm)")
plt.ylabel("Number of Geographical Locations")

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "data/processed/rainfall_distribution_2024.png",
    dpi=300
)

# Display both graphs
plt.show()

dataset.close()

print("Both rainfall visualizations created successfully!")
