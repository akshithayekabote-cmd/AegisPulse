
import xarray as xr

file_path = "data/processed/india_rainfall_2024.nc"

dataset = xr.open_dataset(file_path)

rainfall = dataset["RAINFALL"]

# Calculate average rainfall across all grid locations for each day
daily_average = rainfall.mean(dim="location")

# Calculate monthly average rainfall
monthly_average = daily_average.groupby("TIME.month").mean(dim="TIME")

print("MONTHLY RAINFALL ANALYSIS - 2024")

for month, value in zip(monthly_average.month.values,
                        monthly_average.values):

    print(f"Month {month}: {value:.2f} mm")

# Identify the highest and lowest monthly averages
highest_month = int(monthly_average.argmax().values) + 1
lowest_month = int(monthly_average.argmin().values) + 1

print("\nHighest rainfall month:", highest_month)
print("Highest monthly average:",
      float(monthly_average.max().values), "mm")

print("\nLowest rainfall month:", lowest_month)
print("Lowest monthly average:",
      float(monthly_average.min().values), "mm")
# Identify the top 10 highest-rainfall locations

annual_total = rainfall.sum(dim="TIME")

top_locations = annual_total.sortby(
    annual_total, ascending=False
).isel(location=slice(0, 10))

print("\nTOP 10 HIGHEST RAINFALL LOCATIONS - 2024")

for rank, (lat, lon, value) in enumerate(
    zip(
        top_locations["LATITUDE"].values,
        top_locations["LONGITUDE"].values,
        top_locations.values
    ),
    start=1
):
    print(
        f"{rank}. Latitude: {lat:.2f}, "
        f"Longitude: {lon:.2f}, "
        f"Total Rainfall: {value:.2f} mm"
    )

# Statistical rainfall analysis

print("\nSTATISTICAL RAINFALL ANALYSIS - 2024")

print("Mean Rainfall:",
      float(annual_total.mean().values), "mm")

print("Median Rainfall:",
      float(annual_total.median().values), "mm")

print("Minimum Rainfall:",
      float(annual_total.min().values), "mm")

print("Maximum Rainfall:",
      float(annual_total.max().values), "mm")

print("Standard Deviation:",
      float(annual_total.std().values), "mm")

print("75th Percentile:",
      float(annual_total.quantile(0.75).values), "mm")

print("90th Percentile:",
      float(annual_total.quantile(0.90).values), "mm")

print("95th Percentile:",
      float(annual_total.quantile(0.95).values), "mm")    

dataset.close()
