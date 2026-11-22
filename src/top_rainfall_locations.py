
import xarray as xr
import pandas as pd

file_path = "data/processed/india_rainfall_2024.nc"

dataset = xr.open_dataset(file_path)

rainfall = dataset["RAINFALL"]

# Calculate average daily rainfall at each location
location_average = rainfall.sum(dim="TIME")

# Convert results into a DataFrame
df = location_average.to_dataframe().reset_index()

# Sort locations by average rainfall
df = df.sort_values(
    by="RAINFALL",
    ascending=False
)

# Display the top 10 locations
print("TOP 10 HIGH-RAINFALL LOCATIONS - 2024")

print(
    df[["LATITUDE", "LONGITUDE", "RAINFALL"]]
    .head(10)
    .to_string(index=False)
)

# Save results
df.to_csv(
    "data/processed/annual_rainfall_2024.csv",
    index=False
)

dataset.close()

print("\nResults saved successfully!")
