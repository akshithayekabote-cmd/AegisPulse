
import xarray as xr

input_file = "data/raw/RF25_ind2024_rfp25.nc"

output_file = "data/processed/india_rainfall_2024.nc"

dataset = xr.open_dataset(input_file)

rainfall = dataset["RAINFALL"]

cleaned = rainfall.stack(
    location=("LATITUDE", "LONGITUDE")
).dropna(dim="location", how="all")

# Convert MultiIndex into normal coordinates
cleaned = cleaned.reset_index("location")

cleaned_dataset = cleaned.to_dataset(name="RAINFALL")

cleaned_dataset.to_netcdf(output_file)

print("DATA CLEANING COMPLETED")
print(cleaned_dataset)

print("\nSaved to:", output_file)

dataset.close()

print("\nOriginal dataset remains unchanged.")
