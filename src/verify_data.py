
import xarray as xr

file_path = "data/processed/india_rainfall_2024.nc"

dataset = xr.open_dataset(file_path)

rainfall = dataset["RAINFALL"]

print("CLEANED DATASET VERIFICATION")

print("Dimensions:", dataset.sizes)

print("Missing values:", int(rainfall.isnull().sum().values))

print("Minimum rainfall:", float(rainfall.min().values))

print("Maximum rainfall:", float(rainfall.max().values))

print("Average rainfall:", float(rainfall.mean().values))

print("Latitude range:",
      float(dataset.LATITUDE.min()),
      "to",
      float(dataset.LATITUDE.max()))

print("Longitude range:",
      float(dataset.LONGITUDE.min()),
      "to",
      float(dataset.LONGITUDE.max()))

dataset.close()
