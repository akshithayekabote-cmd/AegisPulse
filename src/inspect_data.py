import xarray as xr

file_path = "data/raw/RF25_ind2024_rfp25.nc"

dataset = xr.open_dataset(file_path)

rainfall = dataset["RAINFALL"]

print("Rainfall Information:")
print("Dimensions:", rainfall.dims)
print("Units:", rainfall.attrs)

print("\nRainfall Statistics:")
print("Minimum:", float(rainfall.min(skipna=True).values))
print("Maximum:", float(rainfall.max(skipna=True).values))
print("Average:", float(rainfall.mean(skipna=True).values))

print("\nMissing Values:")
print(int(rainfall.isnull().sum().values))

dataset.close()