import xarray as xr

path = "data/processed/india_rainfall_cleaned_2015_2024.nc"

print("AEGISPULSE - PREDICTION DATA INSPECTION")
print("=" * 60)

ds = xr.open_dataset(path, engine="scipy")

print("\nDataset details:")
print(ds)

print("\nVariables:")
print(list(ds.data_vars))

print("\nCoordinates:")
print(list(ds.coords))

print("\nDimensions:")
print(ds.dims)

print("\nFirst 5 dates:")
print(ds.TIME.values[:5])

print("\nLast 5 dates:")
print(ds.TIME.values[-5:])

print("\nRainfall data shape:")
print(ds["RAINFALL"].shape)

ds.close()

print("\nINSPECTION COMPLETED")