
import xarray as xr
import os

folder = "data/raw/imd_yearly"
output_folder = "data/processed"

datasets = []

print("AEGISPULSE - HISTORICAL DATA MERGING")
print("=" * 55)

for year in range(2015, 2025):

    if year == 2024:
        file_path = "data/raw/RF25_ind2024_rfp25.nc"
    else:
        filename = f"RF25_ind{year}_rfp25.nc"
        file_path = os.path.join(folder, filename)

    print(f"Loading {year}...")

    ds = xr.open_dataset(file_path, engine="scipy")

    datasets.append(ds)

# Verify spatial coordinates
reference_lat = datasets[0]["LATITUDE"].values
reference_lon = datasets[0]["LONGITUDE"].values

for ds in datasets[1:]:
    assert (reference_lat == ds["LATITUDE"].values).all()
    assert (reference_lon == ds["LONGITUDE"].values).all()

# Combine datasets chronologically
combined = xr.concat(
    datasets,
    dim="TIME"
).sortby("TIME")

# Verify dates
dates = combined["TIME"].values

assert len(dates) == len(set(dates)), "Duplicate dates found!"

print("\nMERGING RESULTS")
print("=" * 55)
print("Total days:", len(dates))
print("Start date:", dates[0])
print("End date:", dates[-1])
print("Dimensions:", combined["RAINFALL"].shape)

# Save combined dataset
os.makedirs(output_folder, exist_ok=True)

output_path = os.path.join(
    output_folder,
    "india_rainfall_2015_2024.nc"
)

combined.to_netcdf(
    output_path,
    engine="scipy",
    format="NETCDF3_64BIT"
)

print("\nHistorical dataset saved successfully!")
print("Location:", output_path)

for ds in datasets:
    ds.close()

combined.close()
