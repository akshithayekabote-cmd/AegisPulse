
import xarray as xr
import numpy as np
import os

folder = "data/raw/imd_yearly"

years = range(2015, 2024)

reference_lat = None
reference_lon = None

print("AEGISPULSE - HISTORICAL DATA VALIDATION")
print("=" * 65)

for year in years:

    filename = f"RF25_ind{year}_rfp25.nc"
    file_path = os.path.join(folder, filename)

    print(f"\nChecking {year}...")

    if not os.path.exists(file_path):
        print("ERROR: File not found!")
        continue

    ds = xr.open_dataset(file_path, engine="scipy")

    rainfall = ds["RAINFALL"].values
    latitude = ds["LATITUDE"].values
    longitude = ds["LONGITUDE"].values
    dates = ds["TIME"].values

    missing = np.isnan(rainfall).sum()
    valid = np.isfinite(rainfall).sum()

    print("File:", filename)
    print("Days:", len(dates))
    print("Dimensions:", ds["RAINFALL"].shape)
    print("Latitude:", latitude.min(), "to", latitude.max())
    print("Longitude:", longitude.min(), "to", longitude.max())
    print("Missing values:", missing)
    print("Valid values:", valid)
    print("Minimum rainfall:", np.nanmin(rainfall))
    print("Maximum rainfall:", np.nanmax(rainfall))

    if reference_lat is None:
        reference_lat = latitude.copy()
        reference_lon = longitude.copy()
    else:
        print(
            "Coordinates match:",
            np.array_equal(reference_lat, latitude)
            and np.array_equal(reference_lon, longitude)
        )

    print("Date range:", dates[0], "to", dates[-1])

    ds.close()

print("\nVALIDATION COMPLETED")
