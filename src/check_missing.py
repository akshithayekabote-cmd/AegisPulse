import xarray as xr

file_path = "data/raw/RF25_ind2024_rfp25.nc"

dataset = xr.open_dataset(file_path)

rainfall = dataset["RAINFALL"]

valid_days = rainfall.notnull().sum(dim="TIME")

total_locations = rainfall.sizes["LATITUDE"] * rainfall.sizes["LONGITUDE"]

no_data = int((valid_days == 0).sum().values)

partial_data = int(
    ((valid_days > 0) & (valid_days < 366)).sum().values
)

complete_data = int((valid_days == 366).sum().values)

print("GEOGRAPHICAL DATA COVERAGE")
print("Total grid locations:", total_locations)
print("Locations with no data:", no_data)
print("Locations with partial data:", partial_data)
print("Locations with complete data:", complete_data)

dataset.close()