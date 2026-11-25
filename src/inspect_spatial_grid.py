
import xarray as xr
import numpy as np

INPUT_PATH = (
    "data/processed/"
    "india_rainfall_cleaned_2015_2024.nc"
)

print("AEGISPULSE - SPATIAL GRID INSPECTION")
print("=" * 65)

ds = xr.open_dataset(
    INPUT_PATH,
    engine="scipy"
)

latitudes = ds["LATITUDE"].values
longitudes = ds["LONGITUDE"].values

coordinates = [
    (round(float(lat), 2), round(float(lon), 2))
    for lat, lon in zip(latitudes, longitudes)
]

coordinate_set = set(coordinates)

unique_latitudes = np.unique(latitudes)
unique_longitudes = np.unique(longitudes)

print("\nGRID INFORMATION")
print("-" * 65)

print("Total grid locations:", len(coordinates))
print("Unique latitudes:", len(unique_latitudes))
print("Unique longitudes:", len(unique_longitudes))

print(
    "Latitude range:",
    round(float(latitudes.min()), 2),
    "to",
    round(float(latitudes.max()), 2)
)

print(
    "Longitude range:",
    round(float(longitudes.min()), 2),
    "to",
    round(float(longitudes.max()), 2)
)

# Check grid spacing
print("\nGRID SPACING")
print("-" * 65)

lat_steps = np.unique(
    np.round(np.diff(unique_latitudes), 4)
)

lon_steps = np.unique(
    np.round(np.diff(unique_longitudes), 4)
)

print("Latitude spacing:", lat_steps)
print("Longitude spacing:", lon_steps)

# Check eight surrounding grid cells
directions = [
    (-0.25, -0.25),
    (-0.25, 0.00),
    (-0.25, 0.25),
    (0.00, -0.25),
    (0.00, 0.25),
    (0.25, -0.25),
    (0.25, 0.00),
    (0.25, 0.25)
]

neighbor_counts = []

for lat, lon in coordinates:

    count = 0

    for lat_offset, lon_offset in directions:

        neighbor = (
            round(lat + lat_offset, 2),
            round(lon + lon_offset, 2)
        )

        if neighbor in coordinate_set:
            count += 1

    neighbor_counts.append(count)

neighbor_counts = np.array(
    neighbor_counts
)

print("\nNEIGHBOUR AVAILABILITY")
print("-" * 65)

for count in range(9):

    locations = int(
        np.sum(neighbor_counts == count)
    )

    print(
        f"Locations with {count} neighbours:",
        locations
    )

print("\nSUMMARY")
print("-" * 65)

print(
    "Average neighbours per location:",
    round(float(neighbor_counts.mean()), 2)
)

print(
    "Locations with at least 4 neighbours:",
    int(np.sum(neighbor_counts >= 4))
)

print(
    "Locations with no neighbours:",
    int(np.sum(neighbor_counts == 0))
)

ds.close()

print("\nSPATIAL GRID INSPECTION COMPLETED.")
