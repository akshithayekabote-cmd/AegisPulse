
import xarray as xr
import numpy as np
import os

INPUT_PATH = (
    "data/processed/"
    "india_rainfall_cleaned_2015_2024.nc"
)

OUTPUT_PATH = (
    "data/processed/"
    "spatial_neighbor_mapping.npz"
)

print("AEGISPULSE - SPATIAL NEIGHBOUR MAPPING")
print("=" * 65)

ds = xr.open_dataset(
    INPUT_PATH,
    engine="scipy"
)

latitudes = ds["LATITUDE"].values.astype(float)
longitudes = ds["LONGITUDE"].values.astype(float)

total_locations = len(latitudes)

# Create coordinate-to-index lookup
coordinate_lookup = {}

for index in range(total_locations):

    coordinate = (
        round(latitudes[index], 2),
        round(longitudes[index], 2)
    )

    if coordinate in coordinate_lookup:
        raise ValueError(
            f"Duplicate grid coordinate found: {coordinate}"
        )

    coordinate_lookup[coordinate] = index

# Eight surrounding directions
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

# -1 means no neighbour in that direction
neighbor_indices = np.full(
    (total_locations, 8),
    -1,
    dtype=np.int32
)

for index in range(total_locations):

    lat = latitudes[index]
    lon = longitudes[index]

    for direction_index, (lat_offset, lon_offset) in enumerate(directions):

        neighbor_coordinate = (
            round(lat + lat_offset, 2),
            round(lon + lon_offset, 2)
        )

        neighbor_index = coordinate_lookup.get(
            neighbor_coordinate,
            -1
        )

        neighbor_indices[index, direction_index] = (
            neighbor_index
        )

neighbor_counts = np.sum(
    neighbor_indices >= 0,
    axis=1
)

# Save mapping
os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

np.savez_compressed(
    OUTPUT_PATH,
    latitudes=latitudes,
    longitudes=longitudes,
    neighbor_indices=neighbor_indices,
    neighbor_counts=neighbor_counts
)

print("\nMAPPING SUMMARY")
print("-" * 65)

print("Total locations:", total_locations)
print("Mapping shape:", neighbor_indices.shape)
print("Average neighbours:", round(float(neighbor_counts.mean()), 2))
print("Minimum neighbours:", int(neighbor_counts.min()))
print("Maximum neighbours:", int(neighbor_counts.max()))

print(
    "Locations with 8 neighbours:",
    int(np.sum(neighbor_counts == 8))
)

print(
    "Locations with fewer than 4 neighbours:",
    int(np.sum(neighbor_counts < 4))
)

print("\nSaved mapping:", OUTPUT_PATH)

ds.close()

print("\nSPATIAL NEIGHBOUR MAPPING COMPLETED.")
