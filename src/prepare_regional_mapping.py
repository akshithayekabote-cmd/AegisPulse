
import json
import numpy as np
import pandas as pd
import xarray as xr

GEOJSON_PATH = "data/raw/imd_subdivisions.geojson"
DATASET_PATH = "data/processed/india_rainfall_cleaned_2015_2024.nc"
OUTPUT_PATH = "data/processed/india_grid_subdivision_mapping.csv"

# WGS84 ellipsoid and UTM Zone 44N parameters
a = 6378137.0
f = 1 / 298.257223563
e2 = f * (2 - f)
ep2 = e2 / (1 - e2)
k0 = 0.9996
lon0 = np.radians(81.0)


def utm44_to_latlon(easting, northing):
    x = (easting - 500000) / k0
    y = northing / k0

    mu = y / (
        a * (
            1 - e2 / 4
            - 3 * e2**2 / 64
            - 5 * e2**3 / 256
        )
    )

    e1 = (1 - np.sqrt(1 - e2)) / (
        1 + np.sqrt(1 - e2)
    )

    j1 = 3 * e1 / 2 - 27 * e1**3 / 32
    j2 = 21 * e1**2 / 16 - 55 * e1**4 / 32
    j3 = 151 * e1**3 / 96
    j4 = 1097 * e1**4 / 512

    fp = (
        mu
        + j1 * np.sin(2 * mu)
        + j2 * np.sin(4 * mu)
        + j3 * np.sin(6 * mu)
        + j4 * np.sin(8 * mu)
    )

    c1 = ep2 * np.cos(fp)**2
    t1 = np.tan(fp)**2

    n1 = a / np.sqrt(1 - e2 * np.sin(fp)**2)

    r1 = (
        a * (1 - e2)
        / (1 - e2 * np.sin(fp)**2)**1.5
    )

    d = x / n1

    latitude = (
        fp
        - (n1 * np.tan(fp) / r1) * (
            d**2 / 2
            - (5 + 3*t1 + 10*c1 - 4*c1**2 - 9*ep2)
            * d**4 / 24
            + (
                61 + 90*t1 + 298*c1
                + 45*t1**2 - 252*ep2 - 3*c1**2
            ) * d**6 / 720
        )
    )

    longitude = (
        lon0
        + (
            d
            - (1 + 2*t1 + c1) * d**3 / 6
            + (
                5 - 2*c1 + 28*t1
                - 3*c1**2 + 8*ep2 + 24*t1**2
            ) * d**5 / 120
        ) / np.cos(fp)
    )

    return np.degrees(latitude), np.degrees(longitude)


def convert_ring(ring):
    coordinates = np.asarray(ring, dtype=float)

    lat, lon = utm44_to_latlon(
        coordinates[:, 0],
        coordinates[:, 1]
    )

    # GeoJSON point order is longitude, latitude
    return np.column_stack((lon, lat))


def points_in_ring(px, py, ring):
    inside = np.zeros(len(px), dtype=bool)
    boundary = np.zeros(len(px), dtype=bool)

    x1 = ring[:-1, 0]
    y1 = ring[:-1, 1]
    x2 = ring[1:, 0]
    y2 = ring[1:, 1]

    tolerance = 1e-9

    for i in range(len(x1)):
        dx = x2[i] - x1[i]
        dy = y2[i] - y1[i]

        cross = (
            (px - x1[i]) * dy
            - (py - y1[i]) * dx
        )

        dot = (
            (px - x1[i]) * dx
            + (py - y1[i]) * dy
        )

        length_squared = dx**2 + dy**2

        on_edge = (
            (np.abs(cross) <= tolerance * (abs(dx) + abs(dy) + 1))
            & (dot >= -tolerance)
            & (dot <= length_squared + tolerance)
        )

        boundary |= on_edge

        crosses = (
            (y1[i] > py) != (y2[i] > py)
        ) & (
            px < x1[i] + (py - y1[i]) * dx / (dy + 1e-30)
        )

        inside ^= crosses

    return inside, boundary


def points_in_polygon(px, py, rings):
    outer_inside, outer_boundary = points_in_ring(
        px, py, rings[0]
    )

    result = outer_inside | outer_boundary

    # Exclude points inside holes
    for hole in rings[1:]:
        hole_inside, hole_boundary = points_in_ring(
            px, py, hole
        )

        result &= ~(hole_inside & ~hole_boundary)

    return result


print("AEGISPULSE - IMD REGIONAL GRID MAPPING")
print("=" * 65)

# Load rainfall grid coordinates
ds = xr.open_dataset(DATASET_PATH, engine="scipy")

latitudes = ds["LATITUDE"].values.astype(float)
longitudes = ds["LONGITUDE"].values.astype(float)

ds.close()

print("\nRAINFALL GRID")
print("-" * 65)
print("Total locations:", len(latitudes))
print("Latitude range:", latitudes.min(), "to", latitudes.max())
print("Longitude range:", longitudes.min(), "to", longitudes.max())

# Load subdivision boundaries
with open(GEOJSON_PATH, "r", encoding="utf-8") as file:
    geojson = json.load(file)

features = geojson["features"]

print("\nBOUNDARY DATA")
print("-" * 65)
print("Total subdivisions:", len(features))

# Convert all subdivision geometries to geographic coordinates
subdivisions = []

for feature in features:
    properties = feature["properties"]
    geometry = feature["geometry"]

    name = properties.get("subdivisio", "UNKNOWN")
    geometry_type = geometry["type"]
    coordinates = geometry["coordinates"]

    if geometry_type == "Polygon":
        polygons = [coordinates]
    elif geometry_type == "MultiPolygon":
        polygons = coordinates
    else:
        print("Skipping unsupported geometry:", geometry_type)
        continue

    converted_polygons = []

    for polygon in polygons:
        converted_rings = [
            convert_ring(ring) for ring in polygon
        ]

        converted_polygons.append(converted_rings)

    subdivisions.append({
        "name": name,
        "polygons": converted_polygons
    })

# Validate converted boundary coordinates
all_boundary_points = np.vstack([
    ring
    for subdivision in subdivisions
    for polygon in subdivision["polygons"]
    for ring in polygon
])

print("\nCONVERTED BOUNDARY RANGES")
print("-" * 65)
print(
    "Latitude:",
    round(all_boundary_points[:, 1].min(), 3),
    "to",
    round(all_boundary_points[:, 1].max(), 3)
)
print(
    "Longitude:",
    round(all_boundary_points[:, 0].min(), 3),
    "to",
    round(all_boundary_points[:, 0].max(), 3)
)

# Match each rainfall grid point to a subdivision
matches = [[] for _ in range(len(latitudes))]

px = longitudes
py = latitudes

print("\nMATCHING GRID LOCATIONS")
print("-" * 65)

for subdivision in subdivisions:
    name = subdivision["name"]
    matched_indices = set()

    for polygon in subdivision["polygons"]:
        outer = polygon[0]

        # Bounding-box screening
        min_lon = outer[:, 0].min()
        max_lon = outer[:, 0].max()
        min_lat = outer[:, 1].min()
        max_lat = outer[:, 1].max()

        candidates = np.where(
            (px >= min_lon - 1e-8)
            & (px <= max_lon + 1e-8)
            & (py >= min_lat - 1e-8)
            & (py <= max_lat + 1e-8)
        )[0]

        if len(candidates) == 0:
            continue

        inside = points_in_polygon(
            px[candidates],
            py[candidates],
            polygon
        )

        for index in candidates[inside]:
            matched_indices.add(int(index))

    for index in matched_indices:
        matches[index].append(name)

    print(f"{name}: {len(matched_indices)} locations")

# Create mapping table
rows = []

for index in range(len(latitudes)):
    found = matches[index]

    rows.append({
        "LOCATION_ID": index,
        "LATITUDE": latitudes[index],
        "LONGITUDE": longitudes[index],
        "SUBDIVISION": found[0] if found else "UNMAPPED",
        "MATCH_COUNT": len(found),
        "STATUS": (
            "UNMAPPED" if len(found) == 0
            else "AMBIGUOUS" if len(found) > 1
            else "MAPPED"
        )
    })

mapping = pd.DataFrame(rows)

mapping.to_csv(OUTPUT_PATH, index=False)

# Final diagnostics
mapped_count = (mapping["STATUS"] == "MAPPED").sum()
unmapped_count = (mapping["STATUS"] == "UNMAPPED").sum()
ambiguous_count = (mapping["STATUS"] == "AMBIGUOUS").sum()

print("\nFINAL MAPPING SUMMARY")
print("=" * 65)
print("Total grid locations:", len(mapping))
print("Mapped locations:", mapped_count)
print("Unmapped locations:", unmapped_count)
print("Ambiguous locations:", ambiguous_count)

print(
    "Mapping coverage:",
    round(mapped_count / len(mapping) * 100, 2),
    "%"
)

print("\nOUTPUT FILE")
print("-" * 65)
print(OUTPUT_PATH)

print("\nREGIONAL MAPPING COMPLETED")
print("=" * 65)
