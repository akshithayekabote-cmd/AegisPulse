
import pandas as pd
import numpy as np

df = pd.read_csv(
    "data/processed/india_grid_subdivision_mapping.csv"
)

city_lat = 19.076
city_lon = 72.8777

df["DISTANCE"] = np.sqrt(
    (df["LATITUDE"] - city_lat) ** 2
    + (df["LONGITUDE"] - city_lon) ** 2
)

nearest = df.sort_values("DISTANCE").head(15)

print("AEGISPULSE - MUMBAI GRID VALIDATION")
print("=" * 70)

print(
    nearest[
        [
            "LATITUDE",
            "LONGITUDE",
            "SUBDIVISION",
            "STATUS",
            "DISTANCE"
        ]
    ].to_string(index=False)
)

print("\nVALIDATION COMPLETED")
print("=" * 70)
