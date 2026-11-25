
import pandas as pd

file_path = "data/processed/india_grid_subdivision_mapping.csv"

df = pd.read_csv(file_path)

unmapped = df[df["STATUS"] == "UNMAPPED"].copy()

print("AEGISPULSE - UNMAPPED LOCATION DIAGNOSTICS")
print("=" * 65)

print("\nSUMMARY")
print("-" * 65)
print("Total locations:", len(df))
print("Unmapped locations:", len(unmapped))

if len(unmapped) > 0:

    print("\nUNMAPPED COORDINATE RANGES")
    print("-" * 65)

    print(
        "Latitude:",
        unmapped["LATITUDE"].min(),
        "to",
        unmapped["LATITUDE"].max()
    )

    print(
        "Longitude:",
        unmapped["LONGITUDE"].min(),
        "to",
        unmapped["LONGITUDE"].max()
    )

    print("\nFIRST 40 UNMAPPED LOCATIONS")
    print("-" * 65)

    print(
        unmapped[
            ["LATITUDE", "LONGITUDE"]
        ].head(40).to_string(index=False)
    )

    print("\nUNMAPPED LOCATIONS BY LATITUDE BAND")
    print("-" * 65)

    unmapped["LATITUDE_BAND"] = (
        (unmapped["LATITUDE"] // 5) * 5
    )

    print(
        unmapped.groupby("LATITUDE_BAND").size()
    )

    print("\nUNMAPPED LOCATIONS BY LONGITUDE BAND")
    print("-" * 65)

    unmapped["LONGITUDE_BAND"] = (
        (unmapped["LONGITUDE"] // 5) * 5
    )

    print(
        unmapped.groupby("LONGITUDE_BAND").size()
    )

print("\nDIAGNOSTICS COMPLETED")
print("=" * 65)
