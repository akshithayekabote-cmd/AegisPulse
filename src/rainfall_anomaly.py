
import xarray as xr
import pandas as pd

# Load the cleaned rainfall dataset
file_path = "data/processed/india_rainfall_2024.nc"
dataset = xr.open_dataset(file_path)

rainfall = dataset["RAINFALL"]

# Calculate total annual rainfall at each location
annual_rainfall = rainfall.sum(dim="TIME")

# Convert rainfall data into a DataFrame
df = annual_rainfall.to_dataframe().reset_index()

# Calculate rainfall classification thresholds
percentile_75 = df["RAINFALL"].quantile(0.75)
percentile_90 = df["RAINFALL"].quantile(0.90)
percentile_95 = df["RAINFALL"].quantile(0.95)

# Classify rainfall levels
def classify_rainfall(value):
    if value > percentile_95:
        return "Extreme"
    elif value > percentile_90:
        return "Very High"
    elif value > percentile_75:
        return "Elevated"
    else:
        return "Normal"

df["RISK_LEVEL"] = df["RAINFALL"].apply(classify_rainfall)

# Display classification summary
print("RAINFALL CLASSIFICATION SUMMARY - 2024")
print(df["RISK_LEVEL"].value_counts())

# Display the highest rainfall locations
print("\nTOP 10 EXTREME RAINFALL LOCATIONS")
extreme_locations = df[df["RISK_LEVEL"] == "Extreme"]
extreme_locations = extreme_locations.sort_values(
    by="RAINFALL", ascending=False
)

print(
    extreme_locations[
        ["LATITUDE", "LONGITUDE", "RAINFALL", "RISK_LEVEL"]
    ].head(10).to_string(index=False)
)

# Save classified results
df.to_csv("data/processed/rainfall_classification_2024.csv", index=False)

dataset.close()

print("\nRainfall classification completed successfully!")
print("Results saved at: data/processed/rainfall_classification_2024.csv")
