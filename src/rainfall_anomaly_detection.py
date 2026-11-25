
import pandas as pd

# Load daily rainfall data
file_path = "data/processed/rainfall_ml_features_2024.csv"

df = pd.read_csv(
    file_path,
    usecols=[
        "TIME",
        "location",
        "LATITUDE",
        "LONGITUDE",
        "RAINFALL"
    ]
)

# Calculate the 99th percentile for each location
thresholds = df.groupby("location")["RAINFALL"].quantile(0.99)

# Attach location-specific thresholds
df["ANOMALY_THRESHOLD"] = df["location"].map(thresholds)

# Identify unusually high rainfall observations
anomalies = df[
    df["RAINFALL"] > df["ANOMALY_THRESHOLD"]
].copy()

# Sort anomalies by rainfall
anomalies = anomalies.sort_values(
    by="RAINFALL",
    ascending=False
)

# Save anomaly report
output_path = "data/processed/rainfall_anomalies_2024.csv"

anomalies.to_csv(output_path, index=False)

# Display results
print("AEGISPULSE - RAINFALL ANOMALY DETECTION")
print("=" * 55)

print("Total observations:", len(df))
print("Detected anomalies:", len(anomalies))
print(
    "Locations with anomalies:",
    anomalies["location"].nunique()
)

print("\nTOP 10 RAINFALL ANOMALIES")

print(
    anomalies[
        [
            "TIME",
            "LATITUDE",
            "LONGITUDE",
            "RAINFALL",
            "ANOMALY_THRESHOLD"
        ]
    ].head(10).to_string(index=False)
)

print("\nAnomaly detection completed!")
print("Report saved at:", output_path)
