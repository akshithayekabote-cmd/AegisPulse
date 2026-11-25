
import pandas as pd

# Load annual rainfall data
df = pd.read_csv("data/processed/annual_rainfall_2024.csv")

# Sort locations by annual rainfall
df = df.sort_values(
    by="RAINFALL",
    ascending=False
)

# Select the top 10 rainfall locations
top_10 = df.head(10).copy()

# Add ranking
top_10.insert(
    0,
    "RANK",
    range(1, len(top_10) + 1)
)

# Save hotspot report
output_path = "data/processed/top_10_rainfall_hotspots.csv"

top_10.to_csv(output_path, index=False)

# Display results
print("AEGISPULSE - TOP 10 RAINFALL HOTSPOTS 2024")
print("=" * 55)

print(
    top_10.to_string(
        index=False,
        formatters={"RAINFALL": "{:.2f}".format}
    )
)

print("\nHotspot report created successfully!")
print("Saved at:", output_path)
