
import pandas as pd
import folium

file_path = "data/processed/annual_rainfall_2024.csv"

df = pd.read_csv(file_path)

# Create an India-centered map
india_map = folium.Map(
    location=[22.5, 80.0],
    zoom_start=5,
    tiles="OpenStreetMap"
)

# Add rainfall locations to the map
for _, row in df.head(10).iterrows(): 

    folium.CircleMarker(
        location=[
            row["LATITUDE"],
            row["LONGITUDE"]
        ],
        radius=7,
        popup=(
            f"Latitude: {row['LATITUDE']}<br>"
            f"Longitude: {row['LONGITUDE']}<br>"
            f"Total Annual Rainfall: {row['RAINFALL']:.2f} mm"
        ),
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=0.7
    ).add_to(india_map)

# Save the interactive map
india_map.save(
    "data/processed/india_annual_hotspots.html"
)

print("Interactive annual rainfall hotspot map created successfully!")
print("Saved at: data/processed/india_annual_hotspots.html")