
import pandas as pd
import folium
from folium.plugins import MarkerCluster

# Load top 10 rainfall hotspots
df = pd.read_csv(
    "data/processed/top_10_rainfall_hotspots.csv"
)

# Create India map
india_map = folium.Map(
    location=[22.5, 80.0],
    zoom_start=5,
    tiles="OpenStreetMap"
)

# Create marker cluster
marker_cluster = MarkerCluster().add_to(india_map)

# Add hotspot markers
for _, row in df.iterrows():

    popup = f"""
    <b>AegisPulse Rainfall Hotspot</b><br>
    Rank: {row['RANK']}<br>
    Latitude: {row['LATITUDE']}<br>
    Longitude: {row['LONGITUDE']}<br>
    Annual Rainfall: {row['RAINFALL']:.2f} mm
    """

    folium.Marker(
        location=[
            row["LATITUDE"],
            row["LONGITUDE"]
        ],
        popup=folium.Popup(popup, max_width=250),
        tooltip=f"Hotspot Rank #{int(row['RANK'])}",
        icon=folium.Icon(
            color="red",
            icon="cloud"
        )
    ).add_to(marker_cluster)

# Add map title
title_html = """
<h3 style="text-align:center; color:#00BFFF;">
AegisPulse - Top 10 Rainfall Hotspots in India (2024)
</h3>
"""

india_map.get_root().html.add_child(
    folium.Element(title_html)
)

# Save hotspot map
output_path = (
    "data/processed/india_top_10_hotspots.html"
)

india_map.save(output_path)

print("AegisPulse hotspot map created successfully!")
print("Total hotspots:", len(df))
print("Saved at:", output_path)
