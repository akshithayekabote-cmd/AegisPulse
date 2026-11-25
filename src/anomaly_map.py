
import pandas as pd
import folium
from folium.plugins import MarkerCluster

# Load anomaly data
file_path = "data/processed/rainfall_anomalies_2024.csv"
df = pd.read_csv(file_path)

# Create India map
india_map = folium.Map(
    location=[22.5, 80],
    zoom_start=5,
    tiles="OpenStreetMap"
)

# Create marker cluster
cluster = MarkerCluster().add_to(india_map)

# Add anomaly markers
for _, row in df.iterrows():

    rainfall = row["RAINFALL"]

    if rainfall >= 500:
        color = "red"
    elif rainfall >= 300:
        color = "orange"
    else:
        color = "blue"

    popup_text = f"""
    <b>Date:</b> {row['TIME']}<br>
    <b>Rainfall:</b> {rainfall:.2f} mm<br>
    <b>Anomaly Threshold:</b> {row['ANOMALY_THRESHOLD']:.2f} mm<br>
    <b>Latitude:</b> {row['LATITUDE']}<br>
    <b>Longitude:</b> {row['LONGITUDE']}
    """

    folium.CircleMarker(
        location=[row["LATITUDE"], row["LONGITUDE"]],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(popup_text, max_width=300)
    ).add_to(cluster)

# Save map
output_path = "data/processed/india_rainfall_anomaly_map.html"

india_map.save(output_path)

print("AegisPulse anomaly map created successfully!")
print("Total anomaly observations:", len(df))
print("Saved at:", output_path)
