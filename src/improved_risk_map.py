
import pandas as pd
import folium
from folium.plugins import MarkerCluster

# Load rainfall classification data
df = pd.read_csv("data/processed/rainfall_classification_2024.csv")

# Create India map
india_map = folium.Map(
    location=[22.5, 80.0],
    zoom_start=5,
    tiles="OpenStreetMap"
)

# Define risk category colors
colors = {
    "Normal": "green",
    "Elevated": "beige",
    "Very High": "orange",
    "Extreme": "red"
}

# Create separate layers for each risk category
for risk_level, color in colors.items():

    category_data = df[df["RISK_LEVEL"] == risk_level]

    layer = folium.FeatureGroup(
        name=f"{risk_level} ({len(category_data)} locations)"
    )

    marker_cluster = MarkerCluster().add_to(layer)

    for _, row in category_data.iterrows():

        popup = f"""
        <b>AegisPulse Rainfall Details</b><br>
        Latitude: {row['LATITUDE']}<br>
        Longitude: {row['LONGITUDE']}<br>
        Annual Rainfall: {row['RAINFALL']:.2f} mm<br>
        Risk Category: {risk_level}
        """

        folium.CircleMarker(
            location=[row["LATITUDE"], row["LONGITUDE"]],
            radius=5,
            popup=folium.Popup(popup, max_width=250),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8
        ).add_to(marker_cluster)

    layer.add_to(india_map)

# Add map title
title_html = """
<div style="text-align:center; font-size:20px; font-weight:bold;
padding:10px; color:#00BFFF;">
AegisPulse - India Rainfall Risk Monitoring 2024
</div>
"""

india_map.get_root().html.add_child(
    folium.Element(title_html)
)

# Add interactive layer controls
folium.LayerControl(collapsed=False).add_to(india_map)

# Add risk summary
summary_html = """
<div style="position:fixed; bottom:25px; left:25px;
background:white; padding:12px; border:2px solid #333;
border-radius:8px; z-index:9999; font-size:13px;">

<b>AegisPulse Risk Summary</b><br>
<span style="color:green;">●</span> Normal<br>
<span style="color:#c5a000;">●</span> Elevated<br>
<span style="color:orange;">●</span> Very High<br>
<span style="color:red;">●</span> Extreme<br>
<br>
<b>Total Locations:</b> 4,964

</div>
"""

india_map.get_root().html.add_child(
    folium.Element(summary_html)
)

# Save improved map
output_path = "data/processed/india_rainfall_risk_dashboard.html"

india_map.save(output_path)

print("AegisPulse improved risk map created successfully!")
print("Total locations:", len(df))
print("Risk distribution:")
print(df["RISK_LEVEL"].value_counts())
print("Saved at:", output_path)
