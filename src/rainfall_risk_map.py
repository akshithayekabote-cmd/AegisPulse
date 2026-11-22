
import pandas as pd
import folium

# Load classified rainfall data
file_path = "data/processed/rainfall_classification_2024.csv"
df = pd.read_csv(file_path)

# Create an interactive map of India
india_map = folium.Map(
    location=[22.5, 80.0],
    zoom_start=5,
    tiles="OpenStreetMap"
)

# Define colors for each rainfall category
colors = {
    "Normal": "green",
    "Elevated": "yellow",
    "Very High": "orange",
    "Extreme": "red"
}

# Add all geographical locations to the map
for _, row in df.iterrows():
    risk_level = row["RISK_LEVEL"]
    color = colors[risk_level]

    folium.CircleMarker(
        location=[row["LATITUDE"], row["LONGITUDE"]],
        radius=4,
        popup=(
            f"Latitude: {row['LATITUDE']}<br>"
            f"Longitude: {row['LONGITUDE']}<br>"
            f"Annual Rainfall: {row['RAINFALL']:.2f} mm<br>"
            f"Classification: {risk_level}"
        ),
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7
    ).add_to(india_map)

# Add map title
title_html = """
<h3 align="center" style="font-size:18px">
<b>AegisPulse - India Rainfall Risk Classification 2024</b>
</h3>
"""
india_map.get_root().html.add_child(folium.Element(title_html))

# Add a legend
legend_html = """
<div style="
position: fixed;
bottom: 30px;
left: 30px;
width: 160px;
background-color: white;
border: 2px solid grey;
z-index: 9999;
padding: 10px;
font-size: 14px;
">
<b>Rainfall Categories</b><br>
<span style="color:green;">●</span> Normal<br>
<span style="color:#d4aa00;">●</span> Elevated<br>
<span style="color:orange;">●</span> Very High<br>
<span style="color:red;">●</span> Extreme
</div>
"""
india_map.get_root().html.add_child(folium.Element(legend_html))

# Save the interactive map
output_path = "data/processed/india_rainfall_risk_map.html"
india_map.save(output_path)

print("India rainfall risk classification map created successfully!")
print(f"Saved at: {output_path}")
