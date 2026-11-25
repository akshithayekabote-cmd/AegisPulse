
import streamlit as st
import pandas as pd
import pydeck as pdk
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AegisPulse | Rainfall Analytics",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent

PREDICTION_FILE = (
    BASE_DIR / "data" / "processed" / "predictions"
    / "aegispulse_next_day_predictions.csv.gz"
)

MAPPING_FILE = (
    BASE_DIR / "data" / "processed"
    / "india_grid_subdivision_mapping.csv"
)

RISK_LEVELS = ["Low", "Moderate", "High", "Very High"]

RISK_COLORS = {
    "Low": [65, 190, 130],
    "Moderate": [240, 190, 70],
    "High": [240, 120, 55],
    "Very High": [220, 65, 80]
}

# =========================================================
# DASHBOARD STYLING
# =========================================================

st.markdown("""
<style>

/* Main background */

.stApp {
    background-color: #101B18;
    color: #E8E8D9;
}

/* Page spacing */

section.main > div.block-container {
    max-width: 100%;
    padding: 1.3rem 1.7rem 1.5rem 1.7rem;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #182720;
}

/* Headings */

h1 {
    color: #55DDA0 !important;
    font-size: 36px !important;
    font-weight: 800 !important;
}

h2 {
    color: #E8E8D9 !important;
    font-size: 25px !important;
    font-weight: 700 !important;
}

h3 {
    color: #55DDA0 !important;
    font-size: 20px !important;
    font-weight: 700 !important;
}

/* General text */

p, label, li {
    color: #D2DDD4;
    font-size: 14px;
    line-height: 1.5;
}

/* Metrics */

[data-testid="stMetric"] {
    background-color: #1C3029;
    border: 1px solid #376B50;
    border-radius: 11px;
    padding: 16px 18px;
    min-height: 105px;
}

[data-testid="stMetricLabel"] {
    color: #C1D3C5 !important;
    font-size: 13px !important;
}

[data-testid="stMetricValue"] {
    color: #55DDA0 !important;
    font-size: 30px !important;
    font-weight: 700 !important;
}

/* Tables */

[data-testid="stDataFrame"] {
    border: 1px solid #376B50;
    border-radius: 9px;
}

/* Captions */

.stCaption {
    color: #B5C7BA !important;
    font-size: 12px !important;
}

/* Dividers */

hr {
    border-color: #355344 !important;
    margin-top: 20px;
    margin-bottom: 20px;
}

/* Historical disclaimer */

[data-testid="stAlert"] {
    background-color: #FFF8E5 !important;
    border: 1px solid #E6C76B !important;
    border-radius: 9px !important;
    padding: 12px 16px !important;
}

[data-testid="stAlert"] p {
    color: #493B1E !important;
    font-size: 14px !important;
    line-height: 1.65 !important;
    font-weight: 500 !important;
}

/* Responsive layout */

@media (max-width: 768px) {

    section.main > div.block-container {
        padding: 1rem;
    }

    h1 {
        font-size: 29px !important;
    }

    h2 {
        font-size: 21px !important;
    }

}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.title("🌧️ AEGISPULSE")

st.caption(
    "PREDICT EARLY. PROTECT TOMORROW. | "
    "AI-POWERED RAINFALL ANALYTICS"
)

st.divider()

# =========================================================
# LOAD PREDICTION DATA
# =========================================================

if not PREDICTION_FILE.exists():
    st.error(
        "Prediction file not found. "
        "Please run src/predict_next_day.py first."
    )
    st.stop()

df = pd.read_csv(
    PREDICTION_FILE,
    compression="gzip"
)

required_columns = [
    "LATITUDE",
    "LONGITUDE",
    "INPUT_DATE",
    "PREDICTION_DATE",
    "PREDICTED_RAIN_PROBABILITY",
    "PREDICTED_RAIN",
    "RISK_LEVEL"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(f"Missing prediction columns: {missing_columns}")
    st.stop()

# Convert numeric columns

numeric_columns = [
    "LATITUDE",
    "LONGITUDE",
    "PREDICTED_RAIN_PROBABILITY",
    "PREDICTED_RAIN"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.dropna(
    subset=numeric_columns
).copy()

if df.empty:
    st.error("No valid prediction records found.")
    st.stop()

# Rebuild risk labels from the model probability thresholds.
# This guarantees that labels and map colours always agree.
df["PREDICTED_RAIN_PROBABILITY"] = df[
    "PREDICTED_RAIN_PROBABILITY"
].clip(0, 1)

df["RISK_LEVEL"] = pd.cut(
    df["PREDICTED_RAIN_PROBABILITY"],
    bins=[-float("inf"), 0.20, 0.40, 0.60, float("inf")],
    labels=RISK_LEVELS,
    right=False
).astype(str)

prediction_date = str(df["PREDICTION_DATE"].iloc[0])
input_date = str(df["INPUT_DATE"].iloc[0])

# =========================================================
# FORECAST INFORMATION
# =========================================================

st.subheader("📅 Forecast Information")

date_col1, date_col2 = st.columns(2)

with date_col1:
    st.markdown(
        f"**Prediction date:** {prediction_date}"
    )

with date_col2:
    st.markdown(
        f"**Last observed rainfall date:** {input_date}"
    )

st.warning(
    "Historical demonstration forecast only. "
    "The source dataset ends on 31 December 2024. "
    "This model output predicts 1 January 2025, not current weather. "
    "Probability categories are not official IMD warnings "
    "or flood-risk assessments."
)

# =========================================================
# SIDEBAR FILTERS
# =========================================================

with st.sidebar:

    st.header("⚙️ Dashboard Controls")

    selected_levels = st.multiselect(
        "Filter by probability level",
        RISK_LEVELS,
        default=RISK_LEVELS
    )

    st.divider()

    st.subheader("MODEL INFORMATION")

    st.write("Model: AegisPulse V3")
    st.write("Coverage: India")
    st.write("Grid resolution: 0.25°")

filtered_df = df[
    df["RISK_LEVEL"].isin(selected_levels)
].copy()

# =========================================================
# PREDICTION SUMMARY
# =========================================================

st.divider()

st.subheader("📊 Prediction Summary")

total_locations = len(df)

predicted_rainy = int(
    (df["PREDICTED_RAIN"] == 1).sum()
)

average_probability = (
    df["PREDICTED_RAIN_PROBABILITY"].mean() * 100
)

high_probability = int(
    df["RISK_LEVEL"].isin(
        ["High", "Very High"]
    ).sum()
)

k1, k2, k3, k4 = st.columns(4, gap="medium")

k1.metric(
    "TOTAL GRID LOCATIONS",
    f"{total_locations:,}"
)

k2.metric(
    "PREDICTED RAINY",
    f"{predicted_rainy:,}"
)

k3.metric(
    "AVERAGE PROBABILITY",
    f"{average_probability:.1f}%"
)

k4.metric(
    "HIGH / VERY HIGH",
    f"{high_probability:,}"
)

# =========================================================
# INDIA MAP AND PROBABILITY DISTRIBUTION
# =========================================================

st.divider()

map_column, chart_column = st.columns(
    [1.5, 1],
    gap="large"
)

# ---------------- INDIA MAP ----------------

with map_column:

    st.subheader("🗺️ India Rainfall Map")

    st.caption(
        "Predicted rainfall occurrence probability "
        "across India's 0.25° grid locations."
    )

    map_df = filtered_df.copy()

    if not map_df.empty:

        # Create one map layer per risk category. Each layer uses a
        # fixed RGBA colour, avoiding inconsistent data-driven colours.
        map_layers = []

        for level in RISK_LEVELS:
            category_df = map_df[
                map_df["RISK_LEVEL"] == level
            ].copy()

            if category_df.empty:
                continue

            rgba_color = RISK_COLORS[level] + [235]

            category_layer = pdk.Layer(
                "ScatterplotLayer",
                data=category_df,
                get_position="[LONGITUDE, LATITUDE]",
                get_fill_color=rgba_color,
                get_line_color=[35, 50, 40, 210],
                get_radius=5000,
                radius_min_pixels=3,
                radius_max_pixels=9,
                pickable=True,
                auto_highlight=True,
                stroked=True,
                line_width_min_pixels=1
            )

            map_layers.append(category_layer)

        map_view = pdk.ViewState(
            latitude=22.5,
            longitude=82,
            zoom=3.0,
            pitch=0
        )

        deck = pdk.Deck(
            layers=map_layers,
            initial_view_state=map_view,
            map_style="light",
            tooltip={
                "html": (
                    "<b>Rainfall Prediction</b><br/>"
                    "Latitude: {LATITUDE}<br/>"
                    "Longitude: {LONGITUDE}<br/>"
                    "Probability: {PREDICTED_RAIN_PROBABILITY}<br/>"
                    "Level: {RISK_LEVEL}"
                )
            }
        )

        st.pydeck_chart(
            deck,
            use_container_width=True,
            height=450
        )

    else:

        st.info(
            "No locations match the selected probability filters."
        )

    st.caption(
        "🟢 Low  |  🟡 Moderate  |  "
        "🟠 High  |  🔴 Very High"
    )
# ---------------- PROBABILITY CHART ----------------

with chart_column:

    st.subheader("📈 Probability Distribution")

    st.caption(
        "Number of grid locations in each probability category."
    )

    risk_counts = (
        df["RISK_LEVEL"]
        .value_counts()
        .reindex(RISK_LEVELS, fill_value=0)
    )

    # Keep the chart categories in the same order as the map legend.
    chart_data = pd.DataFrame({
        "Probability Level": pd.Categorical(
            RISK_LEVELS,
            categories=RISK_LEVELS,
            ordered=True
        ),
        "Grid Locations": [int(risk_counts[level]) for level in RISK_LEVELS]
    })

    st.bar_chart(
        chart_data,
        x="Probability Level",
        y="Grid Locations",
        color="#55DDA0",
        height=450
    )

    st.markdown("**Probability Categories**")

    st.caption("🟢 Low: Below 20%")
    st.caption("🟡 Moderate: 20% to below 40%")
    st.caption("🟠 High: 40% to below 60%")
    st.caption("🔴 Very High: 60% and above")

# =========================================================
# TOP LOCATIONS AND REGIONAL ANALYSIS
# =========================================================

st.divider()

top_column, region_column = st.columns(
    2,
    gap="large"
)

# ---------------- TOP PREDICTION LOCATIONS ----------------

with top_column:

    st.subheader("📍 Top Prediction Locations")

    st.caption(
        "Locations with the highest predicted probabilities."
    )

    top_locations = df.nlargest(
        8,
        "PREDICTED_RAIN_PROBABILITY"
    ).copy()

    top_locations["Probability (%)"] = (
        top_locations["PREDICTED_RAIN_PROBABILITY"] * 100
    ).round(2)

    top_locations = top_locations.rename(
        columns={
            "LATITUDE": "Latitude",
            "LONGITUDE": "Longitude",
            "RISK_LEVEL": "Level"
        }
    )

    st.dataframe(
        top_locations[
            [
                "Latitude",
                "Longitude",
                "Probability (%)",
                "Level"
            ]
        ],
        use_container_width=True,
        hide_index=True,
        height=350
    )

# ---------------- REGIONAL ANALYSIS ----------------

with region_column:

    st.subheader("🇮🇳 Regional Analysis")

    st.caption(
        "Subdivision-level rainfall prediction summary."
    )

    if MAPPING_FILE.exists():

        mapping = pd.read_csv(MAPPING_FILE)

        required_mapping_columns = [
            "LATITUDE",
            "LONGITUDE",
            "SUBDIVISION"
        ]

        if all(
            column in mapping.columns
            for column in required_mapping_columns
        ):

            mapping["LATITUDE"] = pd.to_numeric(
                mapping["LATITUDE"],
                errors="coerce"
            )

            mapping["LONGITUDE"] = pd.to_numeric(
                mapping["LONGITUDE"],
                errors="coerce"
            )

            mapping = mapping.dropna(
                subset=["LATITUDE", "LONGITUDE"]
            )

            mapping["SUBDIVISION"] = (
                mapping["SUBDIVISION"]
                .astype(str)
                .str.strip()
            )

            mapping = mapping.drop_duplicates(
                subset=["LATITUDE", "LONGITUDE"]
            )

            regional_data = df.merge(
                mapping[
                    [
                        "LATITUDE",
                        "LONGITUDE",
                        "SUBDIVISION"
                    ]
                ],
                on=["LATITUDE", "LONGITUDE"],
                how="left"
            )

            regional_data["SUBDIVISION"] = (
                regional_data["SUBDIVISION"]
                .fillna("UNMAPPED")
            )

            regional_summary = (
                regional_data.groupby("SUBDIVISION")
                .agg(
                    Grid_Points=("PREDICTED_RAIN", "size"),
                    Rainy_Points=("PREDICTED_RAIN", "sum"),
                    Average_Probability=(
                        "PREDICTED_RAIN_PROBABILITY",
                        "mean"
                    )
                )
                .reset_index()
            )

            regional_summary["Average_Probability"] = (
                regional_summary["Average_Probability"] * 100
            ).round(2)

            regional_summary = regional_summary[
                regional_summary["SUBDIVISION"] != "UNMAPPED"
            ]

            regional_summary = regional_summary.sort_values(
                "Average_Probability",
                ascending=False
            ).head(8)

            regional_summary = regional_summary.rename(
                columns={
                    "SUBDIVISION": "Subdivision",
                    "Grid_Points": "Grid Points",
                    "Rainy_Points": "Rainy Points",
                    "Average_Probability": "Avg. Probability (%)"
                }
            )

            st.dataframe(
                regional_summary,
                use_container_width=True,
                hide_index=True,
                height=350
            )

            mapped_count = int(
                (regional_data["SUBDIVISION"] != "UNMAPPED").sum()
            )

            unmapped_count = (
                len(df) - mapped_count
            )

            coverage = (
                mapped_count / len(df) * 100
                if len(df) > 0 else 0
            )

            st.caption(
                f"Mapped: {mapped_count:,} | "
                f"Unmapped: {unmapped_count:,} | "
                f"Coverage: {coverage:.2f}%"
            )

        else:

            st.warning(
                "Mapping file must contain LATITUDE, LONGITUDE "
                "and SUBDIVISION columns."
            )

    else:

        st.warning(
            "Regional mapping file not found."
        )

# =========================================================
# HISTORICAL DATA DISCLAIMER
# =========================================================

st.divider()

st.subheader("⚠️ Historical Data Disclaimer")

st.warning(
    "The source rainfall dataset ends on 31 December 2024. "
    "This model output predicts 1 January 2025, not current weather. "
    "Probability categories are not official IMD warnings "
    "or flood-risk assessments."
)

# =========================================================
# CENTERED FOOTER
# =========================================================

st.divider()

st.markdown(
    "<div style='width:100%; text-align:center;'>"
    "<h3 style='color:#55DDA0; margin-bottom:10px;'>"
    "🌧️ AEGISPULSE</h3>"
    "<p style='text-align:center; color:#E8E8D9;'>"
    "<b>Predict Early. Protect Tomorrow.</b></p>"
    "<p style='text-align:center; color:#B5C7BA; font-size:13px;'>"
    "AI-Powered Rainfall Occurrence Analytics | Model V3</p>"
    "<p style='text-align:center; color:#B5C7BA; font-size:13px;'>"
    "India | 0.25° Rainfall Grid</p>"
    "</div>",
    unsafe_allow_html=True
)
