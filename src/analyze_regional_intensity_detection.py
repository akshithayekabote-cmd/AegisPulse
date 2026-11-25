
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. LOAD REGIONAL INTENSITY RESULTS
# --------------------------------------------------

df = pd.read_csv(
    "data/processed/regional_rainfall_intensity_2024.csv"
)

print("Regional rainfall intensity data loaded.")

# --------------------------------------------------
# 2. DEFINE REGIONS AND INTENSITY CATEGORIES
# --------------------------------------------------

regions = [
    "JAMMU & KASHMIR",
    "PUNJAB",
    "GANGETIC WEST BENGAL",
    "CHHATTISGARH",
    "ORISSA",
    "HIMACHAL PRADESH",
    "JHARKHAND",
    "MARATHWADA"
]

intensities = [
    "VERY_LIGHT",
    "LIGHT",
    "MODERATE",
    "HEAVY",
    "VERY_HEAVY",
    "EXTREMELY_HEAVY"
]

# --------------------------------------------------
# 3. CALCULATE INTENSITY-WISE DETECTION PERFORMANCE
# --------------------------------------------------

results = []

for region in regions:

    regional = df[
        df["SUBDIVISION"] == region
    ]

    for intensity in intensities:

        intensity_data = regional[
            regional["INTENSITY"] == intensity
        ]

        tp = int(
            intensity_data.loc[
                intensity_data["ERROR_TYPE"] == "TP",
                "RECORDS"
            ].sum()
        )

        fn = int(
            intensity_data.loc[
                intensity_data["ERROR_TYPE"] == "FN",
                "RECORDS"
            ].sum()
        )

        total_events = tp + fn

        detection_rate = (
            tp / total_events * 100
            if total_events > 0
            else np.nan
        )

        results.append({
            "SUBDIVISION": region,
            "INTENSITY": intensity,
            "TRUE_POSITIVES": tp,
            "FALSE_NEGATIVES": fn,
            "TOTAL_RAIN_EVENTS": total_events,
            "DETECTION_RATE_PERCENT": detection_rate
        })

summary = pd.DataFrame(results)

# --------------------------------------------------
# 4. SAVE RESULTS
# --------------------------------------------------

output_path = (
    "data/processed/regional_intensity_detection_performance.csv"
)

summary.to_csv(output_path, index=False)

# --------------------------------------------------
# 5. DISPLAY RESULTS
# --------------------------------------------------

print("\nAEGISPULSE - REGIONAL INTENSITY DETECTION PERFORMANCE")
print("=" * 100)

for region in regions:

    regional = summary[
        summary["SUBDIVISION"] == region
    ]

    print(f"\n{region}")
    print("-" * 100)

    print(
        regional[
            [
                "INTENSITY",
                "TRUE_POSITIVES",
                "FALSE_NEGATIVES",
                "TOTAL_RAIN_EVENTS",
                "DETECTION_RATE_PERCENT"
            ]
        ].to_string(
            index=False,
            formatters={
                "DETECTION_RATE_PERCENT": lambda x:
                    f"{x:.2f}%" if pd.notna(x) else "N/A"
            }
        )
    )

# --------------------------------------------------
# 6. OVERALL INTENSITY PERFORMANCE
# --------------------------------------------------

overall = (
    summary.groupby("INTENSITY", sort=False)
    .agg(
        TRUE_POSITIVES=("TRUE_POSITIVES", "sum"),
        FALSE_NEGATIVES=("FALSE_NEGATIVES", "sum"),
        TOTAL_RAIN_EVENTS=("TOTAL_RAIN_EVENTS", "sum")
    )
    .reset_index()
)

overall["DETECTION_RATE_PERCENT"] = (
    overall["TRUE_POSITIVES"]
    / overall["TOTAL_RAIN_EVENTS"] * 100
)

print("\nOVERALL INTENSITY DETECTION PERFORMANCE")
print("=" * 100)

print(
    overall.to_string(
        index=False,
        formatters={
            "DETECTION_RATE_PERCENT": lambda x: f"{x:.2f}%"
        }
    )
)

print("\nSaved results to:")
print(output_path)

print("\nREGIONAL INTENSITY DETECTION ANALYSIS COMPLETED")
