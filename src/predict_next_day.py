
import os
import numpy as np
import pandas as pd
import xarray as xr

print("AEGISPULSE - NEXT-DAY RAINFALL PREDICTION")
print("=" * 65)

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

DATA_PATH = (
    "data/processed/"
    "india_rainfall_cleaned_2015_2024.nc"
)

MAPPING_PATH = (
    "data/processed/"
    "spatial_neighbor_mapping.npz"
)

MODEL_PATH = (
    "models/"
    "historical_rainfall_model_v3.npz"
)

OUTPUT_FOLDER = "data/processed/predictions"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OUTPUT_PATH = os.path.join(
    OUTPUT_FOLDER,
    "aegispulse_next_day_predictions.csv.gz"
)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

print("\nLoading rainfall dataset...")

ds = xr.open_dataset(
    DATA_PATH,
    engine="scipy"
)

rainfall = ds["RAINFALL"].values

dates = pd.to_datetime(
    ds["TIME"].values
)

latitudes = ds["LATITUDE"].values
longitudes = ds["LONGITUDE"].values

total_days, total_locations = rainfall.shape

print("Total days:", total_days)
print("Total locations:", total_locations)
print("Last available date:", dates[-1].date())

# --------------------------------------------------
# DETERMINE PREDICTION DATE
# --------------------------------------------------

current_date = dates[-1]
prediction_date = current_date + pd.Timedelta(days=1)

print("\nInput rainfall date:", current_date.date())
print("Prediction target date:", prediction_date.date())

# --------------------------------------------------
# LOAD NEIGHBOUR MAPPING
# --------------------------------------------------

print("\nLoading spatial neighbour mapping...")

mapping = np.load(MAPPING_PATH)

neighbor_indices = mapping["neighbor_indices"]
neighbor_counts = mapping["neighbor_counts"]

if not np.allclose(mapping["latitudes"], latitudes):
    raise ValueError("Latitude alignment mismatch.")

if not np.allclose(mapping["longitudes"], longitudes):
    raise ValueError("Longitude alignment mismatch.")

valid_mask = neighbor_indices >= 0

actual_counts = np.sum(valid_mask, axis=1)

if not np.array_equal(actual_counts, neighbor_counts):
    raise ValueError("Neighbour count validation failed.")

print("Neighbour mapping validated.")

# --------------------------------------------------
# LOAD FROZEN MODEL V3
# --------------------------------------------------

print("\nLoading frozen Model V3...")

model = np.load(MODEL_PATH)

weights = model["weights"]
bias = float(model["bias"])
mean = model["mean"]
std = model["std"]
threshold = float(model["threshold"])

features = model["features"].tolist()

print("Model feature count:", len(features))
print("Model threshold:", threshold)

if len(features) != 15:
    raise ValueError("Unexpected Model V3 feature count.")

if len(weights) != len(features):
    raise ValueError("Model weight and feature mismatch.")

# --------------------------------------------------
# GENERATE HISTORICAL RAINFALL FEATURES
# --------------------------------------------------

print("\nGenerating rainfall features...")

current = rainfall[-1]
lag_1 = rainfall[-2]
lag_3 = rainfall[-4]
lag_7 = rainfall[-8]

rolling_3 = (
    rainfall[-1]
    + rainfall[-2]
    + rainfall[-3]
) / 3

rolling_7 = np.zeros(
    total_locations,
    dtype=np.float64
)

for offset in range(7):
    rolling_7 += rainfall[-1 - offset]

rolling_7 /= 7

rain_today = (current > 0).astype(np.float64)

# --------------------------------------------------
# GENERATE SPATIAL FEATURES
# --------------------------------------------------

print("Generating spatial rainfall features...")

neighbor_mean = np.zeros(
    total_locations,
    dtype=np.float64
)

neighbor_max = np.zeros(
    total_locations,
    dtype=np.float64
)

neighbor_rainy_count = np.zeros(
    total_locations,
    dtype=np.float64
)

neighbor_valid_count = np.zeros(
    total_locations,
    dtype=np.float64
)

for i in range(total_locations):

    valid_neighbors = neighbor_indices[i]

    valid_neighbors = valid_neighbors[
        valid_neighbors >= 0
    ]

    neighbor_rain = current[valid_neighbors]

    neighbor_mean[i] = np.mean(neighbor_rain)

    neighbor_max[i] = np.max(neighbor_rain)

    neighbor_rainy_count[i] = np.sum(
        neighbor_rain > 0
    )

    neighbor_valid_count[i] = len(
        valid_neighbors
    )

# --------------------------------------------------
# CREATE FEATURE DATAFRAME
# --------------------------------------------------

print("\nPreparing Model V3 input...")

feature_data = pd.DataFrame({
    "LATITUDE": latitudes,
    "LONGITUDE": longitudes,
    "RAINFALL": current,
    "RAIN_LAG_1": lag_1,
    "RAIN_LAG_3": lag_3,
    "RAIN_LAG_7": lag_7,
    "RAIN_ROLLING_3": rolling_3,
    "RAIN_ROLLING_7": rolling_7,
    "MONTH": prediction_date.month,
    "DAY_OF_YEAR": prediction_date.dayofyear,
    "RAIN_TODAY": rain_today,
    "NEIGHBOR_RAIN_MEAN": neighbor_mean,
    "NEIGHBOR_RAIN_MAX": neighbor_max,
    "NEIGHBOR_RAINY_COUNT": neighbor_rainy_count,
    "NEIGHBOR_VALID_COUNT": neighbor_valid_count
})

# Preserve exact training feature order
X = feature_data[features].to_numpy(
    dtype=np.float64
)

if not np.isfinite(X).all():
    raise ValueError("Invalid values found in model features.")

# --------------------------------------------------
# MODEL V3 PREDICTIONS
# --------------------------------------------------

print("Calculating rainfall probabilities...")

X_scaled = (
    X - mean
) / (std + 1e-8)

logits = X_scaled @ weights + bias

probabilities = 1 / (
    1 + np.exp(
        -np.clip(logits, -500, 500)
    )
)

predictions = (
    probabilities >= threshold
).astype(np.int8)

# --------------------------------------------------
# RISK CLASSIFICATION
# --------------------------------------------------

risk_levels = np.select(
    [
        probabilities < 0.20,
        probabilities < 0.40,
        probabilities < 0.60,
        probabilities >= 0.60
    ],
    [
        "Low",
        "Moderate",
        "High",
        "Very High"
    ],
    default="Unknown"
)

# --------------------------------------------------
# CREATE OUTPUT
# --------------------------------------------------

results = pd.DataFrame({
    "LATITUDE": latitudes,
    "LONGITUDE": longitudes,
    "INPUT_DATE": current_date.strftime("%Y-%m-%d"),
    "PREDICTION_DATE": prediction_date.strftime("%Y-%m-%d"),
    "PREDICTED_RAIN_PROBABILITY": probabilities,
    "PREDICTED_RAIN": predictions,
    "RISK_LEVEL": risk_levels
})

results.to_csv(
    OUTPUT_PATH,
    index=False,
    compression="gzip"
)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\nPREDICTION SUMMARY")
print("-" * 65)

print("Prediction date:", prediction_date.date())
print("Total locations:", len(results))

print(
    "Predicted rainy locations:",
    int(predictions.sum())
)

print(
    "Predicted non-rainy locations:",
    int(len(predictions) - predictions.sum())
)

print("\nRisk category distribution:")

print(
    results["RISK_LEVEL"].value_counts()
)

print("\nTop 10 predicted rainfall probabilities:")

print(
    results.nlargest(
        10,
        "PREDICTED_RAIN_PROBABILITY"
    )[
        [
            "LATITUDE",
            "LONGITUDE",
            "PREDICTED_RAIN_PROBABILITY",
            "RISK_LEVEL"
        ]
    ].to_string(index=False)
)

print("\nOutput saved:")
print(OUTPUT_PATH)

print("\nNEXT-DAY PREDICTION COMPLETED")
print("=" * 65)

ds.close()
