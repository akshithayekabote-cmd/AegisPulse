
import pandas as pd
import numpy as np
import os

BASE_FOLDER = "data/processed"

spatial_folder = os.path.join(
    BASE_FOLDER,
    "spatial_features"
)

original_folder = os.path.join(
    BASE_FOLDER,
    "features"
)

splits = ["train", "validation", "test"]

print("AEGISPULSE - SPATIAL FEATURE VALIDATION")
print("=" * 70)

for split in splits:

    print(f"\nVALIDATING {split.upper()} DATASET")
    print("-" * 70)

    spatial_path = os.path.join(
        spatial_folder,
        f"{split}_spatial.csv.gz"
    )

    original_path = os.path.join(
        original_folder,
        f"{split}.csv.gz"
    )

    spatial = pd.read_csv(spatial_path)

    original = pd.read_csv(
        original_path,
        usecols=["LATITUDE", "LONGITUDE", "MONTH", "DAY_OF_YEAR"]
    )

    print("Spatial records:", len(spatial))
    print("Original records:", len(original))

    if len(spatial) != len(original):
        raise ValueError(
            f"{split}: Record count mismatch."
        )

    # Validate spatial feature values
    feature_columns = [
        "NEIGHBOR_RAIN_MEAN",
        "NEIGHBOR_RAIN_MAX",
        "NEIGHBOR_RAINY_COUNT",
        "NEIGHBOR_VALID_COUNT"
    ]

    if spatial[feature_columns].isna().any().any():
        raise ValueError(
            f"{split}: Missing spatial feature values found."
        )

    if not np.isfinite(
        spatial[feature_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"{split}: Infinite or invalid feature values found."
        )

    if (
        spatial["NEIGHBOR_RAIN_MEAN"] < 0
    ).any():
        raise ValueError(
            f"{split}: Negative neighbouring rainfall mean."
        )

    if (
        spatial["NEIGHBOR_RAIN_MAX"] < 0
    ).any():
        raise ValueError(
            f"{split}: Negative neighbouring rainfall maximum."
        )

    if (
        spatial["NEIGHBOR_RAINY_COUNT"] < 0
    ).any():
        raise ValueError(
            f"{split}: Negative rainy neighbour count."
        )

    if not spatial["NEIGHBOR_VALID_COUNT"].between(2, 8).all():
        raise ValueError(
            f"{split}: Invalid neighbour availability count."
        )

    if (
        spatial["NEIGHBOR_RAINY_COUNT"]
        > spatial["NEIGHBOR_VALID_COUNT"]
    ).any():
        raise ValueError(
            f"{split}: Rainy neighbour count exceeds available neighbours."
        )

    # Validate geographic ordering
    if not np.allclose(
        spatial["LATITUDE"].to_numpy(),
        original["LATITUDE"].to_numpy()
    ):
        raise ValueError(
            f"{split}: Latitude ordering mismatch."
        )

    if not np.allclose(
        spatial["LONGITUDE"].to_numpy(),
        original["LONGITUDE"].to_numpy()
    ):
        raise ValueError(
            f"{split}: Longitude ordering mismatch."
        )

    # Validate date-derived calendar features
    dates = pd.to_datetime(
        spatial["DATE"],
        format="%Y-%m-%d"
    )

    if not np.array_equal(
        dates.dt.month.to_numpy(),
        original["MONTH"].to_numpy()
    ):
        raise ValueError(
            f"{split}: Month alignment mismatch."
        )

    if not np.array_equal(
        dates.dt.dayofyear.to_numpy(),
        original["DAY_OF_YEAR"].to_numpy()
    ):
        raise ValueError(
            f"{split}: Day-of-year alignment mismatch."
        )

    print("Missing values: 0")
    print("Invalid values: 0")
    print("Geographic alignment: VERIFIED")
    print("Date alignment: VERIFIED")

    print(
        "Average neighbour rainfall:",
        round(
            float(spatial["NEIGHBOR_RAIN_MEAN"].mean()),
            4
        )
    )

    print(
        "Maximum neighbour rainfall:",
        round(
            float(spatial["NEIGHBOR_RAIN_MAX"].max()),
            4
        )
    )

    print(
        "Average rainy neighbours:",
        round(
            float(spatial["NEIGHBOR_RAINY_COUNT"].mean()),
            4
        )
    )

    print(
        "Average valid neighbours:",
        round(
            float(spatial["NEIGHBOR_VALID_COUNT"].mean()),
            4
        )
    )

    print(f"{split.upper()} VALIDATION PASSED")

print("\n" + "=" * 70)
print("ALL SPATIAL DATASETS VALIDATED SUCCESSFULLY")
