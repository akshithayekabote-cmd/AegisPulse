
import pandas as pd
import numpy as np
import os

BASE_FOLDER = "data/processed"

V2_FOLDER = os.path.join(
    BASE_FOLDER,
    "features"
)

V3_FOLDER = os.path.join(
    BASE_FOLDER,
    "features_v3"
)

SPLITS = [
    "train",
    "validation",
    "test"
]

SPATIAL_FEATURES = [
    "NEIGHBOR_RAIN_MEAN",
    "NEIGHBOR_RAIN_MAX",
    "NEIGHBOR_RAINY_COUNT",
    "NEIGHBOR_VALID_COUNT"
]

CHUNK_SIZE = 100000

print("AEGISPULSE - MODEL V3 DATASET VALIDATION")
print("=" * 70)

for split in SPLITS:

    print(f"\nVALIDATING {split.upper()}")
    print("-" * 70)

    v2_path = os.path.join(
        V2_FOLDER,
        f"{split}.csv.gz"
    )

    v3_path = os.path.join(
        V3_FOLDER,
        f"{split}_v3.csv.gz"
    )

    v2_reader = pd.read_csv(
        v2_path,
        chunksize=CHUNK_SIZE
    )

    v3_reader = pd.read_csv(
        v3_path,
        chunksize=CHUNK_SIZE
    )

    total_records = 0
    chunk_number = 0

    for v2, v3 in zip(v2_reader, v3_reader):

        chunk_number += 1

        if len(v2) != len(v3):
            raise ValueError(
                f"{split}: Record count mismatch."
            )

        if list(v3.columns[:len(v2.columns)]) != list(v2.columns):
            raise ValueError(
                f"{split}: Original column order changed."
            )

        for column in v2.columns:

            if not np.array_equal(
                v2[column].to_numpy(),
                v3[column].to_numpy()
            ):
                raise ValueError(
                    f"{split}: Original feature changed: {column}"
                )

        missing_features = [
            column
            for column in SPATIAL_FEATURES
            if column not in v3.columns
        ]

        if missing_features:
            raise ValueError(
                f"{split}: Missing spatial features: {missing_features}"
            )

        if v3[SPATIAL_FEATURES].isna().any().any():
            raise ValueError(
                f"{split}: Missing spatial values."
            )

        spatial_values = v3[
            SPATIAL_FEATURES
        ].to_numpy(dtype=float)

        if not np.isfinite(spatial_values).all():
            raise ValueError(
                f"{split}: Invalid spatial values."
            )

        if (
            v3["NEIGHBOR_RAINY_COUNT"]
            > v3["NEIGHBOR_VALID_COUNT"]
        ).any():
            raise ValueError(
                f"{split}: Invalid neighbour counts."
            )

        total_records += len(v3)

    print("Chunks validated:", chunk_number)
    print("Records validated:", f"{total_records:,}")
    print("Original features: PRESERVED")
    print("Target RAIN_OCCURRED: PRESERVED")
    print("Spatial features: VERIFIED")
    print("Missing values: 0")
    print("Invalid values: 0")
    print(f"{split.upper()} VALIDATION PASSED")

print("\n" + "=" * 70)
print("ALL MODEL V3 DATASETS VALIDATED SUCCESSFULLY")
