
import pandas as pd
import numpy as np
import os
from itertools import zip_longest

BASE_FOLDER = "data/processed"

ORIGINAL_FOLDER = os.path.join(
    BASE_FOLDER,
    "features"
)

SPATIAL_FOLDER = os.path.join(
    BASE_FOLDER,
    "spatial_features"
)

OUTPUT_FOLDER = os.path.join(
    BASE_FOLDER,
    "features_v3"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

SPLITS = [
    "train",
    "validation",
    "test"
]

SPATIAL_COLUMNS = [
    "NEIGHBOR_RAIN_MEAN",
    "NEIGHBOR_RAIN_MAX",
    "NEIGHBOR_RAINY_COUNT",
    "NEIGHBOR_VALID_COUNT"
]

CHUNK_SIZE = 100000

print("AEGISPULSE - MODEL V3 DATASET PREPARATION")
print("=" * 70)

for split in SPLITS:

    print(f"\nPROCESSING {split.upper()} DATASET")
    print("-" * 70)

    original_path = os.path.join(
        ORIGINAL_FOLDER,
        f"{split}.csv.gz"
    )

    spatial_path = os.path.join(
        SPATIAL_FOLDER,
        f"{split}_spatial.csv.gz"
    )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        f"{split}_v3.csv.gz"
    )

    if not os.path.exists(original_path):
        raise FileNotFoundError(original_path)

    if not os.path.exists(spatial_path):
        raise FileNotFoundError(spatial_path)

    if os.path.exists(output_path):
        os.remove(output_path)

    original_reader = pd.read_csv(
        original_path,
        chunksize=CHUNK_SIZE
    )

    spatial_reader = pd.read_csv(
        spatial_path,
        chunksize=CHUNK_SIZE
    )

    total_records = 0
    chunk_number = 0

    for original, spatial in zip_longest(
        original_reader,
        spatial_reader,
        fillvalue=None
    ):

        if original is None or spatial is None:
            raise ValueError(
                f"{split}: Original and spatial datasets "
                "have different numbers of chunks."
            )

        chunk_number += 1

        if len(original) != len(spatial):
            raise ValueError(
                f"{split}: Chunk record count mismatch."
            )

        # Verify geographic alignment
        if not np.allclose(
            original["LATITUDE"].to_numpy(),
            spatial["LATITUDE"].to_numpy(),
            rtol=0,
            atol=1e-8
        ):
            raise ValueError(
                f"{split}: Latitude alignment mismatch."
            )

        if not np.allclose(
            original["LONGITUDE"].to_numpy(),
            spatial["LONGITUDE"].to_numpy(),
            rtol=0,
            atol=1e-8
        ):
            raise ValueError(
                f"{split}: Longitude alignment mismatch."
            )

        # Verify target-date alignment
        dates = pd.to_datetime(
            spatial["DATE"],
            format="%Y-%m-%d",
            errors="raise"
        )

        if not np.array_equal(
            dates.dt.month.to_numpy(),
            original["MONTH"].to_numpy()
        ):
            raise ValueError(
                f"{split}: Month alignment mismatch."
            )

        if not np.array_equal(
            dates.dt.day_of_year.to_numpy(),
            original["DAY_OF_YEAR"].to_numpy()
        ):
            raise ValueError(
                f"{split}: Day-of-year alignment mismatch."
            )

        # Validate spatial columns
        if spatial[SPATIAL_COLUMNS].isna().any().any():
            raise ValueError(
                f"{split}: Missing spatial values found."
            )

        spatial_values = spatial[
            SPATIAL_COLUMNS
        ].to_numpy(dtype=float)

        if not np.isfinite(spatial_values).all():
            raise ValueError(
                f"{split}: Invalid spatial values found."
            )

        if (
            spatial["NEIGHBOR_RAIN_MEAN"] < 0
        ).any() or (
            spatial["NEIGHBOR_RAIN_MAX"] < 0
        ).any():
            raise ValueError(
                f"{split}: Negative rainfall values found."
            )

        if (
            spatial["NEIGHBOR_RAINY_COUNT"]
            > spatial["NEIGHBOR_VALID_COUNT"]
        ).any():
            raise ValueError(
                f"{split}: Invalid rainy neighbour count."
            )

        # Add spatial features to original dataset
        for column in SPATIAL_COLUMNS:

            original[column] = (
                spatial[column].to_numpy()
            )

        # Save merged dataset
        original.to_csv(
            output_path,
            mode="a",
            index=False,
            header=(chunk_number == 1),
            compression="gzip"
        )

        total_records += len(original)

        print(
            f"Processed chunks: {chunk_number} | "
            f"Records: {total_records:,}"
        )

    original_reader.close()
    spatial_reader.close()

    print(f"\n{split.upper()} DATASET COMPLETED")
    print("Total records:", f"{total_records:,}")
    print("Output:", output_path)
    print(
        "File size (MB):",
        round(
            os.path.getsize(output_path) / (1024 ** 2),
            2
        )
    )

print("\n" + "=" * 70)
print("ALL MODEL V3 DATASETS CREATED SUCCESSFULLY")
print("=" * 70)
