
import numpy as np

print("AEGISPULSE - NEIGHBOUR MAPPING VALIDATION")
print("=" * 65)

mapping = np.load(
    "data/processed/spatial_neighbor_mapping.npz"
)

indices = mapping["neighbor_indices"]
counts = mapping["neighbor_counts"]

total_locations = len(counts)

# Identify valid entries wherever they appear
valid_mask = indices >= 0

# Check valid neighbour indices
valid_indices = indices[valid_mask]

invalid_indices = np.sum(
    valid_indices >= total_locations
)

# Count actual valid neighbours in each row
actual_counts = np.sum(valid_mask, axis=1)

count_mismatches = np.sum(
    actual_counts != counts
)

# Check for unexpected negative values
invalid_negative_values = np.sum(indices < -1)

print("\nMAPPING SUMMARY")
print("Total locations:", total_locations)
print("Mapping shape:", indices.shape)

print("\nVALIDATION RESULTS")
print("Valid neighbour entries:", len(valid_indices))
print("Invalid positive indices:", invalid_indices)
print("Count mismatches:", count_mismatches)
print("Unexpected negative values:", invalid_negative_values)

print("\nSAMPLE MAPPINGS")

for i in range(5):
    print(
        f"Location {i}: "
        f"expected={counts[i]}, "
        f"actual={actual_counts[i]}, "
        f"indices={indices[i]}"
    )

if (
    invalid_indices == 0
    and count_mismatches == 0
    and invalid_negative_values == 0
):
    print("\nRESULT: NEIGHBOUR MAPPING VALIDATED")
else:
    print("\nRESULT: INVESTIGATION REQUIRED")

print("=" * 65)
