
import pandas as pd

# --------------------------------------------------
# 1. LOAD VALIDATION AND TEST RESULTS
# --------------------------------------------------

validation = pd.read_csv(
    "data/processed/imd_regional_performance.csv"
)

test = pd.read_csv(
    "data/processed/imd_regional_test_performance.csv"
)

# --------------------------------------------------
# 2. VALIDATE REQUIRED COLUMNS
# --------------------------------------------------

validation_required = [
    "SUBDIVISION",
    "RECORDS",
    "BASELINE_F1",
    "V2_F1"
]

test_required = [
    "SUBDIVISION",
    "RECORDS",
    "F1_SCORE"
]

for column in validation_required:
    if column not in validation.columns:
        raise ValueError(
            f"Missing validation column: {column}"
        )

for column in test_required:
    if column not in test.columns:
        raise ValueError(
            f"Missing test column: {column}"
        )

# Check duplicate subdivisions
if validation["SUBDIVISION"].duplicated().any():
    raise ValueError(
        "Duplicate subdivisions found in validation results."
    )

if test["SUBDIVISION"].duplicated().any():
    raise ValueError(
        "Duplicate subdivisions found in test results."
    )

# --------------------------------------------------
# 3. SELECT REQUIRED COLUMNS
# --------------------------------------------------

validation = validation[
    ["SUBDIVISION", "RECORDS", "BASELINE_F1", "V2_F1"]
].copy()

test = test[
    ["SUBDIVISION", "RECORDS", "F1_SCORE"]
].copy()

# Rename columns
validation = validation.rename(columns={
    "RECORDS": "VALIDATION_RECORDS",
    "V2_F1": "VALIDATION_F1"
})

test = test.rename(columns={
    "RECORDS": "TEST_RECORDS",
    "F1_SCORE": "TEST_F1"
})

# --------------------------------------------------
# 4. MERGE REGIONAL RESULTS
# --------------------------------------------------

comparison = pd.merge(
    validation,
    test,
    on="SUBDIVISION",
    how="inner",
    validate="one_to_one"
)

# Check regional coverage
if len(comparison) != len(validation):
    raise ValueError(
        "Some validation subdivisions are missing from test results."
    )

if len(comparison) != len(test):
    raise ValueError(
        "Some test subdivisions are missing from validation results."
    )

if comparison[
    ["BASELINE_F1", "VALIDATION_F1", "TEST_F1"]
].isna().any().any():
    raise ValueError(
        "Missing F1 values found in regional results."
    )

# --------------------------------------------------
# 5. CALCULATE PERFORMANCE CHANGES
# --------------------------------------------------

comparison["F1_CHANGE"] = (
    comparison["TEST_F1"]
    - comparison["VALIDATION_F1"]
)

comparison["BASELINE_IMPROVEMENT"] = (
    comparison["VALIDATION_F1"]
    - comparison["BASELINE_F1"]
)

# --------------------------------------------------
# 6. CLASSIFY REGIONAL STABILITY
# --------------------------------------------------

def classify_change(change):

    if change >= 0.03:
        return "Improved"

    elif change <= -0.03:
        return "Declined"

    else:
        return "Stable"


comparison["PERFORMANCE_STATUS"] = (
    comparison["F1_CHANGE"].apply(classify_change)
)

# --------------------------------------------------
# 7. SORT BY TEST PERFORMANCE
# --------------------------------------------------

comparison = comparison.sort_values(
    "TEST_F1",
    ascending=False
)

# --------------------------------------------------
# 8. DISPLAY RESULTS
# --------------------------------------------------

print("\nAEGISPULSE - REGIONAL VALIDATION VS TEST")
print("=" * 100)

print("Validation subdivisions:", len(validation))
print("Test subdivisions:", len(test))
print("Matched subdivisions:", len(comparison))

print("\nREGIONAL PERFORMANCE COMPARISON")
print("=" * 110)

print(
    comparison.to_string(
        index=False,
        formatters={
            "BASELINE_F1": "{:.4f}".format,
            "VALIDATION_F1": "{:.4f}".format,
            "TEST_F1": "{:.4f}".format,
            "F1_CHANGE": "{:+.4f}".format,
            "BASELINE_IMPROVEMENT": "{:+.4f}".format
        }
    )
)

# --------------------------------------------------
# 9. OVERALL STABILITY SUMMARY
# --------------------------------------------------

print("\nREGIONAL STABILITY SUMMARY")
print("=" * 85)

print("Subdivisions compared:", len(comparison))

print(
    "Improved:",
    (comparison["PERFORMANCE_STATUS"] == "Improved").sum()
)

print(
    "Stable:",
    (comparison["PERFORMANCE_STATUS"] == "Stable").sum()
)

print(
    "Declined:",
    (comparison["PERFORMANCE_STATUS"] == "Declined").sum()
)

# --------------------------------------------------
# 10. SAVE COMPARISON
# --------------------------------------------------

output_path = (
    "data/processed/regional_validation_test_comparison.csv"
)

comparison.to_csv(
    output_path,
    index=False
)

print("\nSaved comparison to:")
print(output_path)

print("\nREGIONAL COMPARISON COMPLETED")
