
import pandas as pd
import numpy as np
import os

folder = "data/processed/features"

files = {
    "VALIDATION": os.path.join(folder, "validation.csv.gz"),
    "TEST": os.path.join(folder, "test.csv.gz")
}

def evaluate_baseline(df, dataset_name):

    actual = df["RAIN_OCCURRED"].to_numpy()

    # Predict tomorrow's rainfall using today's rainfall
    predicted = (df["RAINFALL"].to_numpy() > 0).astype(np.int8)

    tp = np.sum((actual == 1) & (predicted == 1))
    tn = np.sum((actual == 0) & (predicted == 0))
    fp = np.sum((actual == 0) & (predicted == 1))
    fn = np.sum((actual == 1) & (predicted == 0))

    accuracy = (tp + tn) / len(actual)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    print(f"\n{dataset_name} BASELINE RESULTS")
    print("=" * 50)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))

    print("\nConfusion Matrix:")
    print("True Positives:", tp)
    print("True Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)


print("AEGISPULSE - PERSISTENCE BASELINE")
print("=" * 60)

for name, path in files.items():

    print(f"\nLoading {name} dataset...")

    df = pd.read_csv(path)

    evaluate_baseline(df, name)

    del df

print("\nBASELINE EVALUATION COMPLETED")
print("=" * 60)
