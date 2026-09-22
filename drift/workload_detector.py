from pathlib import Path
import pandas as pd
import numpy as np

from model.predict import extract_features


FEATURES = [
    "num_processes",
    "avg_burst",
    "std_burst",
    "min_burst",
    "max_burst",
    "burst_range",
    "avg_arrival",
    "std_arrival",
    "avg_arrival_gap",
    "std_arrival_gap",
    "arrival_burst_corr"
]


def calculate_training_statistics(training_file):
    """
    Calculate mean and standard deviation of each
    workload feature from the training dataset.
    """

    df = pd.read_csv(training_file)

    statistics = {}

    for feature in FEATURES:
        statistics[feature] = {
            "mean": df[feature].mean(),
            "std": df[feature].std()
        }

    return statistics


def calculate_drift_score(features, statistics):
    """
    Calculate how far the current workload is from
    the training workload distribution.

    Uses standardized distance for each feature.
    """

    distances = []

    for feature in FEATURES:

        mean = statistics[feature]["mean"]
        std = statistics[feature]["std"]

        # Prevent division by zero
        if std == 0 or pd.isna(std):
            std = 1

        distance = abs(features[feature] - mean) / std

        distances.append(distance)

    # Average standardized distance
    drift_score = np.mean(distances)

    return drift_score


def detect_drift(
    current_df,
    training_file=None,
    threshold=1.5
):

    if training_file is None:

        BASE_DIR = Path(__file__).resolve().parent.parent

        training_file = (
            BASE_DIR
            / "model"
            / "training_data_diverse.csv"
        )

    current_features = extract_features(current_df)

    statistics = calculate_training_statistics(
        training_file
    )

    drift_score = calculate_drift_score(
        current_features,
        statistics
    )

    if drift_score >= threshold:
        status = "DRIFT DETECTED"
        trust_ai = False
    else:
        status = "WORKLOAD STABLE"
        trust_ai = True

    return {
        "status": status,
        "drift_score": drift_score,
        "trust_ai": trust_ai,
        "features": current_features
    }


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("WORKLOAD DRIFT DETECTION")
    print("=" * 60)

    # Load current workload
    df = pd.read_csv("process_data.csv")

    # Test using first 10 processes
    workload = df.head(10)

    result = detect_drift(workload)

    print(f"\nStatus: {result['status']}")
    print(f"Drift Score: {result['drift_score']:.4f}")
    print(f"Trust AI: {result['trust_ai']}")

    print("\nCurrent Workload Features:")

    for key, value in result["features"].items():
        print(f"{key}: {value:.4f}")