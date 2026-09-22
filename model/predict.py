from pathlib import Path
import pandas as pd
import joblib

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


def extract_features(df):
    arrival_sorted = df["Arrival Time"].sort_values()

    arrival_gaps = arrival_sorted.diff().dropna()

    if len(arrival_gaps) == 0:
        avg_arrival_gap = 0
        std_arrival_gap = 0
    else:
        avg_arrival_gap = arrival_gaps.mean()
        std_arrival_gap = arrival_gaps.std()

    if df["Burst time"].std() == 0 or df["Arrival Time"].std() == 0:
        arrival_burst_corr = 0
    else:
        arrival_burst_corr = df["Arrival Time"].corr(df["Burst time"])

        if pd.isna(arrival_burst_corr):
            arrival_burst_corr = 0

    features = {
        "num_processes": len(df),
        "avg_burst": df["Burst time"].mean(),
        "std_burst": df["Burst time"].std(),
        "min_burst": df["Burst time"].min(),
        "max_burst": df["Burst time"].max(),
        "burst_range": df["Burst time"].max() - df["Burst time"].min(),
        "avg_arrival": df["Arrival Time"].mean(),
        "std_arrival": df["Arrival Time"].std(),
        "avg_arrival_gap": avg_arrival_gap,
        "std_arrival_gap": std_arrival_gap,
        "arrival_burst_corr": arrival_burst_corr
    }

    return features


def predict_scheduler(df):
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = BASE_DIR / "model" / "random_forest_scheduler.pkl"

    model = joblib.load(MODEL_PATH)

    features = extract_features(df)

    X = pd.DataFrame([features])[FEATURES]

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]
    classes = model.classes_

    confidence = dict(zip(classes, probabilities))

    return prediction, confidence, features


if __name__ == "__main__":

    df = pd.read_csv("process_data.csv")

    # Test with first 10 processes
    workload = df.head(10)

    prediction, confidence, features = predict_scheduler(workload)

    print("\n" + "=" * 60)
    print("AI SCHEDULER PREDICTION")
    print("=" * 60)

    print(f"\nRecommended Scheduler: {prediction}")

    print("\nConfidence:")
    for scheduler, probability in confidence.items():
        print(f"{scheduler}: {probability * 100:.2f}%")

    print("\nWorkload Features:")
    for key, value in features.items():
        print(f"{key}: {value:.4f}")