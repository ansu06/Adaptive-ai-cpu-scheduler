from pathlib import Path
import pandas as pd
import numpy as np
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
        avg_arrival_gap = 0.0
        std_arrival_gap = 0.0
    else:
        avg_arrival_gap = float(arrival_gaps.mean()) if not pd.isna(arrival_gaps.mean()) else 0.0
        std_arrival_gap = float(arrival_gaps.std()) if not pd.isna(arrival_gaps.std()) else 0.0

    burst_std = df["Burst time"].std() if len(df) > 1 else 0.0
    arrival_std = df["Arrival Time"].std() if len(df) > 1 else 0.0

    if burst_std == 0 or arrival_std == 0:
        arrival_burst_corr = 0.0
    else:
        corr = df["Arrival Time"].corr(df["Burst time"])
        arrival_burst_corr = float(corr) if not pd.isna(corr) else 0.0

    features = {
        "num_processes": len(df),
        "avg_burst": float(df["Burst time"].mean()) if len(df) > 0 else 0.0,
        "std_burst": float(df["Burst time"].std()) if len(df) > 1 else 0.0,
        "min_burst": float(df["Burst time"].min()) if len(df) > 0 else 0.0,
        "max_burst": float(df["Burst time"].max()) if len(df) > 0 else 0.0,
        "burst_range": float(df["Burst time"].max() - df["Burst time"].min()) if len(df) > 0 else 0.0,
        "avg_arrival": float(df["Arrival Time"].mean()) if len(df) > 0 else 0.0,
        "std_arrival": float(df["Arrival Time"].std()) if len(df) > 1 else 0.0,
        "avg_arrival_gap": avg_arrival_gap,
        "std_arrival_gap": std_arrival_gap,
        "arrival_burst_corr": arrival_burst_corr
    }

    return features


def predict_scheduler(df):
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = BASE_DIR / "model" / "best_model.pkl"
    if not MODEL_PATH.exists():
        MODEL_PATH = BASE_DIR / "model" / "random_forest_scheduler.pkl"

    model = joblib.load(MODEL_PATH)
    features = extract_features(df)
    X = pd.DataFrame([features])[FEATURES]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        classes = list(model.classes_)
        pred_idx = int(np.argmax(probabilities))
        prediction = str(classes[pred_idx])
        confidence = {str(c): float(p) for c, p in zip(classes, probabilities)}
    else:
        prediction = str(model.predict(X)[0])
        classes = list(getattr(model, "classes_", ["FCFS", "SJF", "SRTF"]))
        confidence = {str(c): (1.0 if c == prediction else 0.0) for c in classes}

    return prediction, confidence, features


if __name__ == "__main__":
    df = pd.read_csv("process_data.csv")
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