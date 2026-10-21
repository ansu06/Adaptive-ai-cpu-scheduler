import pandas as pd
from model.predict import extract_features as extract_workload_features, FEATURES

__all__ = ["extract_workload_features", "FEATURES"]

if __name__ == "__main__":
    df = pd.read_csv("process_data.csv")
    workload = df.head(10)
    features = extract_workload_features(workload)

    print("Workload Features (11 Features):")
    for key, value in features.items():
        print(f"{key}: {value:.4f}")