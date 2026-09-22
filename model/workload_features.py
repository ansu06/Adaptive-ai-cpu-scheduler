import pandas as pd


def extract_workload_features(df):
    """
    Extract workload-level features from process data.
    """

    features = {
        "num_processes": len(df),

        "avg_burst": df["Burst time"].mean(),
        "std_burst": df["Burst time"].std(),

        "avg_arrival": df["Arrival Time"].mean(),
        "std_arrival": df["Arrival Time"].std(),

        "avg_resources": df["Resources"].mean(),

        "preemptive_ratio": df["Prremptive"].mean()
    }

    return features


if __name__ == "__main__":

    df = pd.read_csv("process_data.csv")

    # Take first 10 processes as one sample workload
    workload = df.head(10)

    features = extract_workload_features(workload)

    print("Workload Features:")
    for key, value in features.items():
        print(f"{key}: {value:.4f}")