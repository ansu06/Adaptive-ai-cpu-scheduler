import pandas as pd
import numpy as np

from scheduler.schedulers import (
    fcfs,
    sjf,
    srtf,
    round_robin_metrics,
    summarize
)


# --------------------------------------------------
# Convert dataframe to Sayan's process format
# --------------------------------------------------

def convert_to_processes(df):

    processes = []

    for _, row in df.iterrows():

        processes.append({
            "pid": f"P{int(row['Job Id'])}",
            "arrival": float(row["Arrival Time"]),
            "burst": int(row["Burst time"])
        })

    return processes


# --------------------------------------------------
# Extract workload features
# --------------------------------------------------

def extract_features(df):

    return {
        "num_processes": len(df),

        "avg_burst": df["Burst time"].mean(),
        "std_burst": df["Burst time"].std(),

        "avg_arrival": df["Arrival Time"].mean(),
        "std_arrival": df["Arrival Time"].std(),

        "avg_resources": df["Resources"].mean(),

        "preemptive_ratio": df["Prremptive"].mean()
    }


# --------------------------------------------------
# Evaluate all schedulers
# --------------------------------------------------

def evaluate_workload(df):

    processes = convert_to_processes(df)

    results = []

    # FCFS
    order, metrics, trace = fcfs(processes)

    results.append(
        summarize(
            "FCFS",
            order,
            metrics
        )
    )

    # SJF
    order, metrics, trace = sjf(processes)

    results.append(
        summarize(
            "SJF",
            order,
            metrics
        )
    )

    # SRTF
    order, metrics, trace = srtf(processes)

    results.append(
        summarize(
            "SRTF",
            order,
            metrics
        )
    )

    # Round Robin
    order, metrics, trace = round_robin_metrics(
        processes,
        quantum=4
    )

    results.append(
        summarize(
            "Round Robin",
            order,
            metrics
        )
    )

    return results


# --------------------------------------------------
# Generate training dataset
# --------------------------------------------------

def generate_training_data(
    input_file="process_data.csv",
    output_file="model/training_data.csv",
    num_workloads=200,
    processes_per_workload=10
):

    df = pd.read_csv(input_file)

    training_rows = []

    rng = np.random.default_rng(42)

    print("Generating training workloads...")
    print(f"Total workloads: {num_workloads}")
    print(f"Processes per workload: {processes_per_workload}")
    print()

    for i in range(num_workloads):

        # Randomly select processes
        workload = df.sample(
            n=processes_per_workload,
            replace=False,
            random_state=int(rng.integers(0, 1_000_000))
        )

        # ------------------------------------------
        # Feature extraction
        # ------------------------------------------

        features = extract_features(workload)

        # ------------------------------------------
        # Run schedulers
        # ------------------------------------------

        results = evaluate_workload(workload)

        # ------------------------------------------
        # Find best scheduler
        # ------------------------------------------

        best = min(
            results,
            key=lambda x: x["avg_waiting"]
        )

        # ------------------------------------------
        # Create training row
        # ------------------------------------------

        row = features.copy()

        row["fcfs_waiting"] = next(
            r["avg_waiting"]
            for r in results
            if r["scheduler"] == "FCFS"
        )

        row["sjf_waiting"] = next(
            r["avg_waiting"]
            for r in results
            if r["scheduler"] == "SJF"
        )

        row["srtf_waiting"] = next(
            r["avg_waiting"]
            for r in results
            if r["scheduler"] == "SRTF"
        )

        row["rr_waiting"] = next(
            r["avg_waiting"]
            for r in results
            if r["scheduler"] == "Round Robin"
        )

        row["best_scheduler"] = best["scheduler"]

        training_rows.append(row)

        if (i + 1) % 20 == 0:
            print(
                f"Generated {i + 1}/{num_workloads} workloads"
            )

    # ----------------------------------------------
    # Create dataframe
    # ----------------------------------------------

    training_df = pd.DataFrame(training_rows)

    training_df.to_csv(
        output_file,
        index=False
    )

    print()
    print("=" * 60)
    print("TRAINING DATASET CREATED")
    print("=" * 60)

    print(f"Rows: {len(training_df)}")
    print(f"Columns: {len(training_df.columns)}")

    print()
    print("Scheduler label distribution:")

    print(
        training_df["best_scheduler"]
        .value_counts()
    )

    print()
    print(f"Saved to: {output_file}")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    generate_training_data()