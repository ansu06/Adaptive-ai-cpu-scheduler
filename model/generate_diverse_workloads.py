import pandas as pd
import numpy as np

from scheduler.schedulers import (
    fcfs,
    sjf,
    srtf,
    round_robin_metrics,
    summarize
)


# ============================================================
# Convert workload to scheduler format
# ============================================================

def convert_to_processes(workload):

    processes = []

    for i, row in workload.iterrows():

        processes.append({
            "pid": f"P{int(row['Job Id'])}_{i}",
            "arrival": float(row["Arrival Time"]),
            "burst": int(row["Burst time"])
        })

    return processes


# ============================================================
# Extract workload features
# ============================================================

def extract_features(workload):

    bursts = workload["Burst time"]
    arrivals = workload["Arrival Time"]

    # Sort by arrival time for arrival-gap calculation
    sorted_arrivals = np.sort(
        arrivals.to_numpy()
    )

    if len(sorted_arrivals) > 1:
        arrival_gaps = np.diff(
            sorted_arrivals
        )

        avg_arrival_gap = arrival_gaps.mean()
        std_arrival_gap = arrival_gaps.std()
    else:
        avg_arrival_gap = 0
        std_arrival_gap = 0

    # Relationship between arrival time and burst time
    if (
        bursts.std() == 0
        or arrivals.std() == 0
    ):
        arrival_burst_corr = 0

    else:
        arrival_burst_corr = (
            arrivals.corr(bursts)
        )

    return {
        "num_processes": len(workload),

        "avg_burst": bursts.mean(),
        "std_burst": bursts.std(),
        "min_burst": bursts.min(),
        "max_burst": bursts.max(),
        "burst_range": (
            bursts.max() - bursts.min()
        ),

        "avg_arrival": arrivals.mean(),
        "std_arrival": arrivals.std(),

        "avg_arrival_gap": avg_arrival_gap,
        "std_arrival_gap": std_arrival_gap,

        "arrival_burst_corr": arrival_burst_corr
    }


# ============================================================
# Evaluate all four schedulers
# ============================================================

def evaluate_workload(workload):

    processes = convert_to_processes(workload)

    results = []

    # ---------------- FCFS ----------------

    order, metrics, trace = fcfs(processes)

    results.append(
        summarize(
            "FCFS",
            order,
            metrics
        )
    )

    # ---------------- SJF ----------------

    order, metrics, trace = sjf(processes)

    results.append(
        summarize(
            "SJF",
            order,
            metrics
        )
    )

    # ---------------- SRTF ----------------

    order, metrics, trace = srtf(processes)

    results.append(
        summarize(
            "SRTF",
            order,
            metrics
        )
    )

    # ---------------- Round Robin ----------------

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


# ============================================================
# Calculate combined performance score
# ============================================================

def calculate_scores(results):

    waiting = np.array([
        r["avg_waiting"] for r in results
    ])

    turnaround = np.array([
        r["avg_turnaround"] for r in results
    ])

    response = np.array([
        r["avg_response"] for r in results
    ])

    def normalize(values):

        minimum = values.min()
        maximum = values.max()

        if maximum == minimum:
            return np.zeros(len(values))

        return (values - minimum) / (maximum - minimum)

    normalized_waiting = normalize(waiting)
    normalized_turnaround = normalize(turnaround)
    normalized_response = normalize(response)

    # Weighted overall performance
    scores = (
        0.4 * normalized_waiting
        + 0.3 * normalized_turnaround
        + 0.3 * normalized_response
    )

    for i, result in enumerate(results):

        result["performance_score"] = scores[i]

    return results


# ============================================================
# Generate one workload
# ============================================================

def generate_workload(
    source_df,
    rng,
    scenario,
    num_processes
):

    # Randomly select source jobs
    selected = source_df.sample(
        n=num_processes,
        replace=False,
        random_state=int(
            rng.integers(0, 1_000_000)
        )
    ).copy()

    bursts = selected["Burst time"].to_numpy()
    arrivals = selected["Arrival Time"].to_numpy()

    # --------------------------------------------------------
    # Scenario 1: Mixed workload
    # --------------------------------------------------------

    if scenario == "mixed":

        new_bursts = bursts

        new_arrivals = np.sort(
            rng.uniform(0, 50, num_processes)
        )

    # --------------------------------------------------------
    # Scenario 2: Short-job dominant
    # --------------------------------------------------------

    elif scenario == "short_jobs":

        new_bursts = np.clip(
            bursts * 0.45,
            5,
            180
        ).astype(int)

        new_arrivals = np.sort(
            rng.uniform(0, 30, num_processes)
        )

    # --------------------------------------------------------
    # Scenario 3: Long-job dominant
    # --------------------------------------------------------

    elif scenario == "long_jobs":

        new_bursts = np.clip(
            bursts * 1.5,
            50,
            500
        ).astype(int)

        new_arrivals = np.sort(
            rng.uniform(0, 40, num_processes)
        )

    # --------------------------------------------------------
    # Scenario 4: Bimodal workload
    # --------------------------------------------------------

    elif scenario == "bimodal":

        new_bursts = np.where(
            np.arange(num_processes) % 2 == 0,
            rng.integers(10, 50, num_processes),
            rng.integers(250, 450, num_processes)
        )

        new_arrivals = np.sort(
            rng.uniform(0, 60, num_processes)
        )

    # --------------------------------------------------------
    # Scenario 5: Staggered arrivals
    # --------------------------------------------------------

    elif scenario == "staggered":

        new_bursts = bursts

        new_arrivals = np.cumsum(
            rng.uniform(5, 50, num_processes)
        )

    # --------------------------------------------------------
    # Scenario 6: Burst arrivals
    # --------------------------------------------------------

    elif scenario == "bursty":

        new_bursts = bursts

        groups = rng.integers(
            0,
            3,
            num_processes
        )

        new_arrivals = (
            groups * 50
            + rng.uniform(0, 10, num_processes)
        )

        new_arrivals = np.sort(new_arrivals)

    # --------------------------------------------------------
    # Scenario 7: Arrival-order friendly
    # --------------------------------------------------------

    elif scenario == "arrival_ordered":

        new_bursts = np.sort(bursts)

        new_arrivals = np.arange(
            num_processes
        ) * 10.0

    # --------------------------------------------------------
    # Scenario 8: Reverse arrival order
    # --------------------------------------------------------

    elif scenario == "reverse_order":

        new_bursts = np.sort(
            bursts
        )[::-1]

        new_arrivals = np.arange(
            num_processes
        ) * 10.0

    else:

        raise ValueError(
            f"Unknown scenario: {scenario}"
        )

    selected["Burst time"] = new_bursts
    selected["Arrival Time"] = new_arrivals

    return selected


# ============================================================
# Generate training dataset
# ============================================================

def generate_dataset(
    input_file="process_data.csv",
    output_file="model/training_data_diverse.csv",
    num_workloads=400
):

    source_df = pd.read_csv(input_file)

    rng = np.random.default_rng(42)

    scenarios = [
        "mixed",
        "short_jobs",
        "long_jobs",
        "bimodal",
        "staggered",
        "bursty",
        "arrival_ordered",
        "reverse_order"
    ]

    rows = []

    print()
    print("=" * 60)
    print("GENERATING DIVERSE WORKLOAD DATASET")
    print("=" * 60)

    for i in range(num_workloads):

        scenario = scenarios[
            i % len(scenarios)
        ]

        num_processes = int(
            rng.integers(8, 16)
        )

        workload = generate_workload(
            source_df,
            rng,
            scenario,
            num_processes
        )

        features = extract_features(
            workload
        )

        results = evaluate_workload(
            workload
        )

        results = calculate_scores(
            results
        )

        # Best = lowest combined score
        best = min(
            results,
            key=lambda x:
            x["performance_score"]
        )

        row = features.copy()

        # Scheduler performance
        for result in results:

            name = result["scheduler"]

            if name == "FCFS":
                prefix = "fcfs"

            elif name == "SJF":
                prefix = "sjf"

            elif name == "SRTF":
                prefix = "srtf"

            else:
                prefix = "rr"

            row[f"{prefix}_waiting"] = (
                result["avg_waiting"]
            )

            row[f"{prefix}_turnaround"] = (
                result["avg_turnaround"]
            )

            row[f"{prefix}_response"] = (
                result["avg_response"]
            )

            row[f"{prefix}_score"] = (
                result["performance_score"]
            )

        row["scenario"] = scenario

        row["best_scheduler"] = (
            best["scheduler"]
        )

        rows.append(row)

        if (i + 1) % 50 == 0:

            print(
                f"Generated "
                f"{i + 1}/{num_workloads}"
            )

    result_df = pd.DataFrame(rows)

    result_df.to_csv(
        output_file,
        index=False
    )

    print()
    print("=" * 60)
    print("DATASET CREATED")
    print("=" * 60)

    print(
        f"Rows: {len(result_df)}"
    )

    print(
        f"Columns: {len(result_df.columns)}"
    )

    print()
    print("SCENARIO DISTRIBUTION")
    print(
        result_df["scenario"]
        .value_counts()
    )

    print()
    print("SCHEDULER LABEL DISTRIBUTION")
    print(
        result_df["best_scheduler"]
        .value_counts()
    )

    print()
    print(
        f"Saved to: {output_file}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    generate_dataset()