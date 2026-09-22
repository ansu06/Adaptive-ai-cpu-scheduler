import pandas as pd

from scheduler.schedulers import (
    fcfs,
    sjf,
    srtf,
    round_robin_metrics,
    summarize
)


def convert_to_processes(df):
    """
    Convert dataset rows into the format expected
    by Sayan's scheduler functions.
    """

    processes = []

    for _, row in df.iterrows():
        processes.append({
            "pid": f"P{int(row['Job Id'])}",
            "arrival": float(row["Arrival Time"]),
            "burst": int(row["Burst time"])
        })

    return processes


def evaluate_workload(df):

    processes = convert_to_processes(df)

    results = []

    # -------------------------
    # FCFS
    # -------------------------
    order, metrics, trace = fcfs(processes)

    fcfs_result = summarize(
        "FCFS",
        order,
        metrics
    )

    results.append(fcfs_result)

    # -------------------------
    # SJF
    # -------------------------
    order, metrics, trace = sjf(processes)

    sjf_result = summarize(
        "SJF",
        order,
        metrics
    )

    results.append(sjf_result)

    # -------------------------
    # SRTF
    # -------------------------
    order, metrics, trace = srtf(processes)

    srtf_result = summarize(
        "SRTF",
        order,
        metrics
    )

    results.append(srtf_result)

    # -------------------------
    # Round Robin
    # -------------------------
    order, metrics, trace = round_robin_metrics(
        processes,
        quantum=4
    )

    rr_result = summarize(
        "Round Robin",
        order,
        metrics
    )

    results.append(rr_result)

    return results


if __name__ == "__main__":

    # Load dataset
    df = pd.read_csv("process_data.csv")

    # Use first 10 processes as a test workload
    workload = df.head(10)

    results = evaluate_workload(workload)

    print("\nScheduler Performance")
    print("=" * 60)

    for result in results:

        print(f"\n{result['scheduler']}")

        print(
            f"Average Waiting Time: "
            f"{result['avg_waiting']:.2f}"
        )

        print(
            f"Average Turnaround Time: "
            f"{result['avg_turnaround']:.2f}"
        )

        print(
            f"Average Response Time: "
            f"{result['avg_response']:.2f}"
        )

    # Find scheduler with minimum waiting time
    best = min(
        results,
        key=lambda x: x["avg_waiting"]
    )

    print("\n" + "=" * 60)

    print(
        f"BEST SCHEDULER: "
        f"{best['scheduler']}"
    )

    print(
        f"Lowest Average Waiting Time: "
        f"{best['avg_waiting']:.2f}"
    )