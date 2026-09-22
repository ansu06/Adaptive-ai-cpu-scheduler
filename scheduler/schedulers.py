import os
import pandas as pd
from collections import deque

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(BASE_DIR, "..", "process_data.csv")


def load_processes(csv_path=DEFAULT_CSV, n=10):
    """Load the first n processes (sorted by arrival) from the CPU workload CSV."""
    df = pd.read_csv(csv_path)
    df = df.sort_values("Arrival Time").head(n)
    processes = []
    for _, row in df.iterrows():
        processes.append({
            "pid": f"P{int(row['Job Id'])}",
            "arrival": round(row["Arrival Time"] * 100, 2),  # scale 0-1 -> 0-100
            "burst": int(row["Burst time"])
        })
    return processes


def fcfs(processes):
    """First Come First Served: run processes strictly in arrival order."""
    procs = sorted(processes, key=lambda p: p["arrival"])
    time = 0
    order, results, trace = [], [], []
    for p in procs:
        start = max(time, p["arrival"])
        completion = start + p["burst"]
        turnaround = completion - p["arrival"]
        waiting = turnaround - p["burst"]
        response = start - p["arrival"]
        results.append({"pid": p["pid"], "completion": completion,
                         "turnaround": turnaround, "waiting": waiting, "response": response})
        trace.append({"pid": p["pid"], "start": start, "end": completion})
        order.append(p["pid"])
        time = completion
    return order, results, trace


def sjf(processes):
    """Shortest Job First (non-preemptive): always run whichever arrived process has the shortest burst."""
    procs = [dict(p) for p in processes]
    time = 0
    done = []
    order, results, trace = [], [], []
    while len(done) < len(procs):
        available = [p for p in procs if p["arrival"] <= time and p["pid"] not in done]
        if not available:
            time = min(p["arrival"] for p in procs if p["pid"] not in done)
            continue
        p = min(available, key=lambda x: x["burst"])
        start = time
        completion = start + p["burst"]
        turnaround = completion - p["arrival"]
        waiting = turnaround - p["burst"]
        response = start - p["arrival"]
        results.append({"pid": p["pid"], "completion": completion,
                         "turnaround": turnaround, "waiting": waiting, "response": response})
        trace.append({"pid": p["pid"], "start": start, "end": completion})
        order.append(p["pid"])
        done.append(p["pid"])
        time = completion
    return order, results, trace


def ai_sjf(processes, predicted_burst):
    """
    SJF where the *ordering decision* uses predicted_burst (a dict of pid -> predicted burst,
    normally from Ansuman's Random Forest model), but actual execution time still uses the
    real burst, since the CPU can't run faster than reality just because the prediction was wrong.
    """
    procs = [dict(p, predicted=predicted_burst[p["pid"]]) for p in processes]
    time = 0
    done = []
    order, results, trace = [], [], []
    while len(done) < len(procs):
        available = [p for p in procs if p["arrival"] <= time and p["pid"] not in done]
        if not available:
            time = min(p["arrival"] for p in procs if p["pid"] not in done)
            continue
        p = min(available, key=lambda x: x["predicted"])
        start = time
        completion = start + p["burst"]
        turnaround = completion - p["arrival"]
        waiting = turnaround - p["burst"]
        response = start - p["arrival"]
        results.append({"pid": p["pid"], "completion": completion,
                         "turnaround": turnaround, "waiting": waiting, "response": response})
        trace.append({"pid": p["pid"], "start": start, "end": completion})
        order.append(p["pid"])
        done.append(p["pid"])
        time = completion
    return order, results, trace


def srtf(processes):
    """Shortest Remaining Time First (preemptive): always run whichever available process has the least work left."""
    procs = [dict(p, remaining=p["burst"]) for p in processes]
    n = len(procs)
    completed = 0
    time = 0
    completion, start_time = {}, {}
    order = []
    trace = []
    last_pid = None
    segment_start = None

    while completed < n:
        available = [p for p in procs if p["arrival"] <= time and p["remaining"] > 0]
        if not available:
            time += 1
            continue
        p = min(available, key=lambda x: x["remaining"])
        if p["pid"] not in start_time:
            start_time[p["pid"]] = time
        if last_pid != p["pid"]:
            if last_pid is not None:
                trace.append({"pid": last_pid, "start": segment_start, "end": time})
            order.append(p["pid"])
            last_pid = p["pid"]
            segment_start = time
        p["remaining"] -= 1
        time += 1
        if p["remaining"] == 0:
            completed += 1
            completion[p["pid"]] = time
    if last_pid is not None:
        trace.append({"pid": last_pid, "start": segment_start, "end": time})

    results = []
    for p in procs:
        comp = completion[p["pid"]]
        turnaround = comp - p["arrival"]
        waiting = turnaround - p["burst"]
        response = start_time[p["pid"]] - p["arrival"]
        results.append({"pid": p["pid"], "completion": comp,
                         "turnaround": turnaround, "waiting": waiting, "response": response})
    return order, results, trace


def round_robin(processes, quantum=50):
    """Round Robin with correct handling of CPU idle periods."""

    procs = sorted(
        [dict(p, remaining=p["burst"]) for p in processes],
        key=lambda p: p["arrival"]
    )

    queue = deque()
    time = 0
    order = []
    trace = []
    completion = {}
    first_start = {}

    i = 0

    if procs:
        time = procs[0]["arrival"]

    while i < len(procs) or queue:

        # If no process is ready, jump to the next arrival
        if not queue:
            if i < len(procs):
                time = max(time, procs[i]["arrival"])

                while (
                    i < len(procs)
                    and procs[i]["arrival"] <= time
                ):
                    queue.append(procs[i])
                    i += 1
            else:
                break

        p = queue.popleft()

        # Record first response/start time
        if p["pid"] not in first_start:
            first_start[p["pid"]] = time

        run = min(
            quantum,
            p["remaining"]
        )

        seg_start = time

        time += run

        p["remaining"] -= run

        order.append(p["pid"])

        trace.append({
            "pid": p["pid"],
            "start": seg_start,
            "end": time
        })

        # Add newly arrived processes
        while (
            i < len(procs)
            and procs[i]["arrival"] <= time
        ):
            queue.append(procs[i])
            i += 1

        # Put unfinished process back into queue
        if p["remaining"] > 0:
            queue.append(p)
        else:
            completion[p["pid"]] = time

    return order, completion, trace, first_start


def round_robin_metrics(processes, quantum=50):
    """Round Robin metrics with correct response time."""

    order, completion, trace, first_start = round_robin(
        processes,
        quantum
    )

    results = []

    for p in processes:

        comp = completion[p["pid"]]

        turnaround = (
            comp - p["arrival"]
        )

        waiting = (
            turnaround - p["burst"]
        )

        response = (
            first_start[p["pid"]]
            - p["arrival"]
        )

        results.append({
            "pid": p["pid"],
            "completion": comp,
            "turnaround": turnaround,
            "waiting": waiting,
            "response": response
        })

    return order, results, trace


def summarize(name, order, results):
    """Turns a scheduler's raw results into one clean comparable row."""
    avg_wait = sum(r["waiting"] for r in results) / len(results)
    avg_tat = sum(r["turnaround"] for r in results) / len(results)
    avg_resp = sum(r["response"] for r in results) / len(results)
    return {
        "scheduler": name,
        "order": order,
        "avg_waiting": round(avg_wait, 2),
        "avg_turnaround": round(avg_tat, 2),
        "avg_response": round(avg_resp, 2)
    }


if __name__ == "__main__":
    processes = load_processes(n=8)
    print("Loaded processes:", processes)
    print()
    for name, func in [("FCFS", fcfs), ("SJF", sjf), ("SRTF", srtf)]:
        order, results, trace = func(processes)
        print(summarize(name, order, results))

    rr_order, rr_results, rr_trace = round_robin_metrics(processes, quantum=50)
    print(summarize("Round Robin", rr_order, rr_results))