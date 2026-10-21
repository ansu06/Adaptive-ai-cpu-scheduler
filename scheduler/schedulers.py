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
            "arrival": round(float(row["Arrival Time"]) * 100, 2),  # scale 0-1 -> 0-100
            "burst": int(row["Burst time"])
        })
    return processes


def fcfs(processes):
    """First Come First Served: run processes strictly in arrival order."""
    procs = [dict(p, _idx=i) for i, p in enumerate(processes)]
    procs.sort(key=lambda p: (p["arrival"], p["_idx"]))
    time = 0.0
    order, trace = [], []
    results_by_idx = {}

    for p in procs:
        start = max(time, p["arrival"])
        completion = start + p["burst"]
        turnaround = completion - p["arrival"]
        waiting = turnaround - p["burst"]
        response = start - p["arrival"]

        results_by_idx[p["_idx"]] = {
            "pid": p["pid"],
            "completion": round(completion, 2),
            "turnaround": round(turnaround, 2),
            "waiting": round(waiting, 2),
            "response": round(response, 2)
        }
        if p["burst"] > 0:
            trace.append({"pid": p["pid"], "start": round(start, 2), "end": round(completion, 2)})
        order.append(p["pid"])
        time = completion

    results = [results_by_idx[i] for i in range(len(processes))]
    return order, results, trace


def sjf(processes):
    """Shortest Job First (non-preemptive): always run whichever arrived process has shortest burst."""
    procs = [dict(p, _idx=i) for i, p in enumerate(processes)]
    time = 0.0
    done_indices = set()
    order, trace = [], []
    results_by_idx = {}

    while len(done_indices) < len(procs):
        available = [p for p in procs if p["arrival"] <= time and p["_idx"] not in done_indices]
        if not available:
            time = min(p["arrival"] for p in procs if p["_idx"] not in done_indices)
            continue

        p = min(available, key=lambda x: (x["burst"], x["arrival"], x["_idx"]))
        start = time
        completion = start + p["burst"]
        turnaround = completion - p["arrival"]
        waiting = turnaround - p["burst"]
        response = start - p["arrival"]

        results_by_idx[p["_idx"]] = {
            "pid": p["pid"],
            "completion": round(completion, 2),
            "turnaround": round(turnaround, 2),
            "waiting": round(waiting, 2),
            "response": round(response, 2)
        }
        if p["burst"] > 0:
            trace.append({"pid": p["pid"], "start": round(start, 2), "end": round(completion, 2)})
        order.append(p["pid"])
        done_indices.add(p["_idx"])
        time = completion

    results = [results_by_idx[i] for i in range(len(processes))]
    return order, results, trace


def ai_sjf(processes, predicted_burst):
    """
    SJF where the ordering decision uses predicted_burst, but actual execution time
    uses the real burst.
    """
    procs = [dict(p, predicted=predicted_burst.get(p["pid"], p["burst"]), _idx=i) for i, p in enumerate(processes)]
    time = 0.0
    done_indices = set()
    order, trace = [], []
    results_by_idx = {}

    while len(done_indices) < len(procs):
        available = [p for p in procs if p["arrival"] <= time and p["_idx"] not in done_indices]
        if not available:
            time = min(p["arrival"] for p in procs if p["_idx"] not in done_indices)
            continue

        p = min(available, key=lambda x: (x["predicted"], x["arrival"], x["_idx"]))
        start = time
        completion = start + p["burst"]
        turnaround = completion - p["arrival"]
        waiting = turnaround - p["burst"]
        response = start - p["arrival"]

        results_by_idx[p["_idx"]] = {
            "pid": p["pid"],
            "completion": round(completion, 2),
            "turnaround": round(turnaround, 2),
            "waiting": round(waiting, 2),
            "response": round(response, 2)
        }
        if p["burst"] > 0:
            trace.append({"pid": p["pid"], "start": round(start, 2), "end": round(completion, 2)})
        order.append(p["pid"])
        done_indices.add(p["_idx"])
        time = completion

    results = [results_by_idx[i] for i in range(len(processes))]
    return order, results, trace


def srtf(processes):
    """Shortest Remaining Time First (preemptive): always run available process with least work left."""
    procs = [dict(p, remaining=float(p["burst"]), _idx=i) for i, p in enumerate(processes)]
    n = len(procs)
    completed = 0
    time = 0.0
    completion, start_time = {}, {}
    order = []
    trace = []
    last_pid = None
    segment_start = None

    # Complete zero-burst jobs that arrive at start
    for p in procs:
        if p["remaining"] == 0:
            completed += 1
            start_time[p["_idx"]] = p["arrival"]
            completion[p["_idx"]] = p["arrival"]

    while completed < n:
        available = [p for p in procs if p["arrival"] <= time and p["remaining"] > 0]
        if not available:
            # Fast-forward directly through CPU idle intervals
            uncompleted_arrivals = [p["arrival"] for p in procs if p["remaining"] > 0]
            if uncompleted_arrivals:
                time = min(uncompleted_arrivals)
            else:
                break
            continue

        # Choose job with shortest remaining time, tie-breaking by arrival time and original index
        p = min(available, key=lambda x: (x["remaining"], x["arrival"], x["_idx"]))

        if p["_idx"] not in start_time:
            start_time[p["_idx"]] = time

        if last_pid != p["pid"]:
            if last_pid is not None and segment_start is not None and time > segment_start:
                trace.append({"pid": last_pid, "start": round(segment_start, 2), "end": round(time, 2)})
            order.append(p["pid"])
            last_pid = p["pid"]
            segment_start = time

        # Event-driven duration: run until current process finishes or a new job arrives
        future_arrivals = [x["arrival"] for x in procs if x["arrival"] > time]
        if future_arrivals:
            next_arrival = min(future_arrivals)
            step = min(p["remaining"], next_arrival - time)
        else:
            step = p["remaining"]

        if step <= 0:
            step = p["remaining"] if p["remaining"] > 0 else 1.0

        p["remaining"] = round(p["remaining"] - step, 6)
        time = round(time + step, 6)

        if p["remaining"] <= 1e-9:
            completed += 1
            completion[p["_idx"]] = time
            p["remaining"] = 0

    if last_pid is not None and segment_start is not None and time > segment_start:
        trace.append({"pid": last_pid, "start": round(segment_start, 2), "end": round(time, 2)})

    results = []
    for i, p in enumerate(processes):
        comp = completion.get(i, p["arrival"] + p["burst"])
        fst = start_time.get(i, p["arrival"])
        turnaround = comp - p["arrival"]
        waiting = turnaround - p["burst"]
        response = fst - p["arrival"]
        results.append({
            "pid": p["pid"],
            "completion": round(comp, 2),
            "turnaround": round(turnaround, 2),
            "waiting": round(waiting, 2),
            "response": round(response, 2)
        })
    return order, results, trace


def round_robin(processes, quantum=4):
    """Round Robin with correct handling of CPU idle periods, duplicate PIDs, and zero-burst jobs."""
    procs = [dict(p, remaining=float(p["burst"]), _idx=i) for i, p in enumerate(processes)]
    procs.sort(key=lambda p: (p["arrival"], p["_idx"]))

    queue = deque()
    time = 0.0
    order = []
    trace = []
    completion = {}
    first_start = {}

    i = 0
    if procs:
        time = procs[0]["arrival"]

    while i < len(procs) or queue:
        # If no process is ready in queue, jump to next arrival
        if not queue:
            if i < len(procs):
                time = max(time, procs[i]["arrival"])
                while i < len(procs) and procs[i]["arrival"] <= time:
                    queue.append(procs[i])
                    i += 1
            else:
                break

        p = queue.popleft()

        # Handle zero-burst jobs without consuming quantum
        if p["remaining"] <= 0:
            if p["_idx"] not in first_start:
                first_start[p["_idx"]] = time
            completion[p["_idx"]] = time
            order.append(p["pid"])
            continue

        if p["_idx"] not in first_start:
            first_start[p["_idx"]] = time

        run = min(float(quantum), p["remaining"])
        seg_start = time
        time += run
        p["remaining"] -= run

        order.append(p["pid"])
        trace.append({
            "pid": p["pid"],
            "start": round(seg_start, 2),
            "end": round(time, 2)
        })

        # Add newly arrived processes that arrived during this quantum
        while i < len(procs) and procs[i]["arrival"] <= time:
            queue.append(procs[i])
            i += 1

        # Put unfinished process back into queue
        if p["remaining"] > 0:
            queue.append(p)
        else:
            completion[p["_idx"]] = time

    return order, completion, trace, first_start


def round_robin_metrics(processes, quantum=4):
    """Round Robin metrics with correct response time (first start - arrival)."""
    order, completion, trace, first_start = round_robin(processes, quantum=quantum)
    results = []

    for i, p in enumerate(processes):
        comp = completion.get(i, p["arrival"] + p["burst"])
        fst = first_start.get(i, p["arrival"])
        turnaround = comp - p["arrival"]
        waiting = turnaround - p["burst"]
        response = fst - p["arrival"]

        results.append({
            "pid": p["pid"],
            "completion": round(comp, 2),
            "turnaround": round(turnaround, 2),
            "waiting": round(waiting, 2),
            "response": round(response, 2)
        })

    return order, results, trace


def summarize(name, order, results):
    """Turns a scheduler's raw results into one clean comparable row."""
    avg_wait = sum(r["waiting"] for r in results) / len(results) if results else 0.0
    avg_tat = sum(r["turnaround"] for r in results) / len(results) if results else 0.0
    avg_resp = sum(r["response"] for r in results) / len(results) if results else 0.0
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

    rr_order, rr_results, rr_trace = round_robin_metrics(processes, quantum=4)
    print(summarize("Round Robin", rr_order, rr_results))