import pandas as pd

from model.decision_engine import make_decision
from model.evaluate_schedulers import (
    convert_to_processes
)

from scheduler.schedulers import (
    fcfs,
    sjf,
    srtf,
    round_robin_metrics,
    summarize
)


def evaluate_all_schedulers(df):
    """
    Run all traditional schedulers and compare
    their performance.
    """

    processes = convert_to_processes(df)

    results = []

    # FCFS
    order, metrics, trace = fcfs(processes)
    results.append(
        summarize("FCFS", order, metrics)
    )

    # SJF
    order, metrics, trace = sjf(processes)
    results.append(
        summarize("SJF", order, metrics)
    )

    # SRTF
    order, metrics, trace = srtf(processes)
    results.append(
        summarize("SRTF", order, metrics)
    )

    # Round Robin
    order, metrics, trace = round_robin_metrics(
        processes,
        quantum=4
    )

    results.append(
        summarize("Round Robin", order, metrics)
    )

    return results


def select_best_scheduler(results):
    """
    Select scheduler with the lowest average waiting time.
    """

    return min(
        results,
        key=lambda x: x["avg_waiting"]
    )


def adaptive_schedule(df):

    # First make AI decision
    decision = make_decision(df)

    print("\n" + "=" * 60)
    print("ADAPTIVE SCHEDULER")
    print("=" * 60)

    print(f"\nAI Recommendation : {decision['ai_prediction']}")
    print(
        f"AI Confidence     : "
        f"{decision['ai_confidence'] * 100:.2f}%"
    )

    print(f"Drift Status      : {decision['drift_status']}")
    print(f"Drift Score       : {decision['drift_score']:.4f}")

    print("\nDecision           :", decision["decision"])
    print("Reason             :", decision["reason"])

    # ------------------------------------------------
    # CASE 1: Trust AI
    # ------------------------------------------------

    if decision["decision"] == "TRUST AI":

        selected = decision["ai_prediction"]

        print("\nSelected Scheduler:", selected)
        print("Selection Method  : AI Recommendation")

        return {
            "decision": decision,
            "selected_scheduler": selected,
            "method": "AI",
            "results": None
        }

    # ------------------------------------------------
    # CASE 2: Re-evaluate or fallback
    # ------------------------------------------------

    print("\nRunning traditional scheduler evaluation...")

    results = evaluate_all_schedulers(df)

    best = select_best_scheduler(results)

    print("\n" + "-" * 60)
    print("SCHEDULER COMPARISON")
    print("-" * 60)

    for result in results:

        print(
            f"{result['scheduler']:15}"
            f" Waiting: {result['avg_waiting']:.2f}"
            f" | Turnaround: {result['avg_turnaround']:.2f}"
            f" | Response: {result['avg_response']:.2f}"
        )

    print("\n" + "=" * 60)

    print(
        f"SELECTED: {best['scheduler']}"
    )

    print(
        f"Average Waiting Time: "
        f"{best['avg_waiting']:.2f}"
    )

    if decision["decision"] == "FALLBACK":
        method = "DRIFT FALLBACK"
    else:
        method = "PERFORMANCE RE-EVALUATION"

    print(f"Selection Method: {method}")

    return {
        "decision": decision,
        "selected_scheduler": best["scheduler"],
        "method": method,
        "results": results
    }


if __name__ == "__main__":

    df = pd.read_csv("process_data.csv")

    workload = df.head(10)

    result = adaptive_schedule(workload)