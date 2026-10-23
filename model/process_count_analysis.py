from pathlib import Path
import pandas as pd
import numpy as np

from scheduler.schedulers import fcfs, sjf, srtf, round_robin_metrics, summarize
from model.predict import predict_scheduler, extract_features


def evaluate_ground_truth_scheduler(df):
    """
    Evaluate ground-truth optimal scheduler for a workload instance using the
    exact multi-objective scoring formula established in generate_diverse_workloads.py:
    score = 0.4 * norm(waiting) + 0.3 * norm(turnaround) + 0.3 * norm(response).
    """
    processes = [
        {
            "pid": f"P{int(row['Job Id'])}",
            "arrival": float(row["Arrival Time"]),
            "burst": int(row["Burst time"])
        }
        for _, row in df.iterrows()
    ]

    o_f, m_f, _ = fcfs(processes)
    o_sj, m_sj, _ = sjf(processes)
    o_sr, m_sr, _ = srtf(processes)
    o_rr, m_rr, _ = round_robin_metrics(processes, quantum=4)

    results = [
        summarize("FCFS", o_f, m_f),
        summarize("SJF", o_sj, m_sj),
        summarize("SRTF", o_sr, m_sr),
        summarize("Round Robin", o_rr, m_rr)
    ]

    waiting = np.array([r["avg_waiting"] for r in results], dtype=float)
    turnaround = np.array([r["avg_turnaround"] for r in results], dtype=float)
    response = np.array([r["avg_response"] for r in results], dtype=float)

    def normalize(vals):
        min_v = vals.min()
        max_v = vals.max()
        if max_v == min_v:
            return np.zeros(len(vals), dtype=float)
        return (vals - min_v) / (max_v - min_v)

    scores = (
        0.4 * normalize(waiting)
        + 0.3 * normalize(turnaround)
        + 0.3 * normalize(response)
    )

    best_idx = int(np.argmin(scores))
    # If Round Robin had the lowest score, map to its best non-round-robin counterpart if
    # the classifier only models {FCFS, SJF, SRTF}, else return the true best scheduler.
    best_name = results[best_idx]["scheduler"]
    if best_name == "Round Robin":
        # Among standard policies
        sub_scores = scores[:3]
        best_name = results[int(np.argmin(sub_scores))]["scheduler"]

    return best_name


def run_process_count_analysis(process_counts=(2, 3, 4, 5, 6, 8, 10), samples_per_count=6, base_seed=42):
    """
    Run deterministic process-count sensitivity analysis across varying workload sizes.
    Evaluates prediction distributions, average confidence, and ground-truth simulation accuracy.
    """
    base_dir = Path(__file__).resolve().parent.parent
    data_file = base_dir / "process_data.csv"
    if not data_file.exists():
        data_file = Path("process_data.csv")

    source_df = pd.read_csv(data_file)
    summary_rows = []
    all_samples = []

    for n in process_counts:
        n = int(n)
        preds = []
        confs = []
        truths = []
        max_confs = []

        for s in range(samples_per_count):
            rng = np.random.default_rng(base_seed * 1000 + n * 50 + s)
            seed_val = int(rng.integers(0, 1_000_000))
            sample_df = source_df.sample(n=n, replace=False, random_state=seed_val).copy()

            pred, conf_dict, feats = predict_scheduler(sample_df)
            gt = evaluate_ground_truth_scheduler(sample_df)
            max_c = float(max(conf_dict.values()))

            preds.append(pred)
            confs.append(conf_dict)
            truths.append(gt)
            max_confs.append(max_c)

            all_samples.append({
                "process_count": n,
                "sample_idx": s + 1,
                "prediction": pred,
                "ground_truth": gt,
                "confidence": max_c,
                "correct": (pred == gt),
                "features": feats
            })

        pred_counts = pd.Series(preds).value_counts().to_dict()
        correct_count = sum(1 for p, t in zip(preds, truths) if p == t)
        accuracy = (correct_count / len(preds)) * 100.0

        dominant_pred = pd.Series(preds).mode()[0] if len(preds) > 0 else "SRTF"
        mean_conf = float(np.mean(max_confs)) * 100.0

        summary_rows.append({
            "Process Count (N)": n,
            "Evaluated Samples": len(preds),
            "FCFS (%)": round((pred_counts.get("FCFS", 0) / len(preds)) * 100.0, 1),
            "SJF (%)": round((pred_counts.get("SJF", 0) / len(preds)) * 100.0, 1),
            "SRTF (%)": round((pred_counts.get("SRTF", 0) / len(preds)) * 100.0, 1),
            "Dominant Policy": dominant_pred,
            "Mean Confidence (%)": round(mean_conf, 1),
            "Ground-Truth Accuracy (%)": round(accuracy, 1)
        })

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, all_samples


if __name__ == "__main__":
    df, samples = run_process_count_analysis()
    print("\n" + "=" * 70)
    print("PROCESS-COUNT SENSITIVITY ANALYSIS (DETERMINISTIC EVALUATION)")
    print("=" * 70)
    print(df.to_string(index=False))
