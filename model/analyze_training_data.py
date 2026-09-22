import pandas as pd

df = pd.read_csv("model/training_data.csv")

print("\nDATASET SHAPE")
print("=" * 50)
print(df.shape)

print("\nLABEL DISTRIBUTION")
print("=" * 50)
print(df["best_scheduler"].value_counts())

print("\nAVERAGE SCHEDULER WAITING TIMES")
print("=" * 50)

print("FCFS :", df["fcfs_waiting"].mean())
print("SJF  :", df["sjf_waiting"].mean())
print("SRTF :", df["srtf_waiting"].mean())
print("RR   :", df["rr_waiting"].mean())

print("\nSJF vs SRTF")
print("=" * 50)

difference = (
    df["sjf_waiting"] -
    df["srtf_waiting"]
)

print("Mean difference:", difference.mean())
print("Minimum difference:", difference.min())
print("Maximum difference:", difference.max())

print("\nFIRST 10 WORKLOADS")
print("=" * 50)

print(
    df[
        [
            "num_processes",
            "avg_burst",
            "std_burst",
            "avg_arrival",
            "std_arrival",
            "avg_resources",
            "preemptive_ratio",
            "fcfs_waiting",
            "sjf_waiting",
            "srtf_waiting",
            "rr_waiting",
            "best_scheduler"
        ]
    ].head(10).to_string(index=False)
)