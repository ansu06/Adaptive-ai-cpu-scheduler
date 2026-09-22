import pandas as pd

from drift.workload_detector import detect_drift


# Load original process data
df = pd.read_csv("process_data.csv")


# Create a deliberately different workload
drifted_workload = df.head(10).copy()

# Make burst times much larger
drifted_workload["Burst time"] = [
    800, 850, 900, 950, 1000,
    1100, 1200, 1300, 1400, 1500
]

# Make arrival times much more spread out
drifted_workload["Arrival Time"] = [
    0, 50, 100, 150, 200,
    300, 400, 500, 600, 700
]


result = detect_drift(drifted_workload)


print("\n" + "=" * 60)
print("DRIFT TEST")
print("=" * 60)

print(f"\nStatus: {result['status']}")
print(f"Drift Score: {result['drift_score']:.4f}")
print(f"Trust AI: {result['trust_ai']}")

print("\nModified Workload Features:")

for key, value in result["features"].items():
    print(f"{key}: {value:.4f}")