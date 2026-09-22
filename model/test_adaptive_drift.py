import pandas as pd

from model.adaptive_scheduler import adaptive_schedule


df = pd.read_csv("process_data.csv")

# Create deliberately different workload
workload = df.head(10).copy()

workload["Burst time"] = [
    800, 850, 900, 950, 1000,
    1100, 1200, 1300, 1400, 1500
]

workload["Arrival Time"] = [
    0, 50, 100, 150, 200,
    300, 400, 500, 600, 700
]

result = adaptive_schedule(workload)

print("\n" + "=" * 60)
print("DRIFT FALLBACK TEST")
print("=" * 60)

print(f"Final Scheduler : {result['selected_scheduler']}")
print(f"Method          : {result['method']}")