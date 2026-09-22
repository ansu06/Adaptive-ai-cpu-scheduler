import pandas as pd


df = pd.read_csv(
    "model/training_data_diverse.csv"
)


# ============================================================
# Basic information
# ============================================================

print()
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Shape:", df.shape)

print()
print("Columns:")
print(df.columns.tolist())


# ============================================================
# Label distribution
# ============================================================

print()
print("=" * 60)
print("SCHEDULER LABEL DISTRIBUTION")
print("=" * 60)

print(
    df["best_scheduler"]
    .value_counts()
)


print()
print("Label percentages:")

print(
    (
        df["best_scheduler"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


# ============================================================
# Labels by scenario
# ============================================================

print()
print("=" * 60)
print("BEST SCHEDULER BY SCENARIO")
print("=" * 60)

scenario_table = pd.crosstab(
    df["scenario"],
    df["best_scheduler"]
)

print(scenario_table)


# ============================================================
# Average scheduler scores
# ============================================================

print()
print("=" * 60)
print("AVERAGE PERFORMANCE SCORES")
print("=" * 60)

for scheduler in [
    "fcfs",
    "sjf",
    "srtf",
    "rr"
]:

    column = (
        f"{scheduler}_score"
    )

    print(
        f"{scheduler.upper():6} : "
        f"{df[column].mean():.4f}"
    )


# ============================================================
# Average waiting time
# ============================================================

print()
print("=" * 60)
print("AVERAGE WAITING TIMES")
print("=" * 60)

for scheduler in [
    "fcfs",
    "sjf",
    "srtf",
    "rr"
]:

    column = (
        f"{scheduler}_waiting"
    )

    print(
        f"{scheduler.upper():6} : "
        f"{df[column].mean():.2f}"
    )


# ============================================================
# Features by scheduler label
# ============================================================

print()
print("=" * 60)
print("WORKLOAD FEATURES BY BEST SCHEDULER")
print("=" * 60)

feature_columns = [
    "num_processes",
    "avg_burst",
    "std_burst",
    "min_burst",
    "max_burst",
    "burst_range",
    "avg_arrival",
    "std_arrival"
]

print(
    df.groupby(
        "best_scheduler"
    )[feature_columns]
    .mean()
    .round(2)
)


# ============================================================
# Scenario + scheduler counts
# ============================================================

print()
print("=" * 60)
print("SCENARIO / SCHEDULER DETAILS")
print("=" * 60)

for scenario in df["scenario"].unique():

    subset = df[
        df["scenario"] == scenario
    ]

    print()
    print(
        f"{scenario}:"
    )

    print(
        subset[
            "best_scheduler"
        ].value_counts()
    )