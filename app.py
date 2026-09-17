import random

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from monitoring.system_monitor import get_system_metrics, get_top_processes
from scheduler.schedulers import (ai_sjf, fcfs, load_processes,
                                   round_robin_metrics, sjf, srtf, summarize)

st.set_page_config(page_title="AI-Based Intelligent CPU Scheduler", layout="wide")

st.title("AI-Based Intelligent CPU Scheduler")
st.caption("Workload-Aware Scheduling with ML — Sayan Giri & Ansuman Singh")

# ---------------- Live System Status ----------------
st.header("Live System Status")
metrics = get_system_metrics()
col1, col2, col3 = st.columns(3)
col1.metric("CPU Usage", f"{metrics['cpu_percent']}%")
col2.metric("Memory Usage", f"{metrics['memory_percent']}%")
col3.metric("Running Processes", metrics["process_count"])

with st.expander("Top 5 real processes by CPU usage"):
    st.dataframe(pd.DataFrame(get_top_processes()), width='stretch')

# ---------------- Process Workload ----------------
st.header("Process Workload")
n = st.slider("Number of processes to schedule", 4, 20, 8)
processes = load_processes(n=n)
st.dataframe(pd.DataFrame(processes), width='stretch')

# Placeholder predicted burst until Ansuman's model is wired in.
# Swap this dict with model.predict(...) output later — nothing else changes.
random.seed(42)
predicted_burst = {p["pid"]: max(1, p["burst"] + random.randint(-20, 20)) for p in processes}

# ---------------- Run all schedulers ----------------
st.header("Scheduler Comparison")

traces = {}
summary_rows = []

for name, func in [("FCFS", fcfs), ("SJF", sjf), ("SRTF", srtf)]:
    order, results, trace = func(processes)
    summary_rows.append(summarize(name, order, results))
    traces[name] = trace

rr_order, rr_results, rr_trace = round_robin_metrics(processes, quantum=50)
summary_rows.append(summarize("Round Robin", rr_order, rr_results))
traces["Round Robin"] = rr_trace

ai_order, ai_results, ai_trace = ai_sjf(processes, predicted_burst)
summary_rows.append(summarize("AI-SJF (placeholder model)", ai_order, ai_results))
traces["AI-SJF"] = ai_trace

summary_df = pd.DataFrame(summary_rows).drop(columns=["order"])
st.dataframe(summary_df, width='stretch')

st.bar_chart(summary_df.set_index("scheduler")[["avg_waiting", "avg_turnaround", "avg_response"]])

# ---------------- Execution timeline (Gantt-style) ----------------
st.header("Execution Timeline")
chosen = st.selectbox("Choose a scheduler to visualize", list(traces.keys()))
segments = traces[chosen]

fig, ax = plt.subplots(figsize=(10, 0.6 * len(set(s["pid"] for s in segments)) + 1))
pids = sorted(set(s["pid"] for s in segments))
colors = plt.cm.tab20.colors
pid_color = {pid: colors[i % len(colors)] for i, pid in enumerate(pids)}
for i, pid in enumerate(pids):
    bars = [(s["start"], s["end"] - s["start"]) for s in segments if s["pid"] == pid]
    ax.broken_barh(bars, (i * 10, 9), facecolors=pid_color[pid])
ax.set_yticks([i * 10 + 4.5 for i in range(len(pids))])
ax.set_yticklabels(pids)
ax.set_xlabel("Time")
ax.set_title(f"{chosen} Execution Timeline")
st.pyplot(fig)

st.caption(
    "AI-SJF currently uses a placeholder predicted burst (actual burst ± random noise) "
    "until Ansuman's Random Forest model is integrated. Swap the `predicted_burst` dict "
    "for `model.predict(...)` output — no other code changes needed."
)