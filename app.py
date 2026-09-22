import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from monitoring.system_monitor import (
    get_system_metrics,
    get_top_processes
)

from scheduler.schedulers import (
    fcfs,
    load_processes,
    round_robin_metrics,
    sjf,
    srtf,
    summarize
)

from model.adaptive_scheduler import adaptive_schedule


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Adaptive AI CPU Scheduler",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("⚙️ Adaptive AI CPU Scheduler")

st.caption(
    "Workload-Aware Scheduling with Machine Learning — "
    "Sayan Giri & Ansuman Singh"
)

st.divider()


# ============================================================
# LIVE SYSTEM STATUS
# ============================================================

st.header("🖥️ Live System Status")

metrics = get_system_metrics()

col1, col2, col3 = st.columns(3)

col1.metric(
    "CPU Usage",
    f"{metrics['cpu_percent']}%"
)

col2.metric(
    "Memory Usage",
    f"{metrics['memory_percent']}%"
)

col3.metric(
    "Running Processes",
    metrics["process_count"]
)

with st.expander("🔍 Top 5 Real Processes by CPU Usage"):

    top_processes = get_top_processes()

    if top_processes:
        st.dataframe(
            pd.DataFrame(top_processes),
            width="stretch",
            hide_index=True
        )
    else:
        st.info("No process information available.")


# ============================================================
# PROCESS WORKLOAD
# ============================================================

st.header("📋 Process Workload")

st.caption(
    "Normal workload uses generated processes. "
    "Drift Test intentionally creates an abnormal workload "
    "to demonstrate adaptive fallback."
)

n = st.slider(
    "Number of processes to schedule",
    min_value=4,
    max_value=20,
    value=8
)

# ============================================================
# WORKLOAD MODE
# ============================================================

workload_mode = st.radio(
    "Workload Mode",
    ["Normal Workload", "Drift Test"],
    horizontal=True
)

processes = load_processes(n=n)

# ------------------------------------------------------------
# Force an intentionally abnormal workload for demonstration
# ------------------------------------------------------------

if workload_mode == "Drift Test":

    drift_processes = []

    for i, p in enumerate(processes):

        drift_processes.append({
            "pid": p["pid"],
            "arrival": 200 + (i * 25),
            "burst": 800 + (i * 100)
        })

    processes = drift_processes


process_df = pd.DataFrame(processes)

st.dataframe(
    process_df,
    width="stretch",
    hide_index=True
)

st.dataframe(
    process_df,
    width="stretch",
    hide_index=True
)


# ============================================================
# CONVERT WORKLOAD FOR AI MODEL
# ============================================================

ai_workload = pd.DataFrame({
    "Job Id": [
        int(str(p["pid"]).replace("P", ""))
        for p in processes
    ],

    "Burst time": [
        p["burst"]
        for p in processes
    ],

    "Arrival Time": [
        p["arrival"]
        for p in processes
    ]
})


# ============================================================
# ADAPTIVE AI DECISION
# ============================================================

st.header("🤖 AI Workload Analysis")

try:

    adaptive_result = adaptive_schedule(ai_workload)

    decision_data = adaptive_result["decision"]

    ai_prediction = decision_data["ai_prediction"]
    ai_confidence = float(decision_data["ai_confidence"])

    drift_status = decision_data["drift_status"]
    drift_score = float(decision_data["drift_score"])

    decision = decision_data["decision"]
    reason = decision_data["reason"]

    selected_scheduler = adaptive_result["selected_scheduler"]
    selection_method = adaptive_result["method"]

    confidence = decision_data["confidence"]

except Exception as e:

    st.error(f"AI system error: {e}")
    st.stop()

# ============================================================
# VALIDATE SELECTED SCHEDULER
# ============================================================

valid_schedulers = {
    "FCFS",
    "SJF",
    "SRTF",
    "Round Robin"
}

if selected_scheduler not in valid_schedulers:

    st.warning(
        f"Unknown scheduler returned by AI: {selected_scheduler}. "
        f"Using SRTF as fallback."
    )

    selected_scheduler = "SRTF"

# ============================================================
# ADAPTIVE DECISION FLOW
# ============================================================

st.subheader("🔄 Adaptive Decision Flow")

flow_col1, flow_col2, flow_col3, flow_col4 = st.columns(4)

with flow_col1:
    st.info(
        f"📊 Workload\n\n"
        f"{len(processes)} processes"
    )

with flow_col2:
    st.info(
        f"🤖 AI Prediction\n\n"
        f"{ai_prediction}"
    )

with flow_col3:
    st.info(
        f"🧠 Confidence\n\n"
        f"{ai_confidence * 100:.2f}%"
    )

with flow_col4:
    st.success(
        f"🎯 Final Scheduler\n\n"
        f"{selected_scheduler}"
    )

# ============================================================
# AI SUMMARY CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "AI Recommendation",
    ai_prediction
)

col2.metric(
    "AI Confidence",
    f"{ai_confidence * 100:.2f}%"
)

col3.metric(
    "Drift Score",
    f"{drift_score:.4f}"
)

col4.metric(
    "Selected Scheduler",
    selected_scheduler
)


# ============================================================
# ADAPTIVE DECISION STATUS
# ============================================================

st.subheader("🧠 Adaptive Decision")


decision_upper = str(decision).upper()

if "TRUST" in decision_upper:

    st.success(
        f"🟢 **TRUST AI — {selected_scheduler} selected**"
    )

elif "FALLBACK" in decision_upper:

    st.error(
        f"🔴 **FALLBACK — {selected_scheduler} selected**"
    )

elif "RE-EVALUATE" in decision_upper:

    st.warning(
        f"🟡 **RE-EVALUATE — {selected_scheduler} selected**"
    )

else:

    st.info(
        f"Decision: **{decision}**"
    )


# ============================================================
# DECISION DETAILS
# ============================================================

detail_col1, detail_col2, detail_col3 = st.columns(3)

with detail_col1:

    st.markdown("**Workload Status**")

    drift_upper = str(drift_status).upper()

if "DETECTED" in drift_upper:

    st.error(f"🔴 {drift_status}")

else:

    st.success(f"🟢 {drift_status}")


with detail_col2:

    st.markdown("**Selection Method**")

    st.info(selection_method)


with detail_col3:

    st.markdown("**Reason**")

    st.write(reason)


# ============================================================
# AI CONFIDENCE
# ============================================================

st.markdown("### AI Confidence")

st.progress(
    min(max(ai_confidence, 0.0), 1.0),
    text=f"{ai_confidence * 100:.2f}%"
)


# ============================================================
# RUN TRADITIONAL SCHEDULERS
# ============================================================

st.header("📊 Scheduler Comparison")

traces = {}
summary_rows = []


# ---------------- FCFS ----------------

order, results, trace = fcfs(processes)

summary_rows.append(
    summarize(
        "FCFS",
        order,
        results
    )
)

traces["FCFS"] = trace


# ---------------- SJF ----------------

order, results, trace = sjf(processes)

summary_rows.append(
    summarize(
        "SJF",
        order,
        results
    )
)

traces["SJF"] = trace


# ---------------- SRTF ----------------

order, results, trace = srtf(processes)

summary_rows.append(
    summarize(
        "SRTF",
        order,
        results
    )
)

traces["SRTF"] = trace


# ---------------- ROUND ROBIN ----------------

rr_order, rr_results, rr_trace = round_robin_metrics(
    processes,
    quantum=4
)

summary_rows.append(
    summarize(
        "Round Robin",
        rr_order,
        rr_results
    )
)

traces["Round Robin"] = rr_trace


# ============================================================
# PERFORMANCE TABLE
# ============================================================

summary_df = pd.DataFrame(summary_rows)

display_df = summary_df.drop(
    columns=["order"],
    errors="ignore"
)

st.dataframe(
    display_df,
    width="stretch",
    hide_index=True
)

st.caption(
    "Lower waiting, turnaround, and response times indicate better scheduling "
    "performance for the evaluated workload."
)

# ============================================================
# ADAPTIVE SELECTION
# ============================================================

st.subheader("🎯 Adaptive Scheduler Selection")

selection_col1, selection_col2 = st.columns(2)

with selection_col1:

    st.success(
        f"🎯 **FINAL SCHEDULER: {selected_scheduler}**"
    )

with selection_col2:

    st.info(
        f"⚙️ **Selection Method: {selection_method}**"
    )


# ============================================================
# PERFORMANCE CHART
# ============================================================

st.subheader("📈 Scheduler Performance")

scheduler_names = display_df["scheduler"].tolist()

waiting = display_df["avg_waiting"].tolist()
turnaround = display_df["avg_turnaround"].tolist()
response = display_df["avg_response"].tolist()

fig, ax = plt.subplots(
    figsize=(12, 5)
)

x = range(len(scheduler_names))
width = 0.25

ax.bar(
    [i - width for i in x],
    waiting,
    width,
    label="Average Waiting"
)

ax.bar(
    x,
    turnaround,
    width,
    label="Average Turnaround"
)

ax.bar(
    [i + width for i in x],
    response,
    width,
    label="Average Response"
)

ax.set_xticks(list(x))
ax.set_xticklabels(scheduler_names)

ax.set_ylabel("Time")
ax.set_xlabel("Scheduler")

ax.set_title(
    "CPU Scheduling Performance Comparison"
)

ax.set_ylim(
    0,
    max(waiting + turnaround + response) * 1.15
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)

st.pyplot(
    fig,
    width="stretch"
)

plt.close(fig)


# ============================================================
# KEY PERFORMANCE METRIC
# ============================================================

st.subheader("⏱️ Average Waiting Time")

waiting_df = display_df[
    ["scheduler", "avg_waiting"]
].copy()

waiting_df = waiting_df.sort_values(
    "avg_waiting"
)

fig, ax = plt.subplots(
    figsize=(10, 4)
)

ax.barh(
    waiting_df["scheduler"],
    waiting_df["avg_waiting"]
)

ax.set_xlabel("Average Waiting Time")
ax.set_ylabel("Scheduler")

ax.set_title(
    "Average Waiting Time by Scheduler"
)

ax.grid(
    axis="x",
    alpha=0.25
)

st.pyplot(
    fig,
    width="stretch"
)

plt.close(fig)


# ============================================================
# EXECUTION TIMELINE
# ============================================================

st.header("🕒 Execution Timeline")

default_index = (
    list(traces.keys()).index(selected_scheduler)
    if selected_scheduler in traces
    else 0
)

chosen = st.selectbox(
    "Choose a scheduler to visualize",
    list(traces.keys()),
    index=default_index
)

segments = traces[chosen]


if segments:

    pids = sorted(
        set(
            segment["pid"]
            for segment in segments
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            14,
            max(4, 0.65 * len(pids) + 1)
        )
    )

    colors = plt.cm.tab20.colors

    pid_color = {
        pid: colors[i % len(colors)]
        for i, pid in enumerate(pids)
    }

    for i, pid in enumerate(pids):

        bars = [
            (
                segment["start"],
                segment["end"] - segment["start"]
            )
            for segment in segments
            if segment["pid"] == pid
        ]

        ax.broken_barh(
            bars,
            (
                i * 10,
                8
            ),
            facecolors=pid_color[pid]
        )

    ax.set_yticks(
        [
            i * 10 + 4
            for i in range(len(pids))
        ]
    )

    ax.set_yticklabels(pids)

    ax.set_xlabel("Execution Time")

    ax.set_ylabel("Process")

    ax.set_title(
        f"{chosen} Execution Timeline"
    )

    ax.grid(
        axis="x",
        alpha=0.25
    )

    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)


# ============================================================
# AI SCHEDULER PROBABILITIES
# ============================================================

st.header("🧠 AI Scheduler Probabilities")

confidence_df = pd.DataFrame({
    "Scheduler": list(confidence.keys()),
    "Probability": [
        value * 100
        for value in confidence.values()
    ]
})

confidence_df = confidence_df.sort_values(
    "Probability",
    ascending=True
)


# ---------------- Probability table ----------------

st.dataframe(
    confidence_df,
    width="stretch",
    hide_index=True
)


# ---------------- Probability chart ----------------

fig, ax = plt.subplots(
    figsize=(10, 4)
)

ax.barh(
    confidence_df["Scheduler"],
    confidence_df["Probability"]
)

ax.set_xlabel("Probability (%)")

ax.set_ylabel("Scheduler")

ax.set_title(
    "Random Forest Scheduler Probabilities"
)

ax.set_xlim(
    0,
    100
)

ax.grid(
    axis="x",
    alpha=0.25
)

st.pyplot(
    fig,
    width="stretch"
)

plt.close(fig)


# ============================================================
# SYSTEM DECISION SUMMARY
# ============================================================

st.divider()

st.header("🔎 System Decision Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:

    st.markdown("### AI")

    st.write(
        f"Recommendation: **{ai_prediction}**"
    )

    st.write(
        f"Confidence: **{ai_confidence * 100:.2f}%**"
    )


with summary_col2:

    st.markdown("### Workload")

    st.write(
        f"Status: **{drift_status}**"
    )

    st.write(
        f"Drift Score: **{drift_score:.4f}**"
    )


with summary_col3:

    st.markdown("### Final Decision")

    st.write(
        f"Scheduler: **{selected_scheduler}**"
    )

    st.write(
        f"Method: **{selection_method}**"
    )


st.divider()

st.caption(
    "Adaptive AI CPU Scheduler — workload-aware scheduling "
    "using machine learning, confidence estimation, "
    "workload drift detection, and traditional scheduler fallback."
)