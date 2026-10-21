from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
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
    page_icon="🔲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Pure Black / Charcoal Workstation Theme & Spacing
st.markdown(
    """
    <style>
    /* Global Base & Spacing */
    .stApp {
        background-color: #080A0D !important;
        color: #F1F5F9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 2.2rem !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
        max-width: 100% !important;
    }
    [data-testid="stVerticalBlock"] {
        gap: 1.35rem !important;
    }
    [data-testid="stHorizontalBlock"] {
        gap: 1.25rem !important;
    }
    
    /* Left Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0D1015 !important;
        border-right: 1px solid #272D36 !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem !important;
        padding-left: 1.0rem !important;
        padding-right: 1.0rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        display: none;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
        gap: 6px;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 6px;
        padding: 8px 12px;
        color: #94A3B8;
        font-size: 1.05rem !important; /* ~16.8px */
        font-weight: 500;
        transition: all 0.12s ease-in-out;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
        background: #11151B;
        color: #F1F5F9;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background: #151A21 !important;
        border: 1px solid #272D36 !important;
        color: #4F8CFF !important;
        font-weight: 600 !important;
    }
    
    /* Bordered Container Panels */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #11151B !important;
        border: 1px solid #272D36 !important;
        border-radius: 8px !important;
        padding: 16px 18px !important;
        margin: 0 !important;
    }
    
    /* Metric Panels */
    .metric-panel {
        background-color: #11151B;
        border: 1px solid #272D36;
        border-radius: 8px;
        padding: 12px 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        min-height: 72px;
    }
    .metric-inner {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .metric-icon {
        width: 38px;
        height: 38px;
        border-radius: 7px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .metric-label-txt {
        font-size: 0.88rem; /* ~14px */
        color: #94A3B8;
        font-weight: 500;
        letter-spacing: 0.02em;
        margin-bottom: 2px;
    }
    .metric-val-txt {
        font-size: 1.75rem; /* ~28px */
        font-weight: 700;
        color: #F1F5F9;
        font-family: "JetBrains Mono", Consolas, monospace;
        line-height: 1.1;
    }
    .metric-sub-txt {
        font-size: 0.85rem; /* ~13.5px */
        color: #94A3B8;
        margin-top: 2px;
    }
    
    /* Panel Containers */
    .panel-heading {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .panel-title {
        font-size: 1.45rem; /* ~23px */
        font-weight: 700;
        color: #F1F5F9;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .panel-desc {
        font-size: 0.95rem; /* ~15px */
        color: #94A3B8;
        margin-top: 3px;
    }
    
    /* Status Badges */
    .status-badge {
        font-size: 0.78rem; /* ~12.5px */
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 4px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .badge-stable {
        background-color: rgba(53, 211, 153, 0.12);
        color: #35D399;
        border: 1px solid rgba(53, 211, 153, 0.35);
    }
    .badge-drift {
        background-color: rgba(255, 91, 101, 0.12);
        color: #FF5B65;
        border: 1px solid rgba(255, 91, 101, 0.35);
    }
    .badge-reeval {
        background-color: rgba(251, 191, 36, 0.12);
        color: #FBBF24;
        border: 1px solid rgba(251, 191, 36, 0.35);
    }

    /* Form Controls & Tables */
    [data-testid="stRadio"] label p, [data-testid="stRadio"] div[role="radiogroup"] label span {
        font-size: 1.0rem !important; /* 16px */
    }
    [data-testid="stSlider"] label p {
        font-size: 1.05rem !important; /* ~16.8px */
        font-weight: 600 !important;
        color: #F1F5F9 !important;
    }
    [data-testid="stSlider"] div[data-testid="stThumbValue"], 
    [data-testid="stSlider"] div[data-testid="stTickBarMin"], 
    [data-testid="stSlider"] div[data-testid="stTickBarMax"] {
        font-size: 0.92rem !important;
    }
    [data-testid="stDataFrame"] {
        font-size: 15px !important;
    }
    [data-testid="stDataFrame"] div {
        font-size: 15px !important;
    }
    [data-testid="stDataFrame"] * {
        font-size: 15px !important;
    }
    .stCaption {
        font-size: 0.92rem !important;
        color: #94A3B8 !important;
    }

    /* Streamlit overrides */
    div[data-testid="stExpander"] {
        background-color: #11151B !important;
        border: 1px solid #272D36 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stExpander"] details summary {
        color: #94A3B8 !important;
        font-size: 1.0rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LEFT SIDEBAR
# ============================================================

with st.sidebar:
    # Sidebar Header: CPU Line Icon + Title
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 18px; padding: 0 4px;">
            <div style="background: #11151B; border: 1px solid #272D36; border-radius: 8px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#4F8CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="4" y="4" width="16" height="16" rx="2"></rect>
                    <rect x="9" y="9" width="6" height="6"></rect>
                    <path d="M15 2v2M9 2v2M15 20v2M9 20v2M2 15h2M2 9h2M20 15h2M20 9h2"></path>
                </svg>
            </div>
            <div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #F1F5F9; line-height: 1.25;">
                    Adaptive AI<br><span style="color: #4F8CFF;">CPU Scheduler</span>
                </div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 3px;">
                    Workload-Aware Scheduling with Machine Learning
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Navigation items with clean Material Symbols
    nav_labels = [
        ":material/dashboard: Dashboard",
        ":material/tune: Workload Analysis",
        ":material/bar_chart: Scheduler Comparison",
        ":material/schedule: Execution Timeline",
        ":material/memory: System Monitor",
        ":material/info: About"
    ]
    
    selected_raw = st.radio(
        "Navigation",
        nav_labels,
        label_visibility="collapsed"
    )
    selected_nav = selected_raw.split(" ", 1)[-1].strip()
    
    # Project Footer info at the bottom of sidebar
    st.markdown(
        """
        <div style="margin-top: 140px; padding: 14px 4px 4px 4px; border-top: 1px solid #272D36;">
            <div style="font-size: 0.92rem; font-weight: 600; color: #F1F5F9; line-height: 1.4;">Sayan Giri</div>
            <div style="font-size: 0.92rem; font-weight: 600; color: #F1F5F9; line-height: 1.4;">Ansuman Singh</div>
            <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 6px;">VIT Vellore &bull; 2026</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# COMPACT HEADER & LIVE TELEMETRY
# ============================================================

# Real-Time Badge + Header
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.markdown(
        """
        <div style="font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; color: #35D399; margin-bottom: 4px;">
            REAL-TIME &bull; AI-POWERED &bull; ADAPTIVE
        </div>
        <div style="font-size: clamp(2.0rem, 2.8vw, 2.45rem); font-weight: 800; color: #F1F5F9; line-height: 1.15;">
            Adaptive AI <span style="color: #4F8CFF;">CPU Scheduler</span>
        </div>
        <div style="font-size: 1.0rem; color: #94A3B8; margin-top: 4px;">
            Intelligent scheduling decisions using machine learning and workload analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

with header_col2:
    now = datetime.now()
    date_str = now.strftime("%b %d, %Y")
    time_str = now.strftime("%I:%M %p")
    st.markdown(
        f"""
        <div style="background: #11151B; border: 1px solid #272D36; border-radius: 8px; padding: 8px 14px; text-align: right; float: right; margin-top: 4px;">
            <div style="font-size: 0.82rem; color: #94A3B8; display: flex; align-items: center; justify-content: flex-end; gap: 5px;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
                    <line x1="16" y1="2" x2="16" y2="6"></line>
                    <line x1="8" y1="2" x2="8" y2="6"></line>
                    <line x1="3" y1="10" x2="21" y2="10"></line>
                </svg>
                {date_str}
            </div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #F1F5F9; margin-top: 2px; font-family: 'JetBrains Mono', Consolas, monospace;">{time_str}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Live Host System Telemetry Strip (4 Compact Metric Panels)
metrics = get_system_metrics()
top_processes = get_top_processes(n=8)

# Normalize top active application process CPU percentage
active_procs = [
    p for p in top_processes
    if p["name"] and "idle" not in p["name"].lower() and p.get("pid", 0) != 0
]
top_active = active_procs[0] if active_procs else (top_processes[0] if top_processes else None)

cpu_count = os.cpu_count() or 1
if top_active:
    raw_cpu = top_active.get("cpu_percent") or 0.0
    norm_cpu = min(round(raw_cpu / cpu_count, 1), 100.0)
    top_proc_name = top_active["name"]
    top_proc_sub = f"{norm_cpu}% CPU ({cpu_count} cores)"
else:
    top_proc_name = "N/A"
    top_proc_sub = "0.0% CPU"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-panel">
            <div class="metric-inner">
                <div class="metric-icon" style="background: rgba(53, 211, 153, 0.12);">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#35D399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="4" y="4" width="16" height="16" rx="2"></rect>
                        <rect x="9" y="9" width="6" height="6"></rect>
                        <path d="M15 2v2M9 2v2M15 20v2M9 20v2M2 15h2M2 9h2M20 15h2M20 9h2"></path>
                    </svg>
                </div>
                <div>
                    <div class="metric-label-txt">CPU Usage</div>
                    <div class="metric-val-txt">{metrics["cpu_percent"]}%</div>
                </div>
            </div>
            <div>
                <svg width="44" height="18" viewBox="0 0 48 20" fill="none">
                    <path d="M1 14 L10 11 L19 16 L28 5 L37 9 L47 3" stroke="#35D399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-panel">
            <div class="metric-inner">
                <div class="metric-icon" style="background: rgba(79, 140, 255, 0.12);">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#4F8CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="2" y="6" width="20" height="12" rx="2"></rect>
                        <path d="M6 12h.01M10 12h.01M14 12h.01M18 12h.01M2 10h20M2 14h20"></path>
                    </svg>
                </div>
                <div>
                    <div class="metric-label-txt">Memory Usage</div>
                    <div class="metric-val-txt">{metrics["memory_percent"]}%</div>
                </div>
            </div>
            <div>
                <svg width="44" height="18" viewBox="0 0 48 20" fill="none">
                    <path d="M1 12 L10 9 L19 13 L28 6 L37 8 L47 4" stroke="#4F8CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-panel">
            <div class="metric-inner">
                <div class="metric-icon" style="background: rgba(168, 121, 255, 0.12);">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#A879FF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="8" y1="6" x2="21" y2="6"></line>
                        <line x1="8" y1="12" x2="21" y2="12"></line>
                        <line x1="8" y1="18" x2="21" y2="18"></line>
                        <line x1="3" y1="6" x2="3.01" y2="6"></line>
                        <line x1="3" y1="12" x2="3.01" y2="12"></line>
                        <line x1="3" y1="18" x2="3.01" y2="18"></line>
                    </svg>
                </div>
                <div>
                    <div class="metric-label-txt">Running Processes</div>
                    <div class="metric-val-txt">{metrics["process_count"]}</div>
                </div>
            </div>
            <div>
                <svg width="44" height="18" viewBox="0 0 48 20" fill="none">
                    <path d="M1 15 L10 12 L19 14 L28 8 L37 11 L47 5" stroke="#A879FF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-panel">
            <div class="metric-inner">
                <div class="metric-icon" style="background: rgba(148, 163, 184, 0.12);">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="2" y="3" width="20" height="14" rx="2"></rect>
                        <line x1="8" y1="21" x2="16" y2="21"></line>
                        <line x1="12" y1="17" x2="12" y2="21"></line>
                    </svg>
                </div>
                <div>
                    <div class="metric-label-txt">Top Process</div>
                    <div class="metric-val-txt" style="font-size: 0.95rem; word-break: break-all;">{top_proc_name}</div>
                    <div class="metric-sub-txt">{top_proc_sub}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )



# ============================================================
# MAIN WORKLOAD GENERATION & SCHEDULER ENGINE
# ============================================================

# Shared Workload Configuration Controls (Used across Dashboard & Workload Views)
if "workload_mode" not in st.session_state:
    st.session_state["workload_mode"] = "Normal Workload"
if "num_procs" not in st.session_state:
    st.session_state["num_procs"] = 5

# Compute Workload & Schedulers
if selected_nav in ["Dashboard", "Workload Analysis", "Scheduler Comparison", "Execution Timeline"]:
    
    # --------------------------------------------------------
    # SHARED SIMULATION LOGIC
    # --------------------------------------------------------
    
    # In Dashboard or Workload Analysis, controls are interactive inside the panel
    # Synchronize widget state immediately to single canonical state (eliminates 1-frame lag on rerun)
    if "wl_radio" in st.session_state:
        st.session_state["workload_mode"] = st.session_state["wl_radio"]

    if "proc_slider" in st.session_state:
        st.session_state["num_procs"] = st.session_state["proc_slider"]

    # We define default or current values
    workload_mode = st.session_state["workload_mode"]
    n = st.session_state["num_procs"]
    
    # --------------------------------------------------------
    # ROW 1: TWO-COLUMN MAIN LAYOUT (Matching Image 2)
    # --------------------------------------------------------
    if selected_nav in ["Dashboard", "Workload Analysis"]:
        col_workload, col_ai = st.columns([1, 1], gap="medium")
        
        with col_workload.container(border=True):
            tag_color = "#35D399" if workload_mode == "Normal Workload" else "#FF5B65"
            tag_text = "● NORMAL PROFILE" if workload_mode == "Normal Workload" else "▲ DRIFT TEST ACTIVE"
            tag_border = "rgba(53, 211, 153, 0.35)" if workload_mode == "Normal Workload" else "rgba(255, 91, 101, 0.35)"
            tag_bg = "rgba(53, 211, 153, 0.12)" if workload_mode == "Normal Workload" else "rgba(255, 91, 101, 0.12)"
            
            st.markdown(
                f"""
                <div class="panel-heading">
                    <div>
                        <div class="panel-title">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#4F8CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: text-bottom; margin-right: 6px;">
                                <line x1="4" y1="21" x2="4" y2="14"></line>
                                <line x1="4" y1="10" x2="4" y2="3"></line>
                                <line x1="12" y1="21" x2="12" y2="12"></line>
                                <line x1="12" y1="8" x2="12" y2="3"></line>
                                <line x1="20" y1="21" x2="20" y2="16"></line>
                                <line x1="20" y1="12" x2="20" y2="3"></line>
                                <line x1="1" y1="14" x2="7" y2="14"></line>
                                <line x1="9" y1="8" x2="15" y2="8"></line>
                                <line x1="17" y1="16" x2="23" y2="16"></line>
                            </svg>
                            Process Workload
                        </div>
                        <div class="panel-desc">Configure the workload for scheduling analysis.</div>
                    </div>
                    <div>
                        <span class="status-badge" style="background-color: {tag_bg}; color: {tag_color}; border: 1px solid {tag_border};">
                            {tag_text}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            ctrl_c1, ctrl_c2 = st.columns([1, 1], gap="medium")
            with ctrl_c1:
                st.markdown('<div style="font-size: 0.78rem; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 2px;">Workload Profile</div>', unsafe_allow_html=True)
                workload_mode = st.radio(
                    "Workload Profile",
                    ["Normal Workload", "Drift Test"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="wl_radio"
                )
                st.session_state["workload_mode"] = workload_mode
            with ctrl_c2:
                st.markdown(f'<div style="font-size: 0.78rem; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 2px;">Processes: <span style="color: #4F8CFF; font-family: monospace;">{st.session_state["num_procs"]}</span></div>', unsafe_allow_html=True)
                n = st.slider(
                    "Number of processes",
                    min_value=4,
                    max_value=20,
                    value=st.session_state["num_procs"],
                    label_visibility="collapsed",
                    key="proc_slider"
                )
                st.session_state["num_procs"] = n
            
            processes = load_processes(n=n)
            
            # Force intentionally abnormal workload for demonstration
            if workload_mode == "Drift Test":
                drift_processes = []
                for i, p in enumerate(processes):
                    drift_processes.append({
                        "pid": p["pid"],
                        "arrival": 200 + (i * 25),
                        "burst": 800 + (i * 100)
                    })
                processes = drift_processes
            
            # Render process table exactly once
            process_df = pd.DataFrame(processes).rename(
                columns={"pid": "PID", "arrival": "Arrival Time", "burst": "Burst Time"}
            )
            
            # Calculate dynamic height based on exact row count: header (35px) + rows (35px each) + border (4px)
            # Minimum dimension at 4 rows (179px); capped at 285px to maintain dashboard balance if n is large
            calc_wl_height = int((len(process_df) * 35) + 35 + 4)
            wl_table_height = min(max(calc_wl_height, 175), 285)
            
            st.dataframe(
                process_df,
                width="stretch",
                hide_index=True,
                height=wl_table_height
            )
        
        # Convert workload for AI model
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
        
        # Run Adaptive AI Decision Engine
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
            
        valid_schedulers = {"FCFS", "SJF", "SRTF", "Round Robin"}
        if selected_scheduler not in valid_schedulers:
            selected_scheduler = "SRTF"
            
        # Render AI Decision & Explainability Panel (Right Column)
        with col_ai.container(border=True):
            if drift_status == "WORKLOAD STABLE":
                badge_html = (
                    '<span class="status-badge badge-stable">'
                    '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#35D399" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">'
                    '<polyline points="20 6 9 17 4 12"></polyline></svg> STABLE WORKLOAD</span>'
                )
            else:
                badge_html = (
                    '<span class="status-badge badge-drift">'
                    '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#FF5B65" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">'
                    '<polygon points="12 2 22 20 2 20 12 2"></polygon><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg> DRIFT DETECTED</span>'
                )
                
            st.markdown(
                f"""
                <div class="panel-heading">
                    <div>
                        <div class="panel-title">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#35D399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: text-bottom; margin-right: 6px;">
                                <rect x="4" y="4" width="16" height="16" rx="2"></rect>
                                <rect x="9" y="9" width="6" height="6"></rect>
                                <line x1="9" y1="1" x2="9" y2="4"></line>
                                <line x1="15" y1="1" x2="15" y2="4"></line>
                                <line x1="9" y1="20" x2="9" y2="23"></line>
                                <line x1="15" y1="20" x2="15" y2="23"></line>
                                <line x1="20" y1="9" x2="23" y2="9"></line>
                                <line x1="20" y1="14" x2="23" y2="14"></line>
                                <line x1="1" y1="9" x2="4" y2="9"></line>
                                <line x1="1" y1="14" x2="4" y2="14"></line>
                            </svg>
                            AI Decision & Explainability
                        </div>
                        <div class="panel-desc">Model prediction with confidence and workload analysis.</div>
                    </div>
                    <div>{badge_html}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            ai_sub1, ai_sub2 = st.columns([1, 1.25], gap="medium")
            
            with ai_sub1:
                st.markdown(
                    f"""
                    <div style="background: #151A21; border: 1px solid #272D36; border-radius: 8px; padding: 10px 12px; margin-bottom: 8px;">
                        <div style="font-size: 0.80rem; color: #94A3B8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.04em;">Recommended Scheduler</div>
                        <div style="display: flex; align-items: baseline; gap: 10px; margin-top: 3px;">
                            <div style="font-size: 1.85rem; font-weight: 800; color: #35D399; font-family: 'JetBrains Mono', Consolas, monospace; line-height: 1;">{ai_prediction}</div>
                            <div style="background: rgba(53, 211, 153, 0.15); border: 1px solid rgba(53, 211, 153, 0.35); color: #35D399; font-size: 0.75rem; font-weight: 700; padding: 3px 7px; border-radius: 4px;">AI Prediction</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.markdown(
                        f"""
                        <div style="background: #151A21; border: 1px solid #272D36; border-radius: 8px; padding: 8px 10px; min-height: 64px;">
                            <div style="font-size: 0.78rem; color: #94A3B8;">Confidence</div>
                            <div style="font-size: 1.35rem; font-weight: 700; color: #F1F5F9; font-family: 'JetBrains Mono', Consolas, monospace; margin: 2px 0;">{ai_confidence * 100:.1f}%</div>
                            <div style="background: #272D36; height: 4px; border-radius: 2px; overflow: hidden;">
                                <div style="background: #35D399; width: {min(max(ai_confidence * 100, 5), 100)}%; height: 100%;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with sc2:
                    d_color = "#35D399" if drift_status == "WORKLOAD STABLE" else "#FF5B65"
                    d_txt = "No Drift" if drift_status == "WORKLOAD STABLE" else "Drift"
                    st.markdown(
                        f"""
                        <div style="background: #151A21; border: 1px solid #272D36; border-radius: 8px; padding: 8px 10px; min-height: 64px;">
                            <div style="font-size: 0.78rem; color: #94A3B8;">Drift Status</div>
                            <div style="font-size: 1.25rem; font-weight: 700; color: {d_color}; margin: 2px 0;">{d_txt}</div>
                            <div style="font-size: 0.75rem; color: #94A3B8;">Score: {drift_score:.3f} (Lim: 1.50)</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                decision_upper = str(decision).upper()
                dec_color = "#35D399" if "TRUST" in decision_upper else ("#FF5B65" if "FALLBACK" in decision_upper else "#FBBF24")
                st.markdown(
                    f"""
                    <div style="background: #151A21; border: 1px solid #272D36; border-radius: 8px; padding: 9px 12px; margin-top: 8px;">
                        <div style="font-size: 0.78rem; color: #94A3B8;">Decision</div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: {dec_color}; margin-top: 2px;">{decision}</div>
                        <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 3px; line-height: 1.35;">{reason}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with ai_sub2:
                st.markdown(
                    """
                    <div style="font-size: 0.95rem; font-weight: 700; color: #F1F5F9; margin-bottom: 6px; padding-left: 1px;">
                        SVM Scheduler Prediction Probabilities
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                conf_df = pd.DataFrame({
                    "Scheduler": list(confidence.keys()),
                    "Probability (%)": [
                        round(value * 100, 2)
                        for value in confidence.values()
                    ]
                }).sort_values("Probability (%)", ascending=True)
                
                fig, ax = plt.subplots(figsize=(5.2, 3.35))
                fig.patch.set_facecolor("#11151B")
                ax.set_facecolor("#11151B")
                
                class_colors = {
                    "SRTF": "#4F8CFF",
                    "SJF": "#A879FF",
                    "FCFS": "#FFAD66",
                    "Round Robin": "#38BDF8"
                }
                bar_colors = [class_colors.get(s, "#4F8CFF") for s in conf_df["Scheduler"]]
                
                bars = ax.barh(
                    conf_df["Scheduler"],
                    conf_df["Probability (%)"],
                    color=bar_colors,
                    height=0.45,
                    edgecolor="#272D36",
                    linewidth=0.8
                )
                
                ax.set_xlabel("Probability (%)", color="#94A3B8", fontsize=11.5)
                ax.set_xlim(0, 105)
                ax.set_xticks([0, 25, 50, 75, 100])
                ax.tick_params(colors="#94A3B8", labelsize=11.5)
                ax.grid(axis="x", linestyle="--", alpha=0.15, color="#94A3B8")
                for spine in ax.spines.values():
                    spine.set_color("#272D36")
                    
                for bar in bars:
                    w = bar.get_width()
                    ax.text(
                        min(w + 2.0, 95),
                        bar.get_y() + bar.get_height() / 2,
                        f"{w:.1f}%",
                        va="center",
                        color="#F1F5F9",
                        fontsize=11.0,
                        fontweight="bold"
                    )
                
                plt.tight_layout(pad=0.4)
                st.pyplot(fig, width="stretch")
                plt.close(fig)
                
    else:
        # Load processes for other views
        processes = load_processes(n=n)
        if workload_mode == "Drift Test":
            drift_processes = []
            for i, p in enumerate(processes):
                drift_processes.append({
                    "pid": p["pid"],
                    "arrival": 200 + (i * 25),
                    "burst": 800 + (i * 100)
                })
            processes = drift_processes
        ai_workload = pd.DataFrame({
            "Job Id": [int(str(p["pid"]).replace("P", "")) for p in processes],
            "Burst time": [p["burst"] for p in processes],
            "Arrival Time": [p["arrival"] for p in processes]
        })
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
        
    # --------------------------------------------------------
    # RUN TRADITIONAL SCHEDULERS
    # --------------------------------------------------------
    traces = {}
    summary_rows = []
    
    o_fcfs, r_fcfs, t_fcfs = fcfs(processes)
    summary_rows.append(summarize("FCFS", o_fcfs, r_fcfs))
    traces["FCFS"] = t_fcfs
    
    o_sjf, r_sjf, t_sjf = sjf(processes)
    summary_rows.append(summarize("SJF", o_sjf, r_sjf))
    traces["SJF"] = t_sjf
    
    o_srtf, r_srtf, t_srtf = srtf(processes)
    summary_rows.append(summarize("SRTF", o_srtf, r_srtf))
    traces["SRTF"] = t_srtf
    
    o_rr, r_rr, t_rr = round_robin_metrics(processes, quantum=4)
    summary_rows.append(summarize("Round Robin (q=4)", o_rr, r_rr))
    traces["Round Robin"] = t_rr
    
    summary_df = pd.DataFrame(summary_rows)
    
    # --------------------------------------------------------
    # ROW 2: TWO-COLUMN LOWER LAYOUT (Matching Image 2 on Dashboard)
    # --------------------------------------------------------
    if selected_nav == "Dashboard":
        col_sched, col_timeline = st.columns([1, 1], gap="medium")
        
        with col_sched.container(border=True):
            st.markdown(
                """
                <div class="panel-heading">
                    <div>
                        <div class="panel-title">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#4F8CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: text-bottom; margin-right: 6px;">
                                <line x1="18" y1="20" x2="18" y2="10"></line>
                                <line x1="12" y1="20" x2="12" y2="4"></line>
                                <line x1="6" y1="20" x2="6" y2="14"></line>
                            </svg>
                            Scheduler Comparison
                        </div>
                        <div class="panel-desc">Performance metrics for all scheduling algorithms.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            display_df = summary_df.drop(
                columns=["order"],
                errors="ignore"
            ).rename(
                columns={
                    "scheduler": "Scheduler",
                    "avg_waiting": "Avg Waiting Time ↓",
                    "avg_turnaround": "Avg Turnaround Time",
                    "avg_response": "Avg Response Time"
                }
            )
            
            for c in ["Avg Waiting Time ↓", "Avg Turnaround Time", "Avg Response Time"]:
                display_df[c] = display_df[c].map(lambda x: f"{x:.2f}")
                
            # Dynamic height based on exact row count: 4 rows (179px) with no blank rows
            calc_sched_height = int((len(display_df) * 35) + 35 + 4)
            sched_table_height = min(max(calc_sched_height, 175), 285)
            
            st.dataframe(
                display_df,
                width="stretch",
                hide_index=True,
                height=sched_table_height
            )
            
            st.caption("Lower values indicate superior policy efficiency &bull; Waiting: ready queue duration &bull; Turnaround: arrival to exit.")
            
        with col_timeline.container(border=True):
            st.markdown(
                f"""
                <div class="panel-heading">
                    <div>
                        <div class="panel-title">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#35D399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: text-bottom; margin-right: 6px;">
                                <circle cx="12" cy="12" r="10"></circle>
                                <polyline points="12 6 12 12 16 14"></polyline>
                            </svg>
                            Execution Timeline
                        </div>
                        <div class="panel-desc">Gantt chart for the selected scheduler ({selected_scheduler}).</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            target_trace_key = selected_scheduler if selected_scheduler in traces else "SRTF"
            segments = traces.get(target_trace_key, [])
            
            if segments:
                pids = sorted(list(set(seg["pid"] for seg in segments)))
                
                palette = ["#4F8CFF", "#A879FF", "#35D399", "#FFAD66", "#FF5B65", "#38BDF8", "#F472B6", "#FACC15"]
                pid_color = {pid: palette[i % len(palette)] for i, pid in enumerate(pids)}
                
                chips_html = "".join([
                    f"<span style='display:inline-flex; align-items:center; gap:5px; font-size:0.85rem; color:#F1F5F9; margin-right:12px; margin-bottom: 4px;'>"
                    f"<span style='width:10px; height:10px; background:{pid_color[pid]}; border-radius:2px; display:inline-block;'></span> {pid}</span>"
                    for pid in pids
                ])
                st.markdown(f"<div style='margin-bottom: 6px; padding-left: 1px;'>{chips_html}</div>", unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(7.8, 2.75))
                fig.patch.set_facecolor("#11151B")
                ax.set_facecolor("#11151B")
                
                max_time = max([seg["end"] for seg in segments]) if segments else 100
                
                for seg in segments:
                    start = seg["start"]
                    dur = seg["end"] - seg["start"]
                    if dur <= 0:
                        continue
                    c = pid_color.get(seg["pid"], "#4F8CFF")
                    ax.broken_barh([(start, dur)], (2, 5.5), facecolors=c, edgecolor="#080A0D", linewidth=1.2)
                    
                    if dur >= max_time * 0.035:
                        ax.text(
                            start + dur / 2.0,
                            4.75,
                            str(seg["pid"]),
                            ha="center",
                            va="center",
                            color="#080A0D",
                            fontsize=10.5,
                            fontweight="bold"
                        )
                
                ax.set_yticks([4.75])
                ax.set_yticklabels(["CPU"], color="#F1F5F9", fontsize=13, fontweight="bold")
                ax.set_xlabel("Time (ms)", color="#94A3B8", fontsize=12.0)
                ax.set_xlim(0, max_time * 1.02)
                ax.tick_params(colors="#94A3B8", labelsize=11.5)
                ax.grid(axis="x", linestyle="--", alpha=0.15, color="#94A3B8")
                for spine in ax.spines.values():
                    spine.set_color("#272D36")
                    
                plt.tight_layout(pad=0.4)
                st.pyplot(fig, width="stretch")
                plt.close(fig)

    # Dedicated View: Workload Analysis
    elif selected_nav == "Workload Analysis":
        with st.container(border=True):
            st.markdown("#### Workload Statistical Feature Inspection & Drift Safety")
            st.caption("Detailed statistical parameters extracted from simulated workload and evaluated against baseline distribution.")
            
            burst_vals = [p["burst"] for p in processes]
            arrival_vals = [p["arrival"] for p in processes]
            
            feat_data = {
                "Feature Metric": [
                    "Job Count (N)",
                    "Mean Burst Time",
                    "Burst Time Std Dev",
                    "Mean Arrival Time",
                    "Arrival Time Std Dev",
                    "Burst-to-Arrival Ratio"
                ],
                "Current Workload": [
                    f"{len(processes)}",
                    f"{np.mean(burst_vals):.2f} ms",
                    f"{np.std(burst_vals):.2f} ms",
                    f"{np.mean(arrival_vals):.2f} ms",
                    f"{np.std(arrival_vals):.2f} ms",
                    f"{np.mean(burst_vals)/(np.mean(arrival_vals)+1e-5):.2f}"
                ],
                "Drift Evaluation": [
                    "Within Range" if len(processes) <= 20 else "Outlier",
                    "Baseline Normal" if drift_score < 1.50 else "High Deviation",
                    "Baseline Normal" if drift_score < 1.50 else "High Variance",
                    "Baseline Normal" if drift_score < 1.50 else "Shifted Arrival",
                    "Baseline Normal" if drift_score < 1.50 else "Shifted Spread",
                    "Normal Profile" if drift_score < 1.50 else "Anomalous Ratio"
                ]
            }
            st.dataframe(pd.DataFrame(feat_data), width="stretch", hide_index=True)

    # Dedicated View: Scheduler Comparison
    elif selected_nav == "Scheduler Comparison":
        col1, col2 = st.columns([1, 1], gap="medium")
        with col1.container(border=True):
            st.markdown("#### Performance Metric Summary")
            display_df = summary_df.drop(columns=["order"], errors="ignore").rename(
                columns={
                    "scheduler": "Scheduler",
                    "avg_waiting": "Avg Waiting Time ↓",
                    "avg_turnaround": "Avg Turnaround Time",
                    "avg_response": "Avg Response Time"
                }
            )
            for c in ["Avg Waiting Time ↓", "Avg Turnaround Time", "Avg Response Time"]:
                display_df[c] = display_df[c].map(lambda x: f"{x:.2f}")
            st.dataframe(display_df, width="stretch", hide_index=True)
            
        with col2.container(border=True):
            st.markdown("#### Comparative Benchmark Across Policies")
            sched_names = summary_df["scheduler"].tolist()
            waiting = summary_df["avg_waiting"].tolist()
            turnaround = summary_df["avg_turnaround"].tolist()
            response = summary_df["avg_response"].tolist()
            
            fig, ax = plt.subplots(figsize=(6.5, 3.5))
            fig.patch.set_facecolor("#11151B")
            ax.set_facecolor("#11151B")
            x = np.arange(len(sched_names))
            w = 0.24
            ax.bar(x - w, waiting, w, label="Waiting", color="#4F8CFF", edgecolor="#272D36")
            ax.bar(x, turnaround, w, label="Turnaround", color="#A879FF", edgecolor="#272D36")
            ax.bar(x + w, response, w, label="Response", color="#35D399", edgecolor="#272D36")
            ax.set_xticks(x)
            ax.set_xticklabels(sched_names, color="#F1F5F9", fontsize=11.5)
            ax.tick_params(colors="#94A3B8", labelsize=11.0)
            ax.set_ylabel("Time (ms)", color="#94A3B8", fontsize=12.0)
            ax.legend(facecolor="#151A21", edgecolor="#272D36", labelcolor="#F1F5F9", fontsize=11.0)
            ax.grid(axis="y", linestyle="--", alpha=0.15, color="#94A3B8")
            for spine in ax.spines.values():
                spine.set_color("#272D36")
            plt.tight_layout(pad=0.4)
            st.pyplot(fig, width="stretch")
            plt.close(fig)

    # Dedicated View: Execution Timeline
    elif selected_nav == "Execution Timeline":
        with st.container(border=True):
            st.markdown("#### Interactive Scheduling Execution Timeline")
            st.caption("Inspect CPU core allocation and process preemption across individual scheduling policies.")
            
            chosen_policy = st.selectbox(
                "Select Scheduling Policy to Inspect",
                list(traces.keys()),
                index=list(traces.keys()).index(selected_scheduler) if selected_scheduler in traces else 0
            )
            
            segments = traces.get(chosen_policy, [])
            if segments:
                pids = sorted(list(set(seg["pid"] for seg in segments)))
                palette = ["#4F8CFF", "#A879FF", "#35D399", "#FFAD66", "#FF5B65", "#38BDF8", "#F472B6", "#FACC15"]
                pid_color = {pid: palette[i % len(palette)] for i, pid in enumerate(pids)}
                
                chips_html = "".join([
                    f"<span style='display:inline-flex; align-items:center; gap:5px; font-size:0.85rem; color:#F1F5F9; margin-right:12px; margin-bottom: 4px;'>"
                    f"<span style='width:10px; height:10px; background:{pid_color[pid]}; border-radius:2px; display:inline-block;'></span> {pid}</span>"
                    for pid in pids
                ])
                st.markdown(f"<div style='margin-bottom: 6px;'>{chips_html}</div>", unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(9.5, 3.0))
                fig.patch.set_facecolor("#11151B")
                ax.set_facecolor("#11151B")
                max_time = max([seg["end"] for seg in segments]) if segments else 100
                
                for seg in segments:
                    start = seg["start"]
                    dur = seg["end"] - seg["start"]
                    if dur <= 0:
                        continue
                    ax.broken_barh([(start, dur)], (2, 5.5), facecolors=pid_color.get(seg["pid"], "#4F8CFF"), edgecolor="#080A0D", linewidth=1.2)
                    if dur >= max_time * 0.035:
                        ax.text(start + dur / 2.0, 4.75, str(seg["pid"]), ha="center", va="center", color="#080A0D", fontsize=10.5, fontweight="bold")
                        
                ax.set_yticks([4.75])
                ax.set_yticklabels(["CPU Core"], color="#F1F5F9", fontsize=13, fontweight="bold")
                ax.set_xlabel("Time (ms)", color="#94A3B8", fontsize=12.0)
                ax.set_xlim(0, max_time * 1.02)
                ax.tick_params(colors="#94A3B8", labelsize=11.5)
                ax.grid(axis="x", linestyle="--", alpha=0.15, color="#94A3B8")
                for spine in ax.spines.values():
                    spine.set_color("#272D36")
                plt.tight_layout(pad=0.4)
                st.pyplot(fig, width="stretch")
                plt.close(fig)

# --------------------------------------------------------
# SECONDARY NAVIGATION VIEWS (Functional Sidebar)
# --------------------------------------------------------
elif selected_nav == "System Monitor":
    with st.container(border=True):
        st.markdown(
            """
            <div class="panel-heading">
                <div class="panel-title">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#4F8CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: text-bottom; margin-right: 5px;">
                        <rect x="2" y="3" width="20" height="14" rx="2"></rect>
                        <line x1="8" y1="21" x2="16" y2="21"></line>
                        <line x1="12" y1="17" x2="12" y2="21"></line>
                    </svg>
                    Host Operating System Telemetry
                </div>
                <div class="panel-desc">Real-time hardware metrics and active process inspection via psutil.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Total Host CPU", f"{metrics['cpu_percent']}%")
        with m_col2:
            st.metric("Virtual Memory Allocated", f"{metrics['memory_percent']}%")
        with m_col3:
            st.metric("Active Process Threads", f"{metrics['process_count']}")
            
        st.markdown("##### Top Active Host Processes")
        if top_processes:
            top_df = pd.DataFrame(top_processes).rename(
                columns={"pid": "PID", "name": "Executable Name", "cpu_percent": "Reported CPU %"}
            )
            st.dataframe(top_df, width="stretch", hide_index=True)
        else:
            st.info("No host process information retrieved.")

elif selected_nav == "About":
    with st.container(border=True):
        st.markdown(
            """
            <div class="panel-heading">
                <div class="panel-title">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#4F8CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: text-bottom; margin-right: 5px;">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="16" x2="12" y2="12"></line>
                        <line x1="12" y1="8" x2="12.01" y2="8"></line>
                    </svg>
                    Project Background & System Architecture
                </div>
                <div class="panel-desc">Adaptive AI CPU Scheduler &bull; Workload-Aware Supervisory Decision Architecture.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            #### System Overview
            The **Adaptive AI CPU Scheduler** is a supervisory operating-system simulation platform. It uses a trained **Support Vector Machine (SVM)** classifier to evaluate process burst times, arrival rates, and distribution variance, dynamically recommending optimal CPU scheduling policies (`FCFS`, `SJF`, `SRTF`).
            
            #### Key Architectural Components
            - **Machine Learning Inference**: An SVM pipeline with StandardScaler and probability calibration predicting optimal scheduler targets.
            - **Distribution Drift Detection**: Continuous Z-score distance measurement against training distribution parameters with a safety threshold of `1.50`.
            - **Automated Fallback Engine**: If drift or low classification confidence is detected, the supervisory engine automatically runs empirical performance evaluations across all traditional algorithms to select the minimum waiting time policy.
            
            #### Academic Details
            - **Contributors**: Sayan Giri & Ansuman Singh
            - **Institution**: Vellore Institute of Technology (VIT), Vellore &bull; 2026
            - **Supervision**: Operating Systems & Machine Learning Capstone
            """
        )

# ============================================================
# TECHNICAL FOOTER
# ============================================================

st.caption(
    "Adaptive AI CPU Scheduler &bull; Pure Black Graphite Workstation UI &bull; "
    "Workload-aware supervisory scheduling simulator &bull; Sayan Giri & Ansuman Singh &bull; VIT Vellore 2026."
)
