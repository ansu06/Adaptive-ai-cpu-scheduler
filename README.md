
# 🧠 Adaptive AI CPU Scheduler
------------------------------------------------------------------------

## 📌 Overview

Traditional CPU schedulers such as **FCFS, SJF, SRTF, and Round Robin**
follow fixed scheduling policies. However, process workloads can change
significantly over time.

This project adds an **adaptive AI decision layer** on top of
traditional CPU scheduling.

The system:

1.  📊 Analyzes the current process workload
2.  🧮 Extracts workload-level statistical features
3.  🤖 Uses a Random Forest model to predict a suitable scheduler
4.  🎯 Estimates prediction confidence
5.  🔍 Detects workload drift
6.  🛡️ Trusts the AI only when the workload is sufficiently familiar
7.  🔄 Falls back to traditional scheduler evaluation when necessary

> **Core idea:** Don't blindly trust ML. Use ML when it is reliable, and
> use traditional scheduling when it is not.

------------------------------------------------------------------------

## ✨ Key Features

  -----------------------------------------------------------------------
  Feature                             Description
  ----------------------------------- -----------------------------------
  🤖 **AI Scheduler Prediction**      Random Forest-based scheduler
                                      classification

  🎯 **Confidence-Aware Decisions**   AI recommendations are evaluated
                                      using prediction confidence

  🔍 **Workload Drift Detection**     Detects when the current workload
                                      differs from the training
                                      distribution

  🛡️ **Safe Fallback**                Uses traditional scheduling when AI
                                      should not be trusted

  ⚡ **Multiple Schedulers**          FCFS, SJF, SRTF and Round Robin

  📈 **Performance Comparison**       Waiting, turnaround and response
                                      time comparison

  🖥️ **Live Monitoring**              CPU, memory and process information
                                      through `psutil`

  📊 **Interactive Dashboard**        Streamlit-based visualization and
                                      decision monitoring

  🧪 **Drift Demonstration**          Dedicated workload mode for testing
                                      the fallback mechanism
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 🏗️ Architecture

``` text
                    ┌─────────────────────┐
                    │   Process Workload  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Extraction  │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
       ┌─────────────────┐           ┌──────────────────┐
       │ Random Forest   │           │ Drift Detection  │
       │ Prediction      │           │                  │
       └────────┬────────┘           └────────┬─────────┘
                │                             │
                └──────────────┬──────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Adaptive Decision   │
                    │       Engine        │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
   │  TRUST AI   │      │ RE-EVALUATE │      │   FALLBACK  │
   │             │      │             │      │             │
   │ AI Scheduler│      │ Traditional │      │ Traditional │
   │             │      │ Evaluation  │      │ Evaluation  │
   └─────────────┘      └──────┬──────┘      └──────┬──────┘
                               │                      │
                               └──────────┬───────────┘
                                          ▼
                              ┌─────────────────────┐
                              │ Selected Scheduler  │
                              └─────────────────────┘
```

------------------------------------------------------------------------

## 🧠 Adaptive Decision Logic

The system follows three possible paths.

### 🟢 1. TRUST AI

When:

-   The workload is stable
-   Maximum model confidence is **≥ 50%**

The predicted scheduler is selected.

``` text
Stable Workload
      +
Confidence ≥ 50%
      ↓
  TRUST AI
      ↓
AI Scheduler
```

### 🟡 2. RE-EVALUATE

When:

-   The workload is stable
-   Maximum model confidence is **\< 50%**

The system evaluates the traditional schedulers and selects the one with
the lowest average waiting time.

``` text
Stable Workload
      +
Confidence < 50%
      ↓
 RE-EVALUATE
      ↓
FCFS / SJF / SRTF / RR
      ↓
Lowest Average Waiting Time
```

### 🔴 3. DRIFT FALLBACK

When workload drift is detected, the AI recommendation is not trusted.

``` text
Drift Detected
      ↓
   FALLBACK
      ↓
Evaluate Traditional Schedulers
      ↓
Lowest Average Waiting Time
```

------------------------------------------------------------------------

## ⚙️ Scheduling Algorithms

The project evaluates four traditional CPU scheduling algorithms.

  -----------------------------------------------------------------------
  Algorithm               Type                    Main Idea
  ----------------------- ----------------------- -----------------------
  **FCFS**                Non-preemptive          Execute processes in
                                                  arrival order

  **SJF**                 Non-preemptive          Select the shortest
                                                  available job

  **SRTF**                Preemptive              Select the process with
                                                  the shortest remaining
                                                  time

  **Round Robin**         Preemptive              Execute processes using
                                                  a fixed time quantum
  -----------------------------------------------------------------------

### 📊 Evaluation Metrics

The schedulers are compared using:

-   **Average Waiting Time**
-   **Average Turnaround Time**
-   **Average Response Time**

------------------------------------------------------------------------

## 🤖 Machine Learning

The AI component uses a **Random Forest classifier** to predict the
scheduler associated with a workload pattern.

### 📐 Workload Features

The model uses **11 workload-level features**:

  -----------------------------------------------------------------------
  Feature                             Description
  ----------------------------------- -----------------------------------
  `num_processes`                     Number of processes

  `avg_burst`                         Average burst time

  `std_burst`                         Standard deviation of burst time

  `min_burst`                         Minimum burst time

  `max_burst`                         Maximum burst time

  `burst_range`                       Maximum burst time − minimum burst
                                      time

  `avg_arrival`                       Average arrival time

  `std_arrival`                       Standard deviation of arrival time

  `avg_arrival_gap`                   Average gap between consecutive
                                      arrivals

  `std_arrival_gap`                   Standard deviation of arrival gaps

  `arrival_burst_corr`                Correlation between arrival time
                                      and burst time
  -----------------------------------------------------------------------

### 🗂️ Model Files

``` text
model/
├── random_forest_scheduler.pkl
├── training_data_diverse.csv
└── training_data_random.csv
```

The trained model is already included in the repository, so **retraining
is not required** to run the dashboard.

------------------------------------------------------------------------

## 🔍 Workload Drift Detection

The system compares the current workload against the training workload
distribution.

For each feature:

``` text
distance = |current_value − training_mean| / training_std
```

The overall drift score is:

``` text
D = mean(|xᵢ − μᵢ| / σᵢ)
```

### Current Threshold

``` text
D < 1.5  →  🟢 WORKLOAD STABLE
D ≥ 1.5  →  🔴 DRIFT DETECTED
```

When drift is detected:

``` text
AI Recommendation
       ↓
   Not Trusted
       ↓
Traditional Scheduler Evaluation
       ↓
Final Scheduler
```

> **Note:** The current drift detector is a lightweight
> standardized-distance heuristic rather than a formal statistical
> concept-drift test.

------------------------------------------------------------------------

## 📊 Dashboard

The Streamlit dashboard provides a visual view of the entire decision
process.

### Dashboard includes

-   🖥️ CPU utilization
-   💾 Memory utilization
-   🔢 Process count
-   📋 Input workload
-   🤖 AI scheduler recommendation
-   🎯 AI confidence
-   🔍 Drift score
-   🟢/🔴 Workload stability
-   🧠 Adaptive decision
-   🎯 Selected scheduler
-   📊 Scheduler performance comparison
-   📈 Execution timeline
-   📉 AI probability distribution

### Demonstration Modes

#### 🟢 Normal Workload

Runs the adaptive scheduler on a normal workload and demonstrates the AI
decision pipeline.

#### 🔴 Drift Test

Creates an intentionally different workload distribution to demonstrate:

``` text
DRIFT DETECTED
      ↓
FALLBACK
      ↓
TRADITIONAL SCHEDULER
```

------------------------------------------------------------------------

## 🧪 Example

A typical drift-fallback scenario can look like:

``` text
AI Recommendation : FCFS
AI Confidence     : 63.93%
Drift Score       : 6.3636
Drift Status      : DRIFT DETECTED

Decision           : FALLBACK
Selection Method   : DRIFT FALLBACK

Reason:
Workload drift detected.
AI recommendation is not trusted.
```

The important point is that **high AI confidence alone does not override
workload drift**.

------------------------------------------------------------------------

## 🧩 Project Structure

``` text
Adaptive-ai-cpu-scheduler/
│
├── 📄 app.py
├── 📄 process_data.csv
├── 📄 requirements.txt
├── 📄 .gitignore
├── 📄 README.md
│
├── 📁 scheduler/
│   └── schedulers.py
│
├── 📁 monitoring/
│   └── system_monitor.py
│
├── 📁 drift/
│   ├── __init__.py
│   ├── test_drift.py
│   └── workload_detector.py
│
└── 📁 model/
    ├── adaptive_scheduler.py
    ├── analyze_diverse_data.py
    ├── analyze_training_data.py
    ├── decision_engine.py
    ├── evaluate_schedulers.py
    ├── generate_diverse_workloads.py
    ├── generate_training_data.py
    ├── predict.py
    ├── random_forest_scheduler.pkl
    ├── test_adaptive_drift.py
    ├── train_model.py
    ├── training_data_diverse.csv
    ├── training_data_random.csv
    └── workload_features.py
```

------------------------------------------------------------------------

## 🛠️ Tech Stack

  Technology            Purpose
  --------------------- --------------------------------
  🐍 **Python**         Core implementation
  🌲 **Scikit-learn**   Random Forest machine learning
  🐼 **Pandas**         Data processing
  🔢 **NumPy**          Numerical computation
  💾 **Joblib**         Model serialization
  🖥️ **psutil**         Live OS monitoring
  📊 **Matplotlib**     Visualization
  🎈 **Streamlit**      Interactive dashboard
:::

------------------------------------------------------------------------

## 🚀 Installation

### 1. Clone the repository

``` bash
git clone https://github.com/ansu06/Adaptive-ai-cpu-scheduler.git
cd Adaptive-ai-cpu-scheduler
```

### 2. Create a virtual environment

**Windows --- PowerShell**

``` powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ Usage

Start the Streamlit dashboard:

``` bash
python -m streamlit run app.py
```

Then open the local Streamlit URL displayed in the terminal.

### Optional: Retrain the model

If you modify the training data or model pipeline:

``` bash
python model/train_model.py
```

------------------------------------------------------------------------

## 🧠 Core Modules

### `model/predict.py`

Handles:

-   Workload feature extraction
-   Random Forest prediction
-   Class probabilities
-   AI confidence

### `drift/workload_detector.py`

Handles:

-   Training workload statistics
-   Current workload feature extraction
-   Drift score calculation
-   Workload stability detection

### `model/decision_engine.py`

Combines:

``` text
AI Prediction
      +
AI Confidence
      +
Drift Status
      ↓
Adaptive Decision
```

Possible decisions:

``` text
TRUST AI
RE-EVALUATE
FALLBACK
```

### `model/adaptive_scheduler.py`

Executes the final scheduling decision and, when necessary, evaluates:

``` text
FCFS
SJF
SRTF
Round Robin
```

------------------------------------------------------------------------

## 🔬 Project Motivation

The project is based on the observation that **process workloads are
dynamic**, while traditional scheduling policies do not learn from
historical workload patterns.

The broader project direction combines:

``` text
Historical Workload
        +
Machine Learning
        +
Live OS Monitoring
        +
Workload Drift Detection
        ↓
Adaptive CPU Scheduling
```

This follows the project's research direction of making ML-assisted
scheduling **workload-aware and confidence-aware**, while retaining
traditional scheduling as a fallback. fileciteturn0file0L79-L89

------------------------------------------------------------------------

## 💡 What Makes It Different?

### 1. 🎯 Confidence-Aware Scheduling

The system does not blindly trust an ML prediction.

``` text
ML Prediction
     ↓
Confidence Check
   ↙       ↘
High       Low
 ↓          ↓
AI       Traditional
         Scheduling
```

This confidence-aware decision flow is part of the project's proposed
novelty. fileciteturn0file2L104-L135

### 2. 🪶 Lightweight ML

Instead of requiring complex reinforcement-learning architectures, the
project uses a lightweight Random Forest approach intended to be easier
to train, interpret and apply to tabular workload data.
fileciteturn0file2L124-L135

### 3. 🔄 Traditional Scheduler Fallback

When AI confidence or workload similarity is insufficient, the system
can return to traditional scheduling instead of forcing an ML decision.

### 4. 📊 AI vs Traditional Comparison

The project evaluates scheduling approaches using:

-   Waiting Time
-   Turnaround Time
-   Response Time
-   CPU Utilization

These metrics are part of the project's stated evaluation objectives.
fileciteturn0file2L53-L70

------------------------------------------------------------------------

## ⚠️ Limitations

-   The machine-learning training data is synthetic.
-   The current training distribution is not perfectly balanced across
    scheduler classes.
-   Classification performance can vary depending on workload
    distribution.
-   The drift detector uses a lightweight standardized-distance
    heuristic.
-   The Drift Test mode is designed for demonstration rather than
    production workload generation.
-   The system provides scheduling recommendations; it does not replace
    the operating system's kernel scheduler.

------------------------------------------------------------------------

## 🔮 Future Improvements

-   [ ] Larger and more balanced training datasets
-   [ ] Real-world CPU workload traces
-   [ ] Improved SJF classification
-   [ ] More robust statistical drift detection
-   [ ] Online / incremental model learning
-   [ ] Additional CPU scheduling algorithms
-   [ ] Reinforcement-learning-based scheduling
-   [ ] Automated model retraining after confirmed drift
-   [ ] More detailed CPU utilization analytics
-   [ ] Direct integration with OS-level scheduling mechanisms

------------------------------------------------------------------------

## 👨‍💻 Team

### Ansuman Singh & Sayan Giri

**Adaptive AI-Based Intelligent CPU Scheduler**

Operating Systems + Machine Learning
:::

------------------------------------------------------------------------

## 📚 Academic Context

This project explores the intersection of **Operating Systems and
Machine Learning**, focusing on workload-aware scheduling, ML-assisted
recommendations, confidence estimation and fallback mechanisms.

The project presentation describes the overall pipeline as historical
data → preprocessing → Random Forest → live OS processes → workload
prediction → confidence estimation → scheduling recommendation →
dashboard. fileciteturn0file2L73-L102

------------------------------------------------------------------------

### ⭐ If you found this project interesting, consider giving the repository a star!

**Built with Python • Machine Learning • Operating Systems • Streamlit**
:::
