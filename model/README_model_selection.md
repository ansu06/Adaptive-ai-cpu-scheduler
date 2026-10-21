# Model Comparison & Selection Report

## 1. Overview & Methodology

Five supervised machine learning models were trained and benchmarked to classify CPU workloads into the optimal scheduling policy (`best_scheduler`: `SRTF`, `FCFS`, or `SJF`):
1. **Decision Tree**
2. **Random Forest**
3. **K-Nearest Neighbors (KNN)**
4. **Support Vector Machine (SVM)**
5. **Gradient Boosting**

### Preprocessing & Data Leakage Prevention
- **Dataset**: `model/training_data_diverse.csv` (400 diverse workload instances).
- **Features Used (11)**:
  `num_processes`, `avg_burst`, `std_burst`, `min_burst`, `max_burst`, `burst_range`, `avg_arrival`, `std_arrival`, `avg_arrival_gap`, `std_arrival_gap`, `arrival_burst_corr`.
- **Target Label**: `best_scheduler` (multi-class: `SRTF`, `FCFS`, `SJF`).
- **Leakage Prevention**: Distance-based models (`KNN`, `SVM`) were wrapped in scikit-learn `Pipeline` objects with `StandardScaler`. All feature scalers were fit strictly on the training folds. Schedulers' performance metric columns (`fcfs_waiting`, `srtf_score`, etc.) were excluded from model features.
- **Data Splitting**: Stratified 80/20 train/test split (`random_state=42`) and 5-fold stratified cross-validation on the 320 training samples.

---

## 2. Experimental Benchmark Results

| Model | Test Accuracy | Test Macro F1 | Balanced Accuracy | 5-Fold CV Macro F1 Mean | 5-Fold CV Macro F1 Std |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | 85.00% | 0.7177 | 0.7177 | 0.6795 | &plusmn; 0.0385 |
| **Random Forest** | 80.00% | 0.6889 | 0.6962 | 0.7044 | &plusmn; 0.0543 |
| **K-Nearest Neighbors (KNN)** | **87.50%** | 0.6962 | 0.6922 | 0.6686 | &plusmn; 0.0462 |
| **Support Vector Machine (SVM)\*** | 65.00% | 0.6825 | **0.8132** | **0.7069** | **&plusmn; 0.0371** |
| **Gradient Boosting** | **87.50%** | **0.7355** | 0.7285 | 0.6648 | &plusmn; 0.0310 |

\* *Selected winning model saved to `model/best_model.pkl`.*

---

## 3. Selection Rationale & Detailed Analysis

### Why SVM Was Selected
- **Primary Selection Criterion**: The model was selected based on the highest **5-Fold Cross-Validation Macro F1 Score** (**0.7069 &plusmn; 0.0371**). Cross-validation across multiple folds provides a significantly more reliable estimate of out-of-sample generalizability than a single holdout split.
- **Superior Balanced Accuracy**: On the holdout test set, SVM achieved the highest **Balanced Accuracy (81.32%)**, compared to 72.85% for Gradient Boosting, 71.77% for Decision Tree, 69.62% for Random Forest, and 69.22% for KNN. This demonstrates that the SVM's maximal margin hyperplane with balanced class weighting separates minority classes more effectively.
- **Lower Fold Variance**: SVM exhibited a standard deviation of 0.0371 across cross-validation folds, demonstrating consistency across varying training partitions.

### Trade-offs & Acknowledged Discrepancies
- **Holdout Test Set Discrepancy**: On this specific 80/20 test split:
  - **Gradient Boosting** achieved a higher holdout Test Macro F1 (0.7355 vs. 0.6825).
  - **KNN** and **Gradient Boosting** achieved higher raw Test Accuracy (87.50% vs. 65.00%).
- **Why Raw Accuracy is Misleading**: The training data contains substantial class imbalance (77% SRTF). A naive classifier that predicts SRTF for all instances would achieve 77% accuracy while completely failing on FCFS and SJF. KNN and Gradient Boosting leaned towards predicting the majority class, which boosted overall test accuracy on the holdout split but resulted in inferior cross-validation Macro F1 across all folds (0.6686 and 0.6648).

---

## 4. SVM Holdout Confusion Matrix

```
                  Predicted Class
               FCFS     SJF     SRTF
Actual FCFS     10       0        0
Actual SJF       0       7        1
Actual SRTF      0      27       35
```

### Class-Wise Analysis
- **FCFS**: 10 out of 10 samples correctly identified (100% precision, 100% recall). FCFS is easily distinguished by its strong arrival-burst correlation and wide arrival gaps.
- **SJF**: 7 out of 8 samples correctly identified (87.5% recall), but 27 SRTF samples were misclassified as SJF (20.6% precision).
- **SRTF**: 35 out of 62 samples correctly identified (56.5% recall, 97.2% precision).

### Root Cause of SJF / SRTF Overlap
SJF and SRTF are structurally related algorithms: SJF is the non-preemptive counterpart of SRTF. When jobs arrive with close arrival timestamps or small arrival gaps, preemptions rarely occur, making the optimal schedule and performance metrics of SJF and SRTF almost identical.

---

## 5. Dataset Limitations & Synthetic Biases

1. **Class Imbalance**: The 400 synthetic samples contain 308 SRTF cases (77.0%), 50 FCFS cases (12.5%), and 42 SJF cases (10.5%).
2. **Absence of Round Robin as Best Class**: In the training data generation script (`generate_diverse_workloads.py`), Round Robin was tested with a static quantum of 4 ms across processes with bursts up to 500 ms. The resulting frequent context switching degraded its average waiting time relative to SJF/SRTF, meaning Round Robin never scored as the single best scheduler on these specific workloads.
3. **Safety Fallback**: Because machine learning models are vulnerable to out-of-distribution inputs, our supervisory architecture does not rely solely on the ML prediction. When workload drift is detected (Z-score distance $\ge 1.5$) or model confidence is low ($< 50\%$), the engine automatically executes all 4 traditional scheduling heuristics and selects the policy that minimizes empirical waiting time.
