import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    make_scorer,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "training_data_diverse.csv"
RESULT_CSV = BASE_DIR / "model_comparison_results.csv"
SELECTED_MODEL_PATH = BASE_DIR / "best_model.pkl"
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_FILE)

# Feature columns – must match those used throughout the project
FEATURES = [
    "num_processes",
    "avg_burst",
    "std_burst",
    "min_burst",
    "max_burst",
    "burst_range",
    "avg_arrival",
    "std_arrival",
    "avg_arrival_gap",
    "std_arrival_gap",
    "arrival_burst_corr",
]

X = df[FEATURES]
y = df["best_scheduler"]

# ---------------------------------------------------------------------------
# Train / test split (stratified 80/20)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

# ---------------------------------------------------------------------------
# Define models (with balanced class weight where appropriate)
# ---------------------------------------------------------------------------
models = {
    "DecisionTree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

# ---------------------------------------------------------------------------
# Helper to evaluate a model (pipeline includes scaling for non‑tree models)
# ---------------------------------------------------------------------------
def evaluate_model(name, estimator, X_tr, y_tr, X_te, y_te):
    # For distance‑based models we scale; tree‑based models can skip scaling.
    if name in {"KNN", "SVM"}:
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    else:
        pipeline = Pipeline([("clf", estimator)])

    # Fit on training data
    pipeline.fit(X_tr, y_tr)

    # Predictions on hold‑out test set
    y_pred = pipeline.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    macro_f1 = f1_score(y_te, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_te, y_pred)
    cm = confusion_matrix(y_te, y_pred, labels=pipeline.classes_)

    # 5‑fold stratified cross‑validation (macro F1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        pipeline,
        X_tr,
        y_tr,
        cv=cv,
        scoring=make_scorer(f1_score, average="macro"),
        n_jobs=-1,
    )
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    return {
        "model": name,
        "test_accuracy": acc,
        "test_macro_f1": macro_f1,
        "test_balanced_accuracy": bal_acc,
        "cv_macro_f1_mean": cv_mean,
        "cv_macro_f1_std": cv_std,
        "confusion_matrix": cm,
        "pipeline": pipeline,
    }

# ---------------------------------------------------------------------------
# Run evaluation for all models
# ---------------------------------------------------------------------------
results = []
for name, est in models.items():
    res = evaluate_model(name, est, X_train, y_train, X_test, y_test)
    results.append(res)

# ---------------------------------------------------------------------------
# Build summary DataFrame for CSV output (excluding large objects)
# ---------------------------------------------------------------------------
summary_rows = []
for r in results:
    summary_rows.append(
        {
            "model": r["model"],
            "test_accuracy": r["test_accuracy"],
            "test_macro_f1": r["test_macro_f1"],
            "test_balanced_accuracy": r["test_balanced_accuracy"],
            "cv_macro_f1_mean": r["cv_macro_f1_mean"],
            "cv_macro_f1_std": r["cv_macro_f1_std"],
        }
    )

summary_df = pd.DataFrame(summary_rows)
RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(RESULT_CSV, index=False)

# ---------------------------------------------------------------------------
# Select best model based on cross‑validation macro F1 (primary criterion)
# ---------------------------------------------------------------------------
best = max(results, key=lambda x: x["cv_macro_f1_mean"])
print("=== Model Comparison Summary ===")
print(summary_df.to_string(index=False))
print("\nBest model (CV macro F1):", best["model"])
print("CV macro F1 mean: {:.4f} +/- {:.4f}".format(best["cv_macro_f1_mean"], best["cv_macro_f1_std"]))
print("Test Accuracy: {:.2%}".format(best["test_accuracy"]))
print("Test Macro F1: {:.4f}".format(best["test_macro_f1"]))
print("\nConfusion Matrix (test set):")
print(pd.DataFrame(best["confusion_matrix"], index=best["pipeline"].classes_, columns=best["pipeline"].classes_))

# ---------------------------------------------------------------------------
# Save the selected model pipeline for later integration
# ---------------------------------------------------------------------------
joblib.dump(best["pipeline"], SELECTED_MODEL_PATH)
print(f"\nSaved best model pipeline to: {SELECTED_MODEL_PATH}")
