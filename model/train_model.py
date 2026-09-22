import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import joblib


# ============================================================
# Load dataset
# ============================================================

DATA_FILE = "model/training_data_diverse.csv"

df = pd.read_csv(DATA_FILE)


# ============================================================
# Features
# ============================================================

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
    "arrival_burst_corr"
]


X = df[FEATURES]

y = df["best_scheduler"]


# ============================================================
# Display dataset information
# ============================================================

print()
print("=" * 60)
print("RANDOM FOREST SCHEDULER CLASSIFIER")
print("=" * 60)

print()

print("Features used:")
for feature in FEATURES:
    print("-", feature)

print()

print("Classes:")
print(y.value_counts())


# ============================================================
# Train / test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print()
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# Handle class imbalance
# ============================================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(
    zip(classes, weights)
)

print()
print("Class weights:")
print(class_weights)


# ============================================================
# Create Random Forest
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=2,
    class_weight=class_weights,
    random_state=42
)


# ============================================================
# Train
# ============================================================

print()
print("Training model...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# Predictions
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# Evaluation
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print()
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


print()
print("Confusion Matrix:")

labels = model.classes_

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print()

print("Labels:", labels)

print(cm)


# ============================================================
# Feature Importance
# ============================================================

print()
print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# Save model
# ============================================================

MODEL_FILE = (
    "model/random_forest_scheduler.pkl"
)

joblib.dump(
    model,
    MODEL_FILE
)


print()
print("=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(
    f"Saved to: {MODEL_FILE}"
)