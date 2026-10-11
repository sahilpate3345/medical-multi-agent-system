"""
Medical High-Recall Training Pipeline
Optimizes model to minimize False Negatives and maximize Sensitivity (Recall).
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    precision_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(os.path.dirname(__file__), "heart.csv")
MODEL_OUT_LOCAL = os.path.join(os.path.dirname(__file__), "modelDP.pkl")
MODEL_OUT_ROOT = os.path.join(BASE_DIR, "modelDP.pkl")

def train_high_recall_model():
    print("=" * 70)
    print("TRAINING CLINICAL HIGH-RECALL MODEL (TARGET: MINIMIZE FALSE NEGATIVES)")
    print("=" * 70)

    # 1. Load dataset
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset Loaded: {df.shape[0]} patient records, {df.shape[1] - 1} clinical features.")

    X = df.drop("target", axis=1)
    y = df["target"]

    # Stratified split to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 2. Build Pipeline with Class Weighting (Heavily penalizes False Negatives)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            class_weight={0: 1.0, 1: 2.5},
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    # 3. Predict Probabilities & Evaluate
    probs_test = pipeline.predict_proba(X_test)[:, 1]

    # Clinical Decision Threshold:
    # A standard 0.50 threshold is too lenient in medicine.
    # We use a clinical sensitivity threshold (0.35) to eliminate False Negatives.
    CLINICAL_THRESHOLD = 0.35
    preds_test = (probs_test >= CLINICAL_THRESHOLD).astype(int)

    # 4. Compute Comprehensive Clinical Metrics
    cm = confusion_matrix(y_test, preds_test)
    tn, fp, fn, tp = cm.ravel()

    recall = recall_score(y_test, preds_test)
    specificity = tn / (tn + fp)
    precision = precision_score(y_test, preds_test)
    f1 = f1_score(y_test, preds_test)
    accuracy = accuracy_score(y_test, preds_test)
    auc = roc_auc_score(y_test, probs_test)
    fnr = fn / (tp + fn)  # False Negative Rate

    print("\nCLINICAL EVALUATION METRICS:")
    print("-" * 50)
    print(f"* Total Test Patients evaluated : {len(y_test)}")
    print(f"* True Positives (Sick identified)  : {tp}")
    print(f"* True Negatives (Healthy id'd)   : {tn}")
    print(f"* False Positives (Safe overcall) : {fp}")
    print(f"* False Negatives (DANGEROUS MISS): {fn}  <-- MINIMIZED TO {fn}!")
    print("-" * 50)
    print(f"[RECALL / Sensitivity]            : {recall * 100:.2f}%  (Target: > 95%)")
    print(f"[False Negative Rate (FNR)]       : {fnr * 100:.2f}%")
    print(f"[Specificity]                     : {specificity * 100:.2f}%")
    print(f"[Precision]                       : {precision * 100:.2f}%")
    print(f"[F1-Score]                        : {f1:.4f}")
    print(f"[ROC-AUC Score]                   : {auc:.4f}")
    print(f"[Overall Accuracy]                : {accuracy * 100:.2f}%")
    print("-" * 50)

    # Save to both locations
    joblib.dump(pipeline, MODEL_OUT_LOCAL)
    joblib.dump(pipeline, MODEL_OUT_ROOT)
    print(f"Saved High-Recall Model to:\n  - {MODEL_OUT_LOCAL}\n  - {MODEL_OUT_ROOT}")

    return {
        "recall": recall,
        "false_negatives": fn,
        "accuracy": accuracy,
        "auc": auc
    }

if __name__ == "__main__":
    train_high_recall_model()