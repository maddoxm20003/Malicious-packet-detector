"""
File: src/evaluate.py
Description: Assessment engine for the project. Compares the model's multi-class predictions
             against the ground-truth test labels. Computes multi-class evaluation scores
             for Accuracy, Precision, Recall, and F1-score[cite: 9], applying appropriate macro/weighted
             averaging strategies to handle classification across three distinct threat categories (C1, C2, C3)[cite: 8, 9].
             Also provides stratified k-fold cross-validation (more robust than a single
             train/test split on a small dataset) and a confusion matrix, printed as text
             and saved as a heatmap image.
"""

import os
import matplotlib
matplotlib.use("Agg")  # headless-safe backend; this runs as a script, not a notebook
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB

CLASS_LABELS = ['C1', 'C2', 'C3']
CLASS_NAMES = ['C1: Safe', 'C2: Suspicious', 'C3: Malicious']

def cross_validate_model(X, y, cv=5):
    """
    Runs stratified k-fold cross-validation over the full dataset using a fresh
    Gaussian Naive Bayes classifier per fold (cross_val_score clones the estimator
    internally). Gives a more reliable accuracy estimate than a single 80/20 split,
    which matters with only 150 rows where one split can be lucky or unlucky.
    """
    print(f"[EVALUATE] Running {cv}-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(GaussianNB(), X, y, cv=skf)

    fold_str = ", ".join(f"{s:.4f}" for s in scores)
    print(f"[EVALUATE] Fold accuracies: [{fold_str}]")
    print(f"[EVALUATE] Cross-Validation Mean Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    return scores

def _render_confusion_matrix(cm, image_destination):
    """
    Prints the confusion matrix as a text grid and saves a heatmap PNG.
    Returns the text grid so it can also be written into the saved report.
    """
    header = "".join(f"{name:>16}" for name in CLASS_NAMES)
    lines = ["Confusion Matrix (rows=actual, cols=predicted):", " " * 16 + header]
    for i, name in enumerate(CLASS_NAMES):
        row = "".join(f"{cm[i][j]:>16}" for j in range(len(CLASS_NAMES)))
        lines.append(f"{name:<16}{row}")
    cm_text = "\n".join(lines)
    print("\n" + cm_text)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Naive Bayes Confusion Matrix")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, cm[i][j], ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(image_destination)
    plt.close(fig)
    print(f"[EVALUATE] Confusion matrix heatmap saved to: {image_destination}")

    return cm_text

def evaluate_predictions(model, X_test, y_test, report_destination="output/reports/classification_report.txt", cv_scores=None):
    """
    Passes the unseen testing features to the trained model, generates traffic
    class predictions, and calculates performance metrics. Writes a formal report to disk,
    including cross-validation results (if provided) and a confusion matrix.
    """
    print("[EVALUATE] Running predictions on test data split...")
    # 1. Generate predictions using the 20% testing subset
    y_pred = model.predict(X_test)

    # 2. Compute the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[EVALUATE] Global Model Accuracy: {accuracy:.4f}")

    # 3. Generate a comprehensive multi-class classification report
    # target_names maps directly to your C1, C2, and C3 traffic categories
    metrics_report = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES
    )

    print("\n--- Model Performance Report ---")
    print(metrics_report)
    print("--------------------------------")

    # 4. Confusion matrix: text grid + saved heatmap PNG
    os.makedirs(os.path.dirname(report_destination), exist_ok=True)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)
    image_destination = os.path.join(os.path.dirname(report_destination), "confusion_matrix.png")
    cm_text = _render_confusion_matrix(cm, image_destination)

    # 5. Save the performance metrics report to disk
    with open(report_destination, "w") as f:
        f.write("==================================================\n")
        f.write(" NAIVE BAYES NETWORK TRAFFIC CLASSIFIER PERFORMANCE\n")
        f.write("==================================================\n\n")
        f.write(f"Overall Global Accuracy (single 80/20 split): {accuracy:.4f}\n\n")
        if cv_scores is not None:
            fold_str = ", ".join(f"{s:.4f}" for s in cv_scores)
            f.write(f"{len(cv_scores)}-Fold Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")
            f.write(f"Fold scores: [{fold_str}]\n\n")
        f.write("Detailed Class Metrics:\n")
        f.write(metrics_report)
        f.write("\n\n" + cm_text + "\n")

    print(f"[EVALUATE] Performance report saved to: {report_destination}")
    return accuracy, metrics_report