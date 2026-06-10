pu"""
File: src/evaluate.py
Description: Assessment engine for the project. Compares the model's multi-class predictions 
             against the ground-truth test labels. Computes multi-class evaluation scores 
             for Accuracy, Precision, Recall, and F1-score[cite: 9], applying appropriate macro/weighted 
             averaging strategies to handle classification across three distinct threat categories (C1, C2, C3)[cite: 8, 9].
"""

import os
from sklearn.metrics import classification_report, accuracy_score

def evaluate_predictions(model, X_test, y_test, report_destination="output/reports/classification_report.txt"):
    """
    Passes the unseen testing features to the trained model, generates traffic 
    class predictions, and calculates performance metrics. Writes a formal report to disk.
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
        target_names=['C1: Safe', 'C2: Suspicious', 'C3: Malicious']
    )
    
    print("\n--- Model Performance Report ---")
    print(metrics_report)
    print("--------------------------------")
    
    # 4. Save the performance metrics report to disk
    os.makedirs(os.path.dirname(report_destination), exist_ok=True)
    with open(report_destination, "w") as f:
        f.write("==================================================\n")
        f.write(" NAIVE BAYES NETWORK TRAFFIC CLASSIFIER PERFORMANCE\n")
        f.write("==================================================\n\n")
        f.write(f"Overall Global Accuracy: {accuracy:.4f}\n\n")
        +f.write("Detailed Class Metrics:\n")
        f.write(metrics_report)
        
    print(f"[EVALUATE] Performance report saved to: {report_destination}")
    return accuracy, metrics_report