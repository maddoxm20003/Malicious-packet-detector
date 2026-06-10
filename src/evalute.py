"""
File: src/evaluate.py
Description: Assessment engine for the project. Compares the model's multi-class predictions 
             against the ground-truth test labels. Computes multi-class evaluation scores 
             for Accuracy, Precision, Recall, and F1-score[cite: 9], applying appropriate macro/weighted 
             averaging strategies to handle classification across three distinct threat categories (C1, C2, C3)[cite: 8, 9].
"""