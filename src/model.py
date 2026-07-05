"""
File: src/model.py
Description: Handles the machine learning model lifecycle. Contains functions to perform the 
             80/20 train/test split [cite: 9], initialize the Scikit-Learn Naive Bayes classifier[cite: 9], 
             fit the model to the training subset, and execute inference on the testing feature matrix.
"""

import os
import json
import joblib
from sklearn.naive_bayes import GaussianNB

def train_naive_bayes(X_train, y_train):
    """
    Instantiates the Gaussian Naive Bayes algorithm and fits it to the 
    preprocessed training dataset.
    """
    print("[MODEL] Initializing Gaussian Naive Bayes classifier...")
    # Initialize the model
    model = GaussianNB()
    
    # Train the model on the 80% training data split
    print("[MODEL] Fitting the model to training data...")
    model.fit(X_train, y_train)
    print("[MODEL] Model training complete.")
    
    return model

def save_model(model, destination_path="output/models/naive_bayes_model.pkl"):
    """
    Serializes (pickles) the trained model object using joblib and saves it to disk.
    This creates a portable file that can be reloaded instantly for inference.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    
    # Save the model
    joblib.dump(model, destination_path)
    print(f"[MODEL] Trained model artifact successfully saved to: {destination_path}")

def save_feature_columns(columns, destination_path="output/models/feature_columns.json"):
    """
    Persists the trained feature matrix's column order/names as the single source of
    truth for what the model expects. predict_live.py and evaluate_real_data.py both
    load this instead of hardcoding their own copy of the column list, which would
    otherwise silently drift out of sync whenever the training schema changes.
    """
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    with open(destination_path, "w") as f:
        json.dump(list(columns), f, indent=2)
    print(f"[MODEL] Feature columns saved to: {destination_path}")
