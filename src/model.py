"""
File: src/model.py
Description: Handles the machine learning model lifecycle. Contains functions to perform the 
             80/20 train/test split [cite: 9], initialize the Scikit-Learn Naive Bayes classifier[cite: 9], 
             fit the model to the training subset, and execute inference on the testing feature matrix.
"""

import os
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
    