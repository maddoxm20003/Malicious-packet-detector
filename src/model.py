"""
File: src/model.py
Description: Handles the machine learning model lifecycle. Contains functions to perform the 
             80/20 train/test split [cite: 9], initialize the Scikit-Learn Naive Bayes classifier[cite: 9], 
             fit the model to the training subset, and execute inference on the testing feature matrix.
"""

import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB

def train_naive_bayes(X_train, y_train):
    """
    Creates a Pipeline that includes preprocessing (Scaling and One-Hot Encoding)
    and the Gaussian Naive Bayes classifier.
    """
    print("[MODEL] Building Pipeline (Preprocessing + Classifier)...")
    
    # Define feature groups
    categorical_features = ['protocol_type', 'source_port_range', 'flags_present']
    numeric_features = ['packet_size_bytes']

    # Create a preprocessor that handles both types of data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Build the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GaussianNB())
    ])
    
    print("[MODEL] Fitting pipeline to training data...")
    pipeline.fit(X_train, y_train)
    print("[MODEL] Model training complete.")
    
    return pipeline

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
    