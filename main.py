"""
Project: Malicious Network Packet Detection using Naive Bayes
File: main.py
Description: The central orchestrator and entry point for the entire machine learning pipeline. 
             This script imports modules from the src/ directory to sequentially load raw 
             network traffic logs, execute preprocessing, split data into an 80/20 train/test 
             split, train the Naive Bayes classifier, and output evaluation metrics.
Author: Maddox Marin
Date: June 2026
"""



from src.preprocess import run_preprocessing_pipeline
from src.model import train_naive_bayes, save_model
from src.evaluate import evaluate_predictions

def main():
    print("=========================================================")
    # 1. Define the path to your raw dataset of 150 network sessions
    RAW_DATA_PATH = "data/raw/network_sessions.csv"
    MODEL_OUTPUT_PATH = "output/models/naive_bayes_model.pkl"
    REPORT_OUTPUT_PATH = "output/reports/classification_report.txt"
    
    print("[PIPELINE] Starting Malicious Packet Detection Pipeline...")
    print("=========================================================")
    
    # 2. Execute Preprocessing Stage
    # This loads the data, runs One-Hot Encoding, and splits it into 80% train / 20% test
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline(RAW_DATA_PATH)
    print("---------------------------------------------------------")
    
    # 3. Execute Model Training Stage
    # This fits the Gaussian Naive Bayes classifier to the training data split
    trained_clf = train_naive_bayes(X_train, y_train)
    
    # Save the trained model artifact to disk for future portability
    save_model(trained_clf, MODEL_OUTPUT_PATH)
    print("---------------------------------------------------------")
    
    # 4. Execute Evaluation Stage
    # This runs predictions on the 20% test split and outputs classification reports
    accuracy, report = evaluate_predictions(
        model=trained_clf, 
        X_test=X_test, 
        y_test=y_test, 
        report_destination=REPORT_OUTPUT_PATH
    )
    
    print("=========================================================")
    print("[PIPELINE] Execution Complete! All steps processed successfully.")
    print("=========================================================")

if __name__ == "__main__":
    main()