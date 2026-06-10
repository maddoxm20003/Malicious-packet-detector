"""
File: src/preprocess.py
Description: Subsystem dedicated to data cleaning and feature engineering. Ingests raw network 
             session logs [cite: 6] and transforms them into numerical arrays. Tasks include 
             handling feature parsing, encoding categorical features (Protocol Type, Flags Present, 
             Source Port Range) [cite: 7], scaling numerical values (Packet Size)[cite: 7], 
             and isolating the target classification labels.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

def load_raw_data(file_path):
    """
    Ingests the initial dataset of 150 connection sessions.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"[PREPROCESS] Successfully loaded dataset from {file_path}. Total sessions: {len(df)}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] The file at {file_path} was not found. Please run generate_data.py first.")
        raise

def encode_and_isolate_features(df):
    """
    Separates the target labels from features, and converts categorical text 
    features (Protocol, Flags, Port Range) into binary vectors via One-Hot Encoding.
    """
    # 1. Isolate target vector (y) and drop tracking columns like session_id
    y = df['traffic_category']
    feature_base = df.drop(columns=['session_id', 'traffic_category'])
    feature_base = df.drop(columns=['session_id', 'traffic_category'], errors='ignore')
    
    # 2. Perform One-Hot Encoding on categorical columns
    # This automatically splits 'protocol_type', 'source_port_range', and 'flags_present'
    # into distinct columns of 0s and 1s, while keeping 'packet_size_bytes' intact.
    categorical_cols = ['protocol_type', 'source_port_range', 'flags_present']
    X = pd.get_dummies(feature_base, columns=categorical_cols, dtype=int)
    
    print(f"[PREPROCESS] Feature matrix shape after One-Hot Encoding: {X.shape}")
    return X, y

def split_data(X, y, test_size=0.20, random_state=42):
    """
    Partitions the matrix into separate datasets using an 80/20 train/test ratio.
    Setting a fixed random_state ensures the split is reproducible every time it runs.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Ensures equal class proportions in both train and test splits
    )
    print(f"[PREPROCESS] Data successfully split into training ({len(X_train)} rows) and testing ({len(X_test)} rows).")
    return X_train, X_test, y_train, y_test

def run_preprocessing_pipeline(raw_data_path):
    """
    Facilitates the entire sequential workflow of the preprocessing stage.
    Can be called directly by main.py.
    """
    # Step 1: Load data
    df = load_raw_data(raw_data_path)
    
    # Step 2: One-Hot Encode and isolate features
    X, y = encode_and_isolate_features(df)
    
    # Step 3: Train/Test Split
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    return X_train, X_test, y_train, y_test