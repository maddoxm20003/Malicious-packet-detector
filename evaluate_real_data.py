"""
File: evaluate_real_data.py
Description: Scores the already-trained model against data/raw/real_world_test.csv -- the
             real-world validation set built by extract_pcap_features.py and
             map_kaggle_dataset.py -- reusing the same evaluation reporting main.py uses
             for the synthetic test split (accuracy, classification report, confusion
             matrix), just pointed at real data instead.
"""

import json
import joblib
from src.preprocess import load_raw_data, encode_and_isolate_features
from src.evaluate import evaluate_predictions

MODEL_PATH = "output/models/naive_bayes_model.pkl"
FEATURE_COLUMNS_PATH = "output/models/feature_columns.json"
REAL_DATA_PATH = "data/raw/real_world_test.csv"
REPORT_OUTPUT_PATH = "output/reports/real_world_report.txt"

def main():
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH) as f:
        feature_columns = json.load(f)

    df = load_raw_data(REAL_DATA_PATH)
    X_real, y_real = encode_and_isolate_features(df)

    # Reindex to the exact columns the model was trained on -- the real dataset may not
    # contain every category (e.g. no ICMP rows), so this fills any missing one-hot
    # columns with 0 instead of crashing or silently misaligning.
    X_real = X_real.reindex(columns=feature_columns, fill_value=0)

    print(f"[EVALUATE-REAL] Scoring model against {REAL_DATA_PATH} ({len(df)} real rows)...")
    evaluate_predictions(
        model=model,
        X_test=X_real,
        y_test=y_real,
        report_destination=REPORT_OUTPUT_PATH
    )

if __name__ == "__main__":
    main()
