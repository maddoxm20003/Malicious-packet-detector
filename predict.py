"""
File: predict.py
Description: Inference script to test the ML bot with a single network packet.
"""

import joblib
import pandas as pd
import numpy as np

def predict_packet(packet_dict):
    """
    Uses the trained pipeline to classify a single packet dictionary.
    """
    try:
        # 1. Load the saved Pipeline (Preprocessor + Model)
        model_path = "output/models/naive_bayes_model.pkl"
        pipeline = joblib.load(model_path)
        
        # 2. Convert dictionary to DataFrame (matches training format)
        df = pd.DataFrame([packet_dict])
        
        # 3. Perform inference
        # The pipeline automatically applies Scaling and One-Hot Encoding
        prediction = pipeline.predict(df)
        probabilities = pipeline.predict_proba(df)
        
        confidence = np.max(probabilities) * 100
        
        print(f"\n[BOT] Packet Data: {packet_dict}")
        print(f"[RESULT] Traffic Category: {prediction[0]}")
        print(f"[CONFIDENCE] {confidence:.2f}%")
        
        return prediction[0]
    
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        return None

if __name__ == "__main__":
    # Test with a potentially Malicious packet (ICMP + Large Size)
    sample_packet = {
        'protocol_type': 'ICMP',
        'packet_size_bytes': 1450,
        'source_port_range': 'System',
        'flags_present': 'NONE'
    }
    predict_packet(sample_packet)