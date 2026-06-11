"""
File: generate_data.py
Description: Generates a synthetic dataset of network connection sessions 
             for testing the Malicious Packet Detector.
"""

import pandas as pd
import numpy as np
import os

def generate_synthetic_data(output_path="data/raw/network_sessions.csv"):
    """
    Creates a CSV file with 150 samples containing session features and categories.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # For reproducibility
    np.random.seed(42)
    n_samples = 150
    
    # Generate base features
    protocols = np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples)
    packet_sizes = np.random.randint(40, 1500, n_samples)
    port_ranges = np.random.choice(['System', 'User', 'Dynamic'], n_samples)
    flags = np.random.choice(['SYN', 'ACK', 'FIN', 'NONE'], n_samples)
    
    # Create logic-based labels so the ML bot can actually "learn"
    categories = []
    for i in range(n_samples):
        if protocols[i] == 'ICMP' and packet_sizes[i] > 1000:
            categories.append('Malicious')  # Large ICMP packets (Ping of Death style)
        elif protocols[i] == 'UDP' and port_ranges[i] == 'Dynamic':
            categories.append('Suspicious') # Random UDP traffic
        else:
            categories.append('Safe')

    data = {
        'session_id': range(1000, 1000 + n_samples),
        'protocol_type': protocols,
        'packet_size_bytes': packet_sizes,
        'source_port_range': port_ranges,
        'flags_present': flags,
        'traffic_category': categories
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"[DATA] Successfully generated synthetic dataset at: {output_path}")

if __name__ == "__main__":
    generate_synthetic_data()