
"""
File: generate_data.py
Description: Automates the generation of a 150-session mock network traffic dataset
             for the Naive Bayes Malicious Packet Detection project. Generates
             realistic statistical profiles for Safe (C1), Scanning (C2), and
             Malicious (C3) traffic classes. Includes flow-level fields (duration_ms,
             packet_count) alongside packet_size_bytes to give the classifier more
             separable signal between the Scanning and Malicious classes.
Author: Maddox Marin [cite: 1]
Date: June 2026 [cite: 1]
"""

import os
import random
import pandas as pd

def generate_mock_dataset(filename="data/raw/network_sessions.csv", total_samples=150):
    # Ensure the directory structure exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Target distribution: 70 Safe, 40 Scanning, 40 Malicious [cite: 8]
    distributions = {
        'C1': 70,  # Safe/Normal Traffic [cite: 8]
        'C2': 40,  # Suspicious/Scanning Traffic [cite: 8]
        'C3': 40   # Highly Likely Malicious Traffic [cite: 8]
    }
    
    data = []
    session_id = 1
    
    for category, count in distributions.items():
        for _ in range(count):
            if category == 'C1':
                # Safe Traffic Profile: Varied, larger packet sizes, standard protocols
                packet_size = random.randint(300, 1500) # [cite: 7]
                protocol = random.choices(['TCP', 'UDP'], weights=[0.75, 0.25])[0] # [cite: 7]
                port_range = random.choices(['Well-Known', 'Registered', 'Dynamic'], weights=[0.60, 0.30, 0.10])[0] # [cite: 7]
                flags = random.choices(['ACK', 'NONE', 'SYN'], weights=[0.70, 0.25, 0.05])[0] # [cite: 7]
                # Safe traffic has two sub-profiles, not one: most sessions are longer/
                # chattier real data exchanges, but a real share are legitimately quick
                # (fast TLS handshakes, small API calls, health checks). Discovered by
                # comparing against real captured traffic -- several genuinely safe short
                # connections (~40ms, ~6 packets) were being misclassified as scans when
                # C1 only had the single long-tailed profile below.
                if random.random() < 0.70:
                    duration_ms = max(1, round(random.gauss(900, 500)))
                    packet_count = max(1, round(random.gauss(70, 40)))
                else:
                    duration_ms = max(1, round(random.gauss(40, 20)))
                    packet_count = max(1, round(random.gauss(6, 3)))

            elif category == 'C2':
                # Scanning Traffic Profile: Small uniform packets, heavy ICMP/TCP, dynamic ports
                packet_size = random.choice([64, 74, 84]) # [cite: 7]
                protocol = random.choices(['TCP', 'ICMP', 'UDP'], weights=[0.50, 0.40, 0.10])[0] # [cite: 7]
                port_range = random.choices(['Dynamic', 'Registered', 'Well-Known'], weights=[0.70, 0.20, 0.10])[0] # [cite: 7]
                flags = random.choices(['SYN', 'NONE', 'FIN'], weights=[0.60, 0.30, 0.10])[0] # [cite: 7]
                # Single probe: no completed handshake, tool moves on almost instantly
                duration_ms = max(1, round(random.gauss(30, 25)))
                packet_count = max(1, round(random.gauss(4, 3)))

            elif category == 'C3':
                # Malicious Traffic Profile: Polarized sizes, heavy TCP anomalies, high SYN flags
                packet_size = random.choice([40, 1500, 2048]) # [cite: 7]
                protocol = random.choices(['TCP', 'UDP'], weights=[0.85, 0.15])[0] # [cite: 7]
                port_range = random.choices(['Dynamic', 'Registered'], weights=[0.90, 0.10])[0] # [cite: 7]
                flags = random.choices(['SYN', 'ACK', 'FIN'], weights=[0.80, 0.15, 0.05])[0] # [cite: 7]
                # Flood/brute-force burst: high packet volume crammed into a short window
                duration_ms = max(1, round(random.gauss(180, 150)))
                packet_count = max(1, round(random.gauss(280, 180)))

            data.append({
                'session_id': session_id,
                'packet_size_bytes': packet_size,
                'protocol_type': protocol,
                'source_port_range': port_range,
                'flags_present': flags,
                'duration_ms': duration_ms,
                'packet_count': packet_count,
                'traffic_category': category
            })
            session_id += 1
            
    # Shuffle the dataset so rows aren't neatly grouped by category
    random.shuffle(data)
    
    # Re-assign sequential session IDs after shuffling
    for idx, row in enumerate(data):
        row['session_id'] = idx + 1
        
    # Write to CSV
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Successfully generated {total_samples} network logs at: {filename}")

if __name__ == "__main__":
    generate_mock_dataset()