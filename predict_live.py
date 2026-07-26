"""
File: predict_live.py
Description: Bridges the gap between raw network traffic and the ML model.
             Sniffs live traffic using Scapy, transforms packets into features,
             and runs inference using the trained Naive Bayes model.
"""

import pandas as pd
import joblib
from scapy.all import sniff, IP, TCP, UDP, ICMP
import os
import argparse

# Load the pre-trained model
MODEL_PATH = "output/models/naive_bayes_model.pkl"
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model not found at {MODEL_PATH}. Please run main.py first.")
    exit()

model = joblib.load(MODEL_PATH)

# The exact columns the model expects (must match training output)
FEATURE_COLUMNS = [
    'packet_size_bytes', 'duration_ms', 'packet_count',
    'protocol_type_ICMP', 'protocol_type_TCP',
    'protocol_type_UDP', 'source_port_range_Dynamic',
    'source_port_range_Registered', 'source_port_range_Well-Known',
    'flags_present_ACK', 'flags_present_FIN', 'flags_present_NONE',
    'flags_present_SYN'
]

# Tracks in-progress flows so duration_ms/packet_count can be computed live:
# {(src_ip, dst_ip, sport, dport, proto): {"start_time": float, "packet_count": int}}
# NOTE: entries are never evicted, so this grows unbounded on a long-running live
# capture. Fine for a demo/short pcap; a real deployment would need a flow timeout.
active_flows = {}

def get_flow_key(packet, proto):
    sport = packet.sport if packet.haslayer(TCP) or packet.haslayer(UDP) else 0
    dport = packet.dport if packet.haslayer(TCP) or packet.haslayer(UDP) else 0
    return (packet[IP].src, packet[IP].dst, sport, dport, proto)

def get_port_range(port):
    if 0 <= port <= 1023:
        return "Well-Known"
    elif 1024 <= port <= 49151:
        return "Registered"
    else:
        return "Dynamic"

def process_packet(packet):
    if not packet.haslayer(IP):
        return

    # 1. Extract Raw Features
    src_ip = packet[IP].src
    size = len(packet)
    proto = "UDP" if packet.haslayer(UDP) else "TCP" if packet.haslayer(TCP) else "ICMP" if packet.haslayer(ICMP) else "OTHER"

    port = 0
    if packet.haslayer(TCP) or packet.haslayer(UDP):
        port = packet.sport
    port_range = get_port_range(port)

    flags = "NONE"
    if packet.haslayer(TCP):
        f = packet[TCP].flags
        if 'S' in f: flags = "SYN"
        elif 'A' in f: flags = "ACK"
        elif 'F' in f: flags = "FIN"

    # 2. Track flow-level state (duration/packet count) across packets in this connection.
    # Uses packet.time (the capture timestamp) rather than wall-clock time, so offline
    # pcap replay -- which scapy plays back almost instantly -- still reflects the real
    # duration recorded at capture time instead of collapsing every flow to ~0ms.
    flow_key = get_flow_key(packet, proto)
    flow = active_flows.setdefault(flow_key, {"start_time": packet.time, "packet_count": 0})
    flow["packet_count"] += 1
    duration_ms = (packet.time - flow["start_time"]) * 1000
    packet_count = flow["packet_count"]

    # 3. Construct a DataFrame for One-Hot Encoding
    data = {
        'packet_size_bytes': [size],
        'duration_ms': [duration_ms],
        'packet_count': [packet_count],
        f'protocol_type_{proto}': [1],
        f'source_port_range_{port_range}': [1],
        f'flags_present_{flags}': [1]
    }

    df = pd.DataFrame(data)

    # Reindex to match the columns expected by the model, filling missing with 0
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    # 4. Predict
    prediction = model.predict(df)[0]

    # Mapping labels back to human readable
    labels = {'C1': 'SAFE', 'C2': 'SUSPICIOUS (Scanning)', 'C3': 'MALICIOUS'}
    print(f"[LIVE] {src_ip.ljust(15)} | Size: {str(size).rjust(4)}b | Proto: {proto:4} | "
          f"Dur: {duration_ms:6.1f}ms | Pkts: {packet_count:3} | Result: {labels.get(prediction, prediction)}")

def main():
    parser = argparse.ArgumentParser(description="ML-based Malicious Packet Detector")
    parser.add_argument("--file", help="Path to a PCAP file for offline analysis", default=None)
    args = parser.parse_args()

    if args.file:
        if not os.path.exists(args.file):
            print(f"[ERROR] PCAP file not found: {args.file}")
            return
        print(f"[STREAMS] Analyzing offline PCAP file: {args.file}")
        sniff(offline=args.file, prn=process_packet, store=0)
    else:
        print("=========================================================")
        print("[STREAMS] Starting live Malicious Packet Detection...")
        print("[STREAMS] Listening for IP traffic (Press Ctrl+C to stop)")
        print("=========================================================")

        try:
            # Sniff IP packets and pass them to our processing function
            sniff(filter="ip", prn=process_packet, store=0)
        except KeyboardInterrupt:
            print("\n[STREAMS] Stopping capture...")
        except PermissionError:
            print("[ERROR] You must run this script with Administrator/Root privileges to sniff traffic.")

if __name__ == "__main__":
    main()
