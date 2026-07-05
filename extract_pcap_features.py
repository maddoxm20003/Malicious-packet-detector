"""
File: extract_pcap_features.py
Description: Converts a captured pcap file into one labeled row per network flow, using
             the same feature schema as data/raw/network_sessions.csv. This lets real
             captured traffic (labeled by you, e.g. "this whole file is a scan") be scored
             through the existing evaluate.py reporting instead of only being eyeballed
             via predict_live.py's per-packet output.
"""

import os
import csv
import argparse
from scapy.all import rdpcap, IP, TCP, UDP
from src.packet_features import get_proto_name, get_port_range, classify_flags, get_flow_key

VALID_LABELS = ["C1", "C2", "C3"]
CSV_COLUMNS = [
    "session_id", "packet_size_bytes", "protocol_type", "source_port_range",
    "flags_present", "duration_ms", "packet_count", "traffic_category"
]

def extract_flows(pcap_path):
    """
    Reads every packet in the pcap and groups them into flows keyed by
    (src_ip, dst_ip, sport, dport, proto) -- the same key predict_live.py uses for live
    tracking, but aggregated in one pass over the whole file instead of updated live.
    """
    packets = rdpcap(pcap_path)
    flows = {}

    for packet in packets:
        if not packet.haslayer(IP):
            continue

        proto = get_proto_name(packet)
        flow_key = get_flow_key(packet, proto)
        sport = packet.sport if packet.haslayer(TCP) or packet.haslayer(UDP) else 0

        if flow_key not in flows:
            flows[flow_key] = {
                "protocol_type": proto,
                "source_port_range": get_port_range(sport),
                "sizes": [],
                "flag_chars": "",
                "start_time": packet.time,
                "end_time": packet.time,
            }

        flow = flows[flow_key]
        flow["sizes"].append(len(packet))
        if packet.haslayer(TCP):
            flow["flag_chars"] += str(packet[TCP].flags)
        flow["end_time"] = packet.time

    return flows

def flows_to_rows(flows, label, start_session_id):
    rows = []
    session_id = start_session_id
    for flow in flows.values():
        rows.append({
            "session_id": session_id,
            "packet_size_bytes": round(sum(flow["sizes"]) / len(flow["sizes"])),
            "protocol_type": flow["protocol_type"],
            "source_port_range": flow["source_port_range"],
            "flags_present": classify_flags(flow["flag_chars"]),
            "duration_ms": (flow["end_time"] - flow["start_time"]) * 1000,
            "packet_count": len(flow["sizes"]),
            "traffic_category": label,
        })
        session_id += 1
    return rows

def next_session_id(output_path):
    if not os.path.exists(output_path):
        return 1
    with open(output_path, newline="") as f:
        ids = [int(row["session_id"]) for row in csv.DictReader(f)]
    return max(ids, default=0) + 1

def append_rows(rows, output_path):
    """
    Appends rows to output_path, writing a header only if the file doesn't exist yet,
    so running this once per captured pcap builds up a single combined real-world
    validation CSV rather than overwriting it each time.
    """
    file_exists = os.path.exists(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(
        description="Extract one labeled row per network flow from a pcap file."
    )
    parser.add_argument("--file", required=True, help="Path to the pcap file to extract from")
    parser.add_argument("--label", required=True, choices=VALID_LABELS,
                         help="Traffic category to apply to every flow extracted from this file")
    parser.add_argument("--output", default="data/raw/real_world_test.csv",
                         help="CSV file to append the extracted rows to")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"[ERROR] PCAP file not found: {args.file}")
        return

    print(f"[EXTRACT] Reading packets from {args.file}...")
    flows = extract_flows(args.file)
    print(f"[EXTRACT] Found {len(flows)} flows.")

    rows = flows_to_rows(flows, args.label, next_session_id(args.output))
    append_rows(rows, args.output)
    print(f"[EXTRACT] Appended {len(rows)} labeled '{args.label}' rows to {args.output}")

if __name__ == "__main__":
    main()
