"""
File: map_iot23_conn_log.py
Description: Maps a Zeek conn.log.labeled file from the IoT-23 dataset (Stratosphere Lab,
             CTU-IoT-Malware-Capture scenarios) into this project's feature schema,
             appending rows to the same real-world validation CSV that
             extract_pcap_features.py and map_kaggle_dataset.py build.

             Unlike CICIDS2017, this format has a genuine source port (id.orig_p), a real
             per-flow duration, and real packet counts -- so this mapping is more faithful
             and doesn't need the "destination port as a source-port proxy" compromise that
             had to be documented for map_kaggle_dataset.py.
"""

import os
import csv
import re
import argparse
from src.packet_features import get_port_range

CSV_COLUMNS = [
    "session_id", "packet_size_bytes", "protocol_type", "source_port_range",
    "flags_present", "duration_ms", "packet_count", "traffic_category"
]

# Zeek conn.log.labeled's #fields column order
CONN_LOG_COLUMNS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto",
    "service", "duration", "orig_bytes", "resp_bytes", "conn_state", "local_orig",
    "local_resp", "missed_bytes", "history", "orig_pkts", "orig_ip_bytes",
    "resp_pkts", "resp_ip_bytes", "tunnel_parents", "label", "detailed-label"
]

UNSET_VALUES = ("-", "(empty)", "")

def classify_label(label, detailed_label):
    label_clean = (label or "").strip().lower()
    detail_clean = (detailed_label or "").strip().lower()
    if "portscan" in detail_clean or "scan" in detail_clean:
        return "C2"
    elif label_clean == "benign":
        return "C1"
    else:
        return "C3"

def classify_history_flags(history):
    """
    Zeek's 'history' field encodes the sequence of flags/events per direction
    (uppercase = originator, lowercase = responder), e.g. 'ShADafF'. Reduced to the
    same SYN > ACK > FIN > NONE priority used everywhere else in this project,
    checked case-insensitively since either direction can carry the flag.
    """
    h = (history or "").lower()
    if "-" == history:
        return "NONE"
    if "s" in h:
        return "SYN"
    elif "a" in h:
        return "ACK"
    elif "f" in h:
        return "FIN"
    else:
        return "NONE"

def to_float(value, default=0.0):
    if value in UNSET_VALUES:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def to_int(value, default=0):
    return int(to_float(value, default))

def parse_conn_log(path):
    """
    Zeek conn.log.labeled is tab-separated with '#'-prefixed metadata lines. Splits on
    runs of whitespace rather than a strict tab, so this still works if the tabs got
    collapsed into spaces (e.g. through a copy/paste into a plain text file).
    """
    records = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = re.split(r"\s+", line.strip())
            if len(fields) != len(CONN_LOG_COLUMNS):
                continue
            records.append(dict(zip(CONN_LOG_COLUMNS, fields)))
    return records

def map_rows(records, start_session_id):
    rows = []
    session_id = start_session_id
    for r in records:
        proto = r.get("proto", "").upper()
        if proto not in ("TCP", "UDP", "ICMP"):
            proto = "OTHER"

        orig_pkts = to_int(r.get("orig_pkts"))
        resp_pkts = to_int(r.get("resp_pkts"))
        packet_count = max(orig_pkts + resp_pkts, 1)

        orig_ip_bytes = to_int(r.get("orig_ip_bytes"))
        resp_ip_bytes = to_int(r.get("resp_ip_bytes"))
        packet_size_bytes = round((orig_ip_bytes + resp_ip_bytes) / packet_count)

        # Zeek duration is in seconds; "-" means a single unanswered packet with no
        # computed duration (treated as an effectively instantaneous 0ms session).
        duration_ms = to_float(r.get("duration"), default=0.0) * 1000.0

        try:
            source_port_range = get_port_range(int(r.get("id.orig_p", 0)))
        except (TypeError, ValueError):
            source_port_range = "Dynamic"

        rows.append({
            "session_id": session_id,
            "packet_size_bytes": packet_size_bytes,
            "protocol_type": proto,
            "source_port_range": source_port_range,
            "flags_present": classify_history_flags(r.get("history")),
            "duration_ms": duration_ms,
            "packet_count": packet_count,
            "traffic_category": classify_label(r.get("label"), r.get("detailed-label")),
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
    file_exists = os.path.exists(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(
        description="Map an IoT-23 Zeek conn.log.labeled file into this project's feature schema."
    )
    parser.add_argument("--file", required=True, help="Path to the conn.log.labeled file")
    parser.add_argument("--output", default="data/raw/real_world_test.csv",
                         help="CSV file to append the mapped rows to")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"[ERROR] File not found: {args.file}")
        return

    print(f"[MAP] Parsing {args.file}...")
    records = parse_conn_log(args.file)
    print(f"[MAP] Parsed {len(records)} flow records.")

    rows = map_rows(records, next_session_id(args.output))
    append_rows(rows, args.output)

    counts = {}
    for row in rows:
        counts[row["traffic_category"]] = counts.get(row["traffic_category"], 0) + 1
    print(f"[MAP] Appended {len(rows)} rows to {args.output}. Class breakdown: {counts}")

if __name__ == "__main__":
    main()
