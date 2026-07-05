"""
File: map_kaggle_dataset.py
Description: Maps a downloaded CICIDS2017-style Kaggle CSV (CICFlowMeter flow records)
             into this project's feature schema, appending the mapped rows to the same
             real-world validation CSV that extract_pcap_features.py builds from your own
             pcap captures.

             CICIDS2017 columns often have inconsistent leading/trailing whitespace
             depending on which Kaggle upload you use -- this script strips column names
             before matching, but validates every expected source column is present and
             fails with the actual column list if something doesn't match, rather than
             silently mismapping.

             Known caveat: CICIDS2017 only provides a Destination Port column, not a
             source port. This script buckets Destination Port through the same
             get_port_range() ranges and stores it as source_port_range for schema
             compatibility -- it is NOT the same measurement as the source-port bucket
             your own pcap captures produce. Worth a caveat sentence in your report.
"""

import os
import csv
import argparse
import pandas as pd
from src.packet_features import get_port_range

CSV_COLUMNS = [
    "session_id", "packet_size_bytes", "protocol_type", "source_port_range",
    "flags_present", "duration_ms", "packet_count", "traffic_category"
]

# CICIDS2017's standard CICFlowMeter column names (after stripping whitespace)
SOURCE_COLUMNS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Average Packet Size", "SYN Flag Count", "ACK Flag Count", "FIN Flag Count",
    "Protocol", "Label",
]

PROTOCOL_NUMBER_MAP = {6: "TCP", 17: "UDP", 1: "ICMP"}

def classify_label(label):
    label_clean = label.strip().upper()
    if label_clean == "BENIGN":
        return "C1"
    elif "PORTSCAN" in label_clean or "PORT SCAN" in label_clean:
        return "C2"
    else:
        return "C3"

def classify_kaggle_flags(row):
    if row["SYN Flag Count"] > 0:
        return "SYN"
    elif row["ACK Flag Count"] > 0:
        return "ACK"
    elif row["FIN Flag Count"] > 0:
        return "FIN"
    else:
        return "NONE"

def load_and_validate(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    missing = [col for col in SOURCE_COLUMNS if col not in df.columns]
    if missing:
        print(f"[ERROR] Expected columns not found in {csv_path}: {missing}")
        print(f"[ERROR] Actual columns in file: {list(df.columns)}")
        print("[ERROR] Update SOURCE_COLUMNS / the mapping logic in map_kaggle_dataset.py to match.")
        raise SystemExit(1)

    return df

def map_rows(df, start_session_id):
    rows = []
    session_id = start_session_id
    for _, row in df.iterrows():
        protocol = PROTOCOL_NUMBER_MAP.get(int(row["Protocol"]), "OTHER")
        rows.append({
            "session_id": session_id,
            "packet_size_bytes": round(row["Average Packet Size"]),
            "protocol_type": protocol,
            "source_port_range": get_port_range(int(row["Destination Port"])),
            "flags_present": classify_kaggle_flags(row),
            "duration_ms": row["Flow Duration"] / 1000.0,  # CICIDS2017 duration is in microseconds
            "packet_count": int(row["Total Fwd Packets"]) + int(row["Total Backward Packets"]),
            "traffic_category": classify_label(str(row["Label"])),
        })
        session_id += 1
    return rows

def next_session_id(output_path):
    if not os.path.exists(output_path):
        return 1
    with open(output_path, newline="") as f:
        ids = [int(r["session_id"]) for r in csv.DictReader(f)]
    return max(ids, default=0) + 1

def append_rows(rows, output_path):
    file_exists = os.path.exists(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(
        description="Map a CICIDS2017-style Kaggle CSV into this project's feature schema."
    )
    parser.add_argument("--file", required=True, help="Path to the downloaded Kaggle CSV")
    parser.add_argument("--output", default="data/raw/real_world_test.csv",
                         help="CSV file to append the mapped rows to")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"[ERROR] File not found: {args.file}")
        return

    print(f"[MAP] Loading {args.file}...")
    df = load_and_validate(args.file)
    print(f"[MAP] Loaded {len(df)} rows. Mapping to project schema...")

    rows = map_rows(df, next_session_id(args.output))
    append_rows(rows, args.output)

    label_counts = {}
    for r in rows:
        label_counts[r["traffic_category"]] = label_counts.get(r["traffic_category"], 0) + 1
    print(f"[MAP] Appended {len(rows)} rows to {args.output}. Class breakdown: {label_counts}")
    print("[MAP] NOTE: source_port_range for this data is bucketed from Destination Port, "
          "not source port -- see this file's module docstring for why.")

if __name__ == "__main__":
    main()
