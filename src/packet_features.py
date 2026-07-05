"""
File: src/packet_features.py
Description: Shared packet-level feature helpers used by both predict_live.py (live/pcap
             inference) and extract_pcap_features.py (offline pcap -> labeled CSV rows), so
             the port-range bucketing and TCP flag classification logic exists in exactly
             one place instead of being duplicated across scripts.
"""

from scapy.all import IP, TCP, UDP, ICMP

def get_proto_name(packet):
    if packet.haslayer(UDP):
        return "UDP"
    elif packet.haslayer(TCP):
        return "TCP"
    elif packet.haslayer(ICMP):
        return "ICMP"
    else:
        return "OTHER"

def get_port_range(port):
    if 0 <= port <= 1023:
        return "Well-Known"
    elif 1024 <= port <= 49151:
        return "Registered"
    else:
        return "Dynamic"

def get_flow_key(packet, proto):
    """
    Identifies which connection/flow a packet belongs to. Canonicalized (endpoints
    sorted) so that both directions of the same conversation -- e.g. a client's
    request and the server's reply -- map to the same key instead of being treated
    as two separate flows. Without this, a single ordinary request/response pair
    gets split into two short, low-packet-count "flows" that look statistically
    identical to a scan, which is exactly what was causing real safe traffic to be
    misclassified. Used both for predict_live.py's live flow tracking and
    extract_pcap_features.py's offline flow aggregation.
    """
    sport = packet.sport if packet.haslayer(TCP) or packet.haslayer(UDP) else 0
    dport = packet.dport if packet.haslayer(TCP) or packet.haslayer(UDP) else 0
    endpoint_a = (packet[IP].src, sport)
    endpoint_b = (packet[IP].dst, dport)
    return (min(endpoint_a, endpoint_b), max(endpoint_a, endpoint_b), proto)

def classify_flags(flag_set):
    """
    Reduces a TCP flag string (e.g. scapy's packet[TCP].flags) down to a single
    category using the same SYN > ACK > FIN > NONE priority used throughout this project.
    """
    if 'S' in flag_set:
        return "SYN"
    elif 'A' in flag_set:
        return "ACK"
    elif 'F' in flag_set:
        return "FIN"
    else:
        return "NONE"
