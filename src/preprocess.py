"""
File: src/preprocess.py
Description: Subsystem dedicated to data cleaning and feature engineering. Ingests raw network 
             session logs [cite: 6] and transforms them into numerical arrays. Tasks include 
             handling feature parsing, encoding categorical features (Protocol Type, Flags Present, 
             Source Port Range) [cite: 7], scaling numerical values (Packet Size)[cite: 7], 
             and isolating the target classification labels.
"""