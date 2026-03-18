#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hamming Distance (HD) Calculation Demo
--------------------------------------
This script computes intra- and inter-device Hamming distances
from demo PUF response sequences. Results are saved to CSV and
summary statistics are printed to stdout.
"""

import numpy as np
import pandas as pd

# ----------------------------- Load Data ---------------------------------------

# Demo placeholder input file.
file_name = "Demo_Stat_Input.csv"
data = pd.read_csv(file_name)

# Use the 5th column (index 4), take 1,000,000 points, reshape to 1000 samples × 1000 points
raw_sequences = data.iloc[1:1000001, 4].values.reshape(1000, 1000)

# ----------------------------- Parameters --------------------------------------

n_segments = 10                # split each sequence into 10 segments
segment_length = 1000 // n_segments

# ----------------------------- Binarization ------------------------------------

binary_data = []
for sample in raw_sequences:
    # Scale to 8-bit integers (0–255)
    discretized = ((sample + 1) * 127.5).astype(np.uint8)
    # Convert to binary representation
    binary_representation = np.unpackbits(discretized[:, np.newaxis], axis=1).reshape(-1)
    binary_data.append(binary_representation)

binary_data = np.array(binary_data)

# ----------------------------- HD Calculation ----------------------------------

intra_distances = []
inter_distances = []

# Intra-HD: compare different segments within one sample
for sample in binary_data:
    segments = np.array_split(sample, n_segments)
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            distance = np.sum(segments[i] != segments[j])
            intra_distances.append(distance)

# Inter-HD: compare different samples
for i in range(len(binary_data)):
    for j in range(i + 1, len(binary_data)):
        distance = np.sum(binary_data[i] != binary_data[j])
        inter_distances.append(distance)

# ----------------------------- Normalization -----------------------------------

sequence_bit_length = 1000 * 8
segment_bit_length = segment_length * 8

intra_distances = np.array(intra_distances) / segment_bit_length
inter_distances = np.array(inter_distances) / sequence_bit_length

# ----------------------------- Save Results ------------------------------------

intra_hd_df = pd.DataFrame({"Intra-HD": intra_distances})
inter_hd_df = pd.DataFrame({"Inter-HD": inter_distances})
output_data = pd.concat([intra_hd_df, inter_hd_df], axis=1)
output_data.to_csv("outputHD_demo.csv", index=False)

# ----------------------------- Print Summary -----------------------------------

print(f"Intra-HD Mean: {np.mean(intra_distances):.4f}")
print(f"Inter-HD Mean: {np.mean(inter_distances):.4f}")
