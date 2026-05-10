#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hamming Distance (HD) Calculation Demo
--------------------------------------
This script computes auxiliary key-level Hamming distance metrics from a
prepared one-column PUF response sequence CSV:
- within-key segment HD
- between-key HD

The input CSV should contain one numeric sequence column prepared from the
local data organization.
"""

import numpy as np
import pandas as pd


# ----------------------------- Load Data ---------------------------------------

file_name = "Demo_Stat_Input.csv"
sequence = pd.read_csv(file_name, usecols=[0]).iloc[:, 0].dropna().to_numpy(dtype=np.float32)

key_length = 1000
num_keys = 1000
required_points = key_length * num_keys
if sequence.size < required_points:
    raise ValueError(f"Need at least {required_points} points to form {num_keys} keys.")

raw_sequences = sequence[:required_points].reshape(num_keys, key_length)

# ----------------------------- Parameters --------------------------------------

n_segments = 10
segment_length = key_length // n_segments

# ----------------------------- Binarization ------------------------------------

binary_data = []
for sample in raw_sequences:
    sample_min, sample_max = float(sample.min()), float(sample.max())
    if sample_max == sample_min:
        discretized = np.zeros(sample.shape, dtype=np.uint8)
    else:
        normalized = (sample - sample_min) / (sample_max - sample_min)
        discretized = np.round(normalized * 255).astype(np.uint8)

    binary_representation = np.unpackbits(discretized[:, np.newaxis], axis=1).reshape(-1)
    binary_data.append(binary_representation)

binary_data = np.array(binary_data)

# ----------------------------- HD Calculation ----------------------------------

within_key_distances = []
between_key_distances = []

for sample in binary_data:
    segments = np.array_split(sample, n_segments)
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            within_key_distances.append(np.sum(segments[i] != segments[j]))

for i in range(len(binary_data)):
    for j in range(i + 1, len(binary_data)):
        between_key_distances.append(np.sum(binary_data[i] != binary_data[j]))

# ----------------------------- Normalization -----------------------------------

sequence_bit_length = key_length * 8
segment_bit_length = segment_length * 8

within_key_distances = np.array(within_key_distances) / segment_bit_length
between_key_distances = np.array(between_key_distances) / sequence_bit_length

# ----------------------------- Save Results ------------------------------------

within_hd_df = pd.DataFrame({"Within-key segment HD": within_key_distances})
between_hd_df = pd.DataFrame({"Between-key HD": between_key_distances})
output_data = pd.concat([within_hd_df, between_hd_df], axis=1)
output_data.to_csv("outputHD_demo.csv", index=False)

# ----------------------------- Print Summary -----------------------------------

print(f"Within-key segment HD Mean: {np.mean(within_key_distances):.4f}")
print(f"Between-key HD Mean: {np.mean(between_key_distances):.4f}")
