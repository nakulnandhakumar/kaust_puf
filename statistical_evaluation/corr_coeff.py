#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Correlation Coefficient Analysis (Demo)
---------------------------------------
This script computes correlation matrices of PUF response sequences:
- Within one device (intra-device correlation)
- Across multiple devices (inter-device correlation)
- Numeric correlation matrices are printed instead of plotted.
"""

import pandas as pd
import numpy as np

# ----------------------------- Dataset Utils -----------------------------------

def create_dataset(df, sequence_size):
    """Slice 1D series into windows and scale to [-1, 1]."""
    data = df.iloc[:, 0].values.astype(np.float32)
    data = np.nan_to_num(data, nan=0.0)
    num_samples = len(data) // sequence_size
    X = np.array([
        data[i * sequence_size:(i + 1) * sequence_size]
        for i in range(num_samples)
        if len(data[i * sequence_size:(i + 1) * sequence_size]) == sequence_size
    ])
    if X.size == 0:
        raise ValueError(f"No full sequences of length {sequence_size} could be created from the input data.")
    min_val, max_val = np.min(X), np.max(X)
    return 2 * (X - min_val) / (max_val - min_val + 1e-12) - 1


# ----------------------------- Load Demo Data ----------------------------------

# Demo placeholder input files.
# Each file should contain one numeric column.
df1 = pd.read_csv("Demo_Corr_Device_1.csv")
df2 = pd.read_csv("Demo_Corr_Device_2.csv")
df3 = pd.read_csv("Demo_Corr_Device_3.csv")

# ----------------------------- Correlation Analysis ----------------------------

# Example: within-device correlation for Device 1
X1 = create_dataset(df1, 1000)
corr_coeff_within_dev = X1[:500]  # take first 500 sequences for demo
corr_coeff_within_dev_mat = np.corrcoef(corr_coeff_within_dev, rowvar=True)

# Example: across-device correlation using one sample from each demo device
devs = [create_dataset(df, 1000)[-1:] for df in [df1, df2, df3]]
acrossdev_data = np.concatenate(devs, axis=0)
device_labels = ["D1", "D2", "D3"]

corr_coeff_across_dev_mat = np.corrcoef(acrossdev_data, rowvar=True)

print("Within-device correlation matrix (Device 1):")
print(corr_coeff_within_dev_mat)
print("\nAcross-device correlation matrix:")
print(pd.DataFrame(corr_coeff_across_dev_mat, index=device_labels, columns=device_labels))
