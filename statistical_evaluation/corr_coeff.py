#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Correlation Coefficient Analysis (Demo)
---------------------------------------
This script computes correlation matrices of PUF response sequences:
- Within one device (intra-device correlation)
- Across multiple devices (inter-device correlation)
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------- Dataset Utils -----------------------------------

def create_dataset(df, sequence_size):
    """Slice 1D series into windows and scale to [-1, 1]."""
    data = df.iloc[:, 0].values.astype(np.float32)
    num_samples = len(data) // sequence_size
    X = np.array([
        data[i * sequence_size:(i + 1) * sequence_size]
        for i in range(num_samples)
        if len(data[i * sequence_size:(i + 1) * sequence_size]) == sequence_size
    ])
    min_val, max_val = np.min(X), np.max(X)
    return 2 * (X - min_val) / (max_val - min_val) - 1


# ----------------------------- Load Demo Data ----------------------------------

# Note:
# - These CSV names are demo placeholders.
# - Replace with your own dataset: each file should contain one numeric column.
df1 = pd.read_csv("Demo_Device1.csv")
df2 = pd.read_csv("Demo_Device2.csv")
df3 = pd.read_csv("Demo_Device3.csv")

# ----------------------------- Correlation Analysis ----------------------------

# Example: within-device correlation for Device 1
X1 = create_dataset(df1, 1000)
corr_coeff_within_dev = X1[:500]  # take first 500 sequences for demo
num_sequences = corr_coeff_within_dev.shape[0]
corr_coeff_within_dev_mat = np.corrcoef(corr_coeff_within_dev, rowvar=True)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_coeff_within_dev_mat,
            cmap='coolwarm',
            annot=False,
            xticklabels=False,
            yticklabels=False,
            cbar_kws={'shrink': 0.8},
            fontfamily='Arial')
plt.gca().invert_yaxis()
plt.savefig('figures/corr_coeff/correlation_matrix_intradev1.png', dpi=300, bbox_inches='tight')

# Example: across-device correlation using one sample from each demo device
devs = [create_dataset(df, 1000)[-1:] for df in [df1, df2, df3]]
acrossdev_data = np.concatenate(devs, axis=0)
device_labels = ["D1", "D2", "D3"]

corr_coeff_across_dev_mat = np.corrcoef(acrossdev_data, rowvar=True)
plt.figure(figsize=(6, 5))
sns.heatmap(corr_coeff_across_dev_mat,
            cmap='Blues',
            annot=False,
            xticklabels=device_labels,
            yticklabels=device_labels,
            cbar_kws={'shrink': 0.8})
plt.xticks(rotation=45, fontsize=12, fontfamily='Arial')
plt.yticks(fontsize=12)
plt.gca().invert_yaxis()
plt.savefig('figures/corr_coeff/correlation_matrix_acrossdev.png', dpi=300, bbox_inches='tight')
