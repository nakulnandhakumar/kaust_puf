#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Correlation Coefficient Analysis (Demo)
---------------------------------------
This script computes correlation matrices of PUF response sequences:
- Within-device pairwise correlations across multiple windows
- Inter-device pairwise correlations across multiple windows from each device
- Numeric correlation matrices are printed instead of plotted.
"""

import pandas as pd
import numpy as np


SEQUENCE_SIZE = 1000
MAX_WINDOWS_PER_DEVICE = 500

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


def normalize_windows(X):
    """Center and L2-normalize each window for Pearson pairwise correlations."""
    X = np.asarray(X, dtype=np.float64)
    X_centered = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X_centered, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-12
    if not np.any(valid):
        raise ValueError("All windows are constant; correlation is undefined.")
    return X_centered[valid] / norms[valid]


def pairwise_corr_values(A, B, same_device=False):
    """
    Return pairwise Pearson correlations between rows of A and B.
    For same-device comparisons, the diagonal self-correlations are removed.
    """
    corr = normalize_windows(A) @ normalize_windows(B).T
    if same_device:
        upper = np.triu_indices_from(corr, k=1)
        return corr[upper]
    return corr.reshape(-1)


def summarize(values):
    """Compact numeric summary for a vector of pairwise correlations."""
    if values.size == 0:
        return {
            "n_pairs": 0,
            "mean_corr": np.nan,
            "std_corr": np.nan,
            "min_corr": np.nan,
            "max_corr": np.nan,
        }
    return {
        "n_pairs": int(values.size),
        "mean_corr": float(np.mean(values)),
        "std_corr": float(np.std(values)),
        "min_corr": float(np.min(values)),
        "max_corr": float(np.max(values)),
    }


# ----------------------------- Load Demo Data ----------------------------------

# Demo placeholder input files.
# Each file should contain one numeric column.
device_files = {
    "D1": "Demo_Corr_Device_1.csv",
    "D2": "Demo_Corr_Device_2.csv",
    "D3": "Demo_Corr_Device_3.csv",
}

# ----------------------------- Correlation Analysis ----------------------------

datasets = {}
for label, path in device_files.items():
    df = pd.read_csv(path)
    windows = create_dataset(df, SEQUENCE_SIZE)[:MAX_WINDOWS_PER_DEVICE]
    if len(windows) < 2:
        raise ValueError(f"{path} must contain at least two full windows for pairwise correlation.")
    datasets[label] = windows

device_labels = list(datasets.keys())
mean_corr_mat = pd.DataFrame(index=device_labels, columns=device_labels, dtype=float)
summary_rows = []

for i, label_i in enumerate(device_labels):
    for j, label_j in enumerate(device_labels):
        same_device = i == j
        if i > j:
            mean_corr_mat.loc[label_i, label_j] = mean_corr_mat.loc[label_j, label_i]
            continue

        values = pairwise_corr_values(datasets[label_i], datasets[label_j], same_device=same_device)
        stats = summarize(values)
        mean_corr_mat.loc[label_i, label_j] = stats["mean_corr"]
        if i != j:
            mean_corr_mat.loc[label_j, label_i] = stats["mean_corr"]

        summary_rows.append({
            "device_a": label_i,
            "device_b": label_j,
            "comparison": "within" if same_device else "between",
            **stats,
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("corr_pairwise_summary.csv", index=False)

print(f"Using up to {MAX_WINDOWS_PER_DEVICE} windows per device, sequence size={SEQUENCE_SIZE}.")
print("\nMean pairwise correlation matrix:")
print(mean_corr_mat)
print("\nPairwise correlation summary:")
print(summary_df)
print("\nSaved pairwise summary to corr_pairwise_summary.csv")
