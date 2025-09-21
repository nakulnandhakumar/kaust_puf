#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PUF Classifier Sequence Length Sweep
------------------------------------
This script trains a simple 1D CNN on PUF response data and sweeps over
different sequence lengths. Accuracy results are recorded and saved to CSV.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os

# ----------------------------- CNN Model ---------------------------------------

class CNN(nn.Module):
    """
    Simplified CNN for sweep experiments.
    Only binary classification head is kept.
    The full model has both multi-class and binary heads.
    """
    def __init__(self, input_length):
        super().__init__()
        kernel_size = 100
        pooling_size = 10
        self.conv1 = nn.Conv1d(1, 32, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(pooling_size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * ((input_length - kernel_size + 1) // pooling_size), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# ----------------------------- Dataset Utils -----------------------------------

def create_datasets(df, sequence_size):
    """Slice 1D series into non-overlapping windows and scale to [-1, 1]."""
    data = df.iloc[:, 0].values.astype(np.float32)
    num_samples = len(data) // sequence_size
    X = np.array([data[i*sequence_size:(i+1)*sequence_size]
                  for i in range(num_samples)], dtype=np.float32)
    X = np.expand_dims(X, axis=1)
    min_val, max_val = X.min(), X.max()
    eps = 1e-12
    return 2 * (X - min_val) / (max_val - min_val + eps) - 1


# ----------------------------- Training/Eval -----------------------------------

def train_and_evaluate(sequence_size):
    """Train CNN on demo data with given sequence length."""
    print(f"Training with sequence_size={sequence_size}")

    # --- Load demo CSVs (replace with your own dataset) ---
    df_real = pd.read_csv("Demo_Device_Real.csv")
    df_fake = pd.read_csv("Demo_Device_Fake.csv")

    X_real = create_datasets(df_real, sequence_size)
    X_fake = create_datasets(df_fake, sequence_size)

    Y_real = np.ones(len(X_real), dtype=np.float32)
    Y_fake = np.zeros(len(X_fake), dtype=np.float32)

    # Split into train/val
    val_size = int(0.3 * (len(X_real) + len(X_fake)))
    X = np.concatenate([X_real, X_fake])
    Y = np.concatenate([Y_real, Y_fake])
    X_val, Y_val = X[-val_size:], Y[-val_size:]
    X_train, Y_train = X[:-val_size], Y[:-val_size]

    # Build datasets
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(Y_train)), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(Y_val)), batch_size=32, shuffle=False)

    # Model setup
    model = CNN(input_length=X.shape[2]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop (short for demo)
    for epoch in range(5):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb).squeeze()
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    # Validation accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = torch.round(torch.sigmoid(model(xb).squeeze()))
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    return {
        "Sequence Size": sequence_size,
        "Validation Accuracy": 100 * correct / total if total > 0 else 0.0
    }


# ----------------------------- Sweep Loop --------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

results = []
# Sweep only over sequence length (noise is handled by separate noise_sweep script)
for seq_len in range(8000, 10001, 500):
    results.append(train_and_evaluate(seq_len))

# Save results
os.makedirs("sweep_results", exist_ok=True)
pd.DataFrame(results).to_csv("sweep_results/sequence_sweep.csv", index=False)
print("Sweep finished, results saved to sweep_results/sequence_sweep.csv")
