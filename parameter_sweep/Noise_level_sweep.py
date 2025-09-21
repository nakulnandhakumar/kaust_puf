#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PUF Classifier Noise Sweep
--------------------------
This script trains a simple 1D CNN on PUF response data and sweeps over
different Gaussian noise levels. For each noise percentage, the model is
trained and evaluated on validation and extra test sets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
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

def create_dataset(df, sequence_size=10000):
    """Slice 1D series into windows and scale to [-1, 1]."""
    data = df.iloc[:, 0].values.astype(np.float32)
    num_samples = len(data) // sequence_size
    X = np.array([
        data[i*sequence_size:(i+1)*sequence_size]
        for i in range(num_samples)
    ], dtype=np.float32)
    X = np.expand_dims(X, axis=1)
    min_val, max_val = X.min(), X.max()
    eps = 1e-12
    return 2 * (X - min_val) / (max_val - min_val + eps) - 1


def add_gaussian_noise(data, noise_percentage=0):
    """Add Gaussian noise with std = noise_percentage% of data std."""
    if noise_percentage == 0:
        return data
    data_std = np.std(data)
    noise_std = (noise_percentage / 100.0) * data_std
    noise = np.random.normal(0, noise_std, data.shape).astype(np.float32)
    return data + noise


# ----------------------------- Training/Eval -----------------------------------

def train_and_evaluate(X, Y, sequence_size=10000, noise_percentage=0.0):
    """Train CNN on dataset (with noise already applied)."""
    dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = CNN(input_length=X.shape[2]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train short epochs for demo
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
        "Noise Percentage": noise_percentage,
        "Validation Accuracy": 100 * correct / total if total > 0 else 0.0
    }, model


def extra_validation(model, test_data, test_labels):
    """Evaluate model on a held-out test set."""
    tensor = torch.tensor(test_data).to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(test_data)):
            out = model(tensor[i].unsqueeze(0))
            pred = torch.round(torch.sigmoid(out))
            preds.append(pred.item())
    acc = 100 * np.sum(np.array(preds) == test_labels) / len(test_labels)
    return acc


# ----------------------------- Main --------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

sequence_size = 10000
results = []

# --- Load demo CSVs (replace with your own dataset) ---
df_real = pd.read_csv("Demo_Device_Real.csv")
df_fake = pd.read_csv("Demo_Device_Fake.csv")
df_real_probe = pd.read_csv("Demo_Device_Real_Probe.csv")
df_fake_probe = pd.read_csv("Demo_Device_Fake_Probe.csv")

# Slice into sequences
X_real = create_dataset(df_real, sequence_size)
X_fake = create_dataset(df_fake, sequence_size)
Y_real = np.ones(len(X_real), dtype=np.float32)
Y_fake = np.zeros(len(X_fake), dtype=np.float32)

X_dataset = np.concatenate([X_real, X_fake])
Y_dataset = np.concatenate([Y_real, Y_fake])

# Extra validation sets
X_real_probe = create_dataset(df_real_probe, sequence_size)
Y_real_probe = np.ones(len(X_real_probe), dtype=np.float32)
X_fake_probe = create_dataset(df_fake_probe, sequence_size)
Y_fake_probe = np.zeros(len(X_fake_probe), dtype=np.float32)

# Sweep through noise levels
for current_noise in [0.0, 5.0, 10.0, 20.0, 50.0]:
    noisy_X = add_gaussian_noise(X_dataset, current_noise)
    result, model = train_and_evaluate(noisy_X, Y_dataset, sequence_size, noise_percentage=current_noise)
    acc_real = extra_validation(model, add_gaussian_noise(X_real_probe, current_noise), Y_real_probe)
    acc_fake = extra_validation(model, add_gaussian_noise(X_fake_probe, current_noise), Y_fake_probe)
    result["Accuracy Real Probe"] = acc_real
    result["Accuracy Fake Probe"] = acc_fake
    results.append(result)

# Save results
os.makedirs("sweep_results", exist_ok=True)
pd.DataFrame(results).to_csv("sweep_results/noise_sweep.csv", index=False)
print("Sweep finished, results saved to sweep_results/noise_sweep.csv")
