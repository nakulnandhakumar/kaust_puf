#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PUF Classifier with 1D CNN
--------------------------
This script trains a CNN-based classifier for PUF responses, with both
multi-class and auxiliary binary outputs. The number of classes is inferred
from the dataset labels.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------- Model Definition ---------------------------------

class CNN(nn.Module):
    """
    1D CNN with two heads:
      - Multi-class classification head (softmax logits)
      - Binary classification head (sigmoid probability)
    """
    def __init__(self, input_length: int, num_classes: int) -> None:
        super().__init__()
        kernel_size = 100
        pooling_size = 10

        self.conv1 = nn.Conv1d(1, 32, kernel_size=kernel_size)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(pooling_size)
        self.flatten = nn.Flatten()

        conv_out_len = (input_length - kernel_size + 1) // pooling_size
        feat_len = 32 * conv_out_len

        self.fc1 = nn.Linear(feat_len, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)

        self.fc_multi = nn.Linear(32, num_classes)
        self.fc_binary = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, return_features: bool = False):
        x = torch.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.pool(x)
        x = self.flatten(x)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        features = torch.relu(self.fc3(x))

        multi_out = self.fc_multi(features)
        binary_out = torch.sigmoid(self.fc_binary(features))

        if return_features:
            return multi_out, binary_out, features
        return multi_out, binary_out


# ----------------------------- Data Processing ----------------------------------

def create_dataset(df: pd.DataFrame, sequence_size: int) -> np.ndarray:
    """Slice series into windows of length sequence_size and scale to [-1, 1]."""
    data = df.iloc[:, 0].values.astype(np.float32)
    data = np.nan_to_num(data, nan=0.0)

    num_samples = len(data) // sequence_size
    X = np.array([
        data[i*sequence_size:(i+1)*sequence_size]
        for i in range(num_samples)
    ])
    X = np.expand_dims(X, axis=1)  # add channel dimension

    # Normalize to [-1, 1]
    min_val, max_val = np.min(X), np.max(X)
    eps = 1e-12
    X_scaled = 2 * (X - min_val) / (max_val - min_val + eps) - 1
    return X_scaled


# ----------------------------- Training and Eval --------------------------------

def train_and_evaluate(X, Y, B, sequence_size, device, num_epochs=10):
    """Train CNN with both multi-class and binary objectives."""
    num_classes = len(np.unique(Y))
    input_length = X.shape[2]
    model = CNN(input_length, num_classes).to(device)

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.long),
        torch.tensor(B, dtype=torch.float32),
    )
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    crit_multi = nn.CrossEntropyLoss()
    crit_bin = nn.BCELoss()
    optim_ = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        correct_multi, correct_bin, total = 0, 0, 0
        for x, y_multi, y_bin in train_loader:
            x, y_multi, y_bin = x.to(device), y_multi.to(device), y_bin.to(device)

            optim_.zero_grad()
            out_multi, out_bin = model(x)
            loss = 0.7 * crit_multi(out_multi, y_multi) + 0.3 * crit_bin(out_bin.squeeze(), y_bin)
            loss.backward()
            optim_.step()

            _, pred_multi = torch.max(out_multi, 1)
            correct_multi += (pred_multi == y_multi).sum().item()
            pred_bin = (out_bin.squeeze() >= 0.5).float()
            correct_bin += (pred_bin == y_bin).sum().item()
            total += y_multi.size(0)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Multi: {100*correct_multi/total:.2f}% | "
              f"Train Bin: {100*correct_bin/total:.2f}%")

    return model


def plot_confusion_matrix(cm, labels, save_path=None):
    """Plot and optionally save a confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.close()


# ----------------------------- Main ---------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequence_size = 10000

    # Demo: replace with your real CSV files
    df1 = pd.read_csv("Demo_Target_Challenge.csv")
    df2 = pd.read_csv("Demo_Non-Target_Challenge1.csv")
    df3 = pd.read_csv("Demo_Non-Target_Challenge2.csv")

    X1, X2, X3 = [create_dataset(df, sequence_size) for df in (df1, df2, df3)]
    Y1, Y2, Y3 = np.zeros(len(X1)), np.ones(len(X2)), np.full(len(X3), 2)
    B1, B2, B3 = np.ones(len(X1)), np.zeros(len(X2)), np.zeros(len(X3))

    X = np.concatenate([X1, X2, X3])
    Y = np.concatenate([Y1, Y2, Y3]).astype(int)
    B = np.concatenate([B1, B2, B3])

    model = train_and_evaluate(X, Y, B, sequence_size, device)
    torch.save(model.state_dict(), "saved_models/CNN.pth")
    print("Model saved.")
