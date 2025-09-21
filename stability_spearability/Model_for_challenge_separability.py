#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Challenge Separability Test (Demo)
----------------------------------
This script evaluates the minimum separability of PUF challenges.
- A CNN is trained on baseline and maximum condition samples.
- Intermediate challenge conditions are tested progressively.
- Logistic regression is used on extracted CNN features to
  determine the smallest resolvable change (temperature or current).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import gc

# ----------------------------- Config ------------------------------------------

SEQUENCE_SIZE = 10000
BATCH_SIZE = 32
MODEL_PATH = "saved_models/Separability_Original_CNN.pth"
os.makedirs("saved_models", exist_ok=True)


# ----------------------------- Data Utils --------------------------------------

def create_segments(df: pd.DataFrame, seq_len: int) -> np.ndarray:
    """Slice long sequence into fixed-length windows."""
    arr = df.iloc[:, 0].to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr)
    n = len(arr) // seq_len
    segments = np.array([
        arr[i * seq_len:(i + 1) * seq_len]
        for i in range(n)
    ], dtype=np.float32)
    return segments[:, None, :]


def augment_freq(x: torch.Tensor, phase_jitter=0.05, amp_jitter=0.05) -> torch.Tensor:
    """Frequency-domain augmentation with random phase/amp jitter and partial masking."""
    Xf = torch.fft.rfft(x, dim=-1)
    mag, ang = Xf.abs(), Xf.angle()
    mag = mag * (1 + amp_jitter * (2 * torch.rand_like(mag) - 1))
    ang = ang + phase_jitter * (2 * torch.rand_like(ang) - 1)
    if torch.rand(1).item() < 0.3:
        freq_mask = torch.ones_like(mag)
        mask_start = torch.randint(0, mag.shape[-1] // 2, (1,)).item()
        mask_len = torch.randint(5, mag.shape[-1] // 10, (1,)).item()
        freq_mask[..., mask_start:mask_start + mask_len] = 0.5
    else:
        freq_mask = torch.ones_like(mag)
    Xf2 = mag * freq_mask * torch.exp(1j * ang)
    return torch.fft.irfft(Xf2, n=x.shape[-1], dim=-1)


class PUFArrayDataset(Dataset):
    """Custom dataset with optional augmentation (time/frequency perturbations)."""
    def __init__(self, X: np.ndarray, y_multi: np.ndarray, augment: bool = False):
        self.X = X
        self.y_multi = y_multi
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        if self.augment:
            # Time-domain noise
            noise_std = np.random.uniform(0.1, 0.2) * x.std()
            x = x + torch.randn_like(x) * noise_std
            # Time shift
            shift = np.random.randint(-10, 11)
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=-1)
            # Frequency-domain jitter
            x = augment_freq(x, phase_jitter=0.05, amp_jitter=0.05)
            # Random segment replacement
            if np.random.rand() < 0.1:
                seg_len = np.random.randint(50, 200)
                start_idx = np.random.randint(0, x.shape[-1] - seg_len)
                replace_start = np.random.randint(0, x.shape[-1] - seg_len)
                if abs(replace_start - start_idx) > seg_len:
                    x[0, start_idx:start_idx + seg_len] = x[0, replace_start:replace_start + seg_len]
        y_m = torch.tensor(self.y_multi[idx], dtype=torch.long)
        return x, y_m


# ----------------------------- CNN Model ---------------------------------------

class CNN(nn.Module):
    """1D CNN used for feature extraction and classification."""
    def __init__(self, seq_len: int, num_classes: int):
        super().__init__()
        k, p = 100, 10
        self.conv1 = nn.Conv1d(1, 32, kernel_size=k)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(p)
        flat = 32 * ((seq_len - k + 1) // p)
        self.fc1 = nn.Linear(flat, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.drop = nn.Dropout(0.5)
        self.fcm = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = torch.relu(self.fc2(x))
        x = self.drop(x)
        feat = torch.relu(self.fc3(x))
        logits = self.fcm(feat)
        return logits, feat


# ----------------------------- Progressive Test --------------------------------

def progressive_training_test_single(device, condition_type):
    """
    Train CNN on baseline and maximum conditions,
    then test intermediate conditions to estimate separability threshold.
    """
    if condition_type == "Temperature":
        conditions = [
            (17.00, "Demo_Temp_17.00C.csv"),
            (17.01, "Demo_Temp_17.01C.csv"),
            (17.02, "Demo_Temp_17.02C.csv"),
            (17.03, "Demo_Temp_17.03C.csv"),
            (17.04, "Demo_Temp_17.04C.csv"),
            (17.05, "Demo_Temp_17.05C.csv"),
            (17.06, "Demo_Temp_17.06C.csv"),
            (17.07, "Demo_Temp_17.07C.csv"),
            (17.08, "Demo_Temp_17.08C.csv"),
            (17.09, "Demo_Temp_17.09C.csv"),
            (17.10, "Demo_Temp_17.10C.csv"),
        ]
        unit = "°C"
    else:  # Current
        conditions = [
            (50.00, "Demo_Curr_50.00mA.csv"),
            (50.01, "Demo_Curr_50.01mA.csv"),
            (50.02, "Demo_Curr_50.02mA.csv"),
            (50.03, "Demo_Curr_50.03mA.csv"),
            (50.04, "Demo_Curr_50.04mA.csv"),
            (50.05, "Demo_Curr_50.05mA.csv"),
            (50.06, "Demo_Curr_50.06mA.csv"),
            (50.07, "Demo_Curr_50.07mA.csv"),
            (50.08, "Demo_Curr_50.08mA.csv"),
            (50.09, "Demo_Curr_50.09mA.csv"),
            (50.10, "Demo_Curr_50.10mA.csv"),
        ]
        unit = "mA"

    # Use baseline and maximum for training
    train_indices = [0, 10]
    test_indices = [i for i in range(len(conditions)) if i not in train_indices]

    # Load and preprocess
    all_data, all_labels, all_values = [], [], []
    for i, (value, file_path) in enumerate(conditions):
        try:
            df = pd.read_csv(file_path, usecols=[0])  # demo: assume one column
            segs = create_segments(df, SEQUENCE_SIZE)
            # Normalize to [-1, 1]
            mn, mx = segs.min(), segs.max()
            segs = 2 * (segs - mn) / (mx - mn) - 1
            all_data.append(segs)
            all_labels.append(np.full(len(segs), i))
            all_values.append(value)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    # Training data (baseline + max)
    X_train = np.vstack([all_data[i] for i in train_indices])
    y_train = np.concatenate([all_labels[i] for i in train_indices])
    # Map labels to consecutive indices
    label_mapping = {old: new for new, old in enumerate(sorted(np.unique(y_train)))}
    y_train_mapped = np.array([label_mapping[l] for l in y_train])
    train_dataset = PUFArrayDataset(X_train, y_train_mapped, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Train CNN
    prog_model = CNN(SEQUENCE_SIZE, len(train_indices)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(prog_model.parameters(), lr=0.001, weight_decay=1e-4)

    best_loss, patience, max_patience = float('inf'), 0, 5
    best_model_state = None
    for epoch in range(30):
        prog_model.train()
        train_loss, total = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = prog_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
        epoch_loss = train_loss / total
        if epoch_loss < best_loss:
            best_loss, patience = epoch_loss, 0
            best_model_state = copy.deepcopy(prog_model.state_dict())
        else:
            patience += 1
            if patience >= max_patience:
                break
    if best_model_state:
        prog_model.load_state_dict(best_model_state)

    # Test intermediate conditions
    condition_results = []
    for i, data in enumerate(all_data):
        condition_value = all_values[i]
        X = torch.tensor(data, dtype=torch.float32).to(device)
        prog_model.eval()
        with torch.no_grad():
            if i in train_indices:
                mapped_label = label_mapping[i]
                outputs, _ = prog_model(X)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == mapped_label).float().mean().item()
                condition_results.append((condition_value, accuracy, "Train"))
            else:
                # Extract features
                feats = []
                for j in range(0, len(X), BATCH_SIZE):
                    batch = X[j:j + BATCH_SIZE]
                    _, feat = prog_model(batch)
                    feats.append(feat)
                features = torch.cat(feats, 0)

                # Binary classification with baseline
                base_data = all_data[0]
                base_X = torch.tensor(base_data, dtype=torch.float32).to(device)
                base_feats = []
                for j in range(0, len(base_X), BATCH_SIZE):
                    batch = base_X[j:j + BATCH_SIZE]
                    _, feat = prog_model(batch)
                    base_feats.append(feat)
                base_features = torch.cat(base_feats, 0)

                bin_X = torch.cat([base_features, features], 0).cpu().numpy()
                bin_y = np.concatenate([np.zeros(len(base_features)), np.ones(len(features))])
                if len(bin_X) >= 10 and len(np.unique(bin_y)) > 1:
                    clf = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear', C=1.0)
                    scores = cross_val_score(clf, bin_X, bin_y, cv=5)
                    accuracy = scores.mean()
                else:
                    accuracy = 0.5
                condition_results.append((condition_value, accuracy, "Test"))

    return {condition_type: condition_results}


# ----------------------------- Main --------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = {}

    # Temperature
    temp_results = progressive_training_test_single(device, "Temperature")
    all_results.update(temp_results)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()

    # Current
    curr_results = progressive_training_test_single(device, "Current")
    all_results.update(curr_results)

    # Summarize threshold
    print("Progressive Training Test Results:")
    for condition_type in ["Temperature", "Current"]:
        if condition_type in all_results:
            train_values = [r[0] for r in all_results[condition_type] if r[2] == "Train"]
            baseline = min(train_values)
            min_detect = None
            for val, acc, group in all_results[condition_type]:
                if group == "Test" and acc >= 0.9:
                    delta = abs(val - baseline)
                    if min_detect is None or delta < min_detect:
                        min_detect = delta
            unit = "°C" if condition_type == "Temperature" else "mA"
            if min_detect is not None:
                print(f"{condition_type} threshold: {min_detect}{unit}")
            else:
                print(f"{condition_type}: No clear threshold found")
