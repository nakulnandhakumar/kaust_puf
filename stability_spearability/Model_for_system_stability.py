import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import torch.fft as fft
import torch.nn.functional as F
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Configuration
SEQUENCE_SIZE = 10000
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 3e-4
TRAIN_VAL_RATIO = 0.7
EARLYSTOP_PATIENCE = 20
LR_SCHED_PATIENCE = 7
LR_REDUCTION_FACTOR = 0.6
MODEL_PATH = "saved_models/Stability_Original_CNN.pth"
CONFIDENCE_THRESHOLD = 0.5

os.makedirs("saved_models", exist_ok=True)


# Helper Functions
def create_segments(df: pd.DataFrame, seq_len: int) -> np.ndarray:
    arr = df.iloc[:, 0].to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr)
    n = len(arr) // seq_len
    segments = np.array([
        arr[i * seq_len:(i + 1) * seq_len]
        for i in range(n)
    ], dtype=np.float32)
    return segments[:, None, :]


def augment_freq(x: torch.Tensor, phase_jitter=0.1, amp_jitter=0.1) -> torch.Tensor:
    # Frequency domain augmentation
    Xf = fft.rfft(x, dim=-1)
    mag = Xf.abs()
    ang = Xf.angle()
    mag = mag * (1 + amp_jitter * (2 * torch.rand_like(mag) - 1))
    ang = ang + phase_jitter * (2 * torch.rand_like(ang) - 1)
    Xf2 = mag * torch.exp(1j * ang)
    return fft.irfft(Xf2, n=x.shape[-1], dim=-1)


# Dataset Class
class PUFArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        if self.augment:
            # Add noise for robustness
            noise_std = 0.15 * x.std()
            x = x + torch.randn_like(x) * noise_std

            # Random time shift
            shift = np.random.randint(-10, 10)
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=-1)

            # Frequency domain augmentation
            x = augment_freq(x)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


# Model Definition
class CNN(nn.Module):
    def __init__(self, seq_len: int, num_classes: int):
        super().__init__()
        k, p = 80, 8
        self.conv1 = nn.Conv1d(1, 16, kernel_size=k, stride=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool = nn.MaxPool1d(p)

        # Calculate flattened size
        flat_size = ((seq_len - k) // 2 + 1) // p
        flat = 16 * flat_size

        self.fc1 = nn.Linear(flat, 64)
        self.fc2 = nn.Linear(64, 32)
        self.drop = nn.Dropout(0.4)
        self.fcm = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        feat = torch.relu(self.fc2(x))
        x = self.drop(feat)
        out = self.fcm(x)
        return out, feat


# Training Function
def train_model(train_loader, val_loader, model, device, num_classes):
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_REDUCTION_FACTOR,
                                  patience=LR_SCHED_PATIENCE, verbose=True)

    best_loss = float('inf')
    patience = 0
    history = {'t_loss': [], 'v_loss': [], 't_acc': [], 'v_acc': [], 'v_confidence': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        t_loss = t_corr = t_tot = 0
        for xb, y in train_loader:
            xb, y = xb.to(device), y.to(device)
            optimizer.zero_grad()
            out, _ = model(xb)
            loss = crit(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item() * xb.size(0)
            _, preds = torch.max(out, 1)
            t_corr += (preds == y).sum().item()
            t_tot += xb.size(0)
        train_loss = t_loss / t_tot
        train_acc = t_corr / t_tot

        model.eval()
        v_loss = v_corr = v_tot = 0
        confidences = []
        with torch.no_grad():
            for xb, y in val_loader:
                xb, y = xb.to(device), y.to(device)
                out, _ = model(xb)
                loss = crit(out, y)
                v_loss += loss.item() * xb.size(0)
                _, preds = torch.max(out, 1)
                v_corr += (preds == y).sum().item()
                v_tot += xb.size(0)

                probs = F.softmax(out, dim=1)
                confidence, _ = torch.max(probs, dim=1)
                confidences.extend(confidence.cpu().numpy().tolist())

        val_loss = v_loss / v_tot
        val_acc = v_corr / v_tot
        val_confidence = np.mean(confidences)

        scheduler.step(val_loss)
        history['t_loss'].append(train_loss)
        history['v_loss'].append(val_loss)
        history['t_acc'].append(train_acc)
        history['v_acc'].append(val_acc)
        history['v_confidence'].append(val_confidence)

        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train loss={train_loss:.4f}, acc={train_acc:.2%} | "
              f"Val loss={val_loss:.4f}, acc={val_acc:.2%}, Conf={val_confidence:.4f} | LR={cur_lr:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience += 1
            if patience >= EARLYSTOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model weights
    model.load_state_dict(torch.load(MODEL_PATH))
    return model


# Evaluation Function
def evaluate_generalization(model, test_loaders, device, confidence_threshold):
    model.eval()
    results = []

    for loader_name, loader, true_label in test_loaders:
        all_preds = []
        all_confidences = []
        all_raw_preds = []
        all_features = []

        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                out, feat = model(xb)
                probs = F.softmax(out, dim=1)
                confidence, raw_pred = torch.max(probs, dim=1)

                pred = raw_pred.clone()
                unknown_idx = confidence < confidence_threshold
                pred[unknown_idx] = model.fcm.out_features

                all_preds.extend(pred.cpu().numpy())
                all_raw_preds.extend(raw_pred.cpu().numpy())
                all_confidences.extend(confidence.cpu().numpy())
                all_features.extend(feat.cpu().numpy())

        num_classes = model.fcm.out_features
        pred_counts = np.bincount(all_preds, minlength=num_classes + 1)
        pred_percent = pred_counts / len(all_preds) * 100

        raw_pred_counts = np.bincount(all_raw_preds, minlength=num_classes)
        raw_pred_percent = raw_pred_counts / len(all_raw_preds) * 100

        accuracy = pred_counts[true_label] / len(all_preds) * 100

        results.append({
            'name': loader_name,
            'true_label': true_label,
            'predictions': all_preds,
            'raw_predictions': all_raw_preds,
            'confidences': all_confidences,
            'features': np.array(all_features),
            'pred_counts': pred_counts,
            'pred_percent': pred_percent,
            'raw_pred_counts': raw_pred_counts,
            'raw_pred_percent': raw_pred_percent,
            'accuracy': accuracy
        })

    return results

# Main Script
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Training data files (Closely spaced CRPs)
    train_files = [
        ("Demo-0min.csv", 0),
        ("Demo-16.90C.csv", 1),
        ("Demo-17.10C.csv", 2),
        ("Demo-49.90mA.csv", 3),
        ("Demo-50.10mA.csv", 4),
    ]

    train_segments, train_Y = [], []
    val_segments, val_Y = [], []

    for path, label in train_files:
        df = pd.read_csv(path, usecols=[4])
        segs = create_segments(df, SEQUENCE_SIZE)

        # Normalization
        mn, mx = segs.min(), segs.max()
        segs = 2 * (segs - mn) / (mx - mn) - 1

        # Train/validation split
        n = len(segs)
        split = int(TRAIN_VAL_RATIO * n)
        indices = np.random.permutation(n)
        train_indices = indices[:split]
        val_indices = indices[split:]

        train_segments.append(segs[train_indices])
        train_Y.append(np.full(len(train_indices), label, dtype=np.int64))
        val_segments.append(segs[val_indices])
        val_Y.append(np.full(len(val_indices), label, dtype=np.int64))

    X_train = np.vstack(train_segments)
    Y_train = np.concatenate(train_Y)
    X_val = np.vstack(val_segments)
    Y_val = np.concatenate(val_Y)

    # Create data loaders
    train_ds = PUFArrayDataset(X_train, Y_train, augment=False)
    val_ds = PUFArrayDataset(X_val, Y_val, augment=False)

    num_workers = 0 if os.name == 'nt' else 2
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    # Create and train model
    num_classes = len(np.unique(Y_train))
    model = CNN(SEQUENCE_SIZE, num_classes).to(device)

    if not os.path.exists(MODEL_PATH):
        model = train_model(train_loader, val_loader, model, device, num_classes)
    else:
        model.load_state_dict(torch.load(MODEL_PATH))

    model.eval()

    # Test data files
    test_files = [
        ("0", "Demo-0min.csv", 0),
        ("10", "Demo-10min.csv", 0),
        ("20", "Demo-20min.csv", 0),
        ("30", "Demo-30min.csv", 0),
        ("40", "Demo-40min.csv", 0),
        ("50", "Demo-50min.csv", 0),
        ("60", "Demo-60min.csv", 0),
    ]

    # Prepare test data loaders
    test_loaders = []
    for name, path, label in test_files:
        df = pd.read_csv(path, usecols=[4])
        segs = create_segments(df, SEQUENCE_SIZE)

        mn, mx = segs.min(), segs.max()
        segs = 2 * (segs - mn) / (mx - mn) - 1

        dummy_labels = np.zeros(len(segs), dtype=np.int64)
        test_ds = PUFArrayDataset(segs, dummy_labels, augment=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
        test_loaders.append((name, test_loader, label))

    # Evaluate model
    results = evaluate_generalization(model, test_loaders, device, CONFIDENCE_THRESHOLD)

    class_names = ['Original', '-$\\Delta$T', '+$\\Delta$T', '-$\\Delta$I', '+$\\Delta$I', 'Unknown']

    # Output results
    for res in results:
        print(f"\nResults - {res['name']} (True: {class_names[res['true_label']]})")
        print(f"Accuracy: {res['accuracy']:.1f}%")
