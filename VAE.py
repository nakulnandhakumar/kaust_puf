#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script implements a 1D VAE for generating and reconstructing PUF response
signals. A CNN detector is trained jointly to distinguish real vs reconstructed
data, providing adversarial feedback. The VAE also incorporates KL regularization
and statistical constraints (mean/std).
"""

from __future__ import annotations

import os
import csv
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Tuple, List


# ----------------------------- CNN Detector -------------------------------------

class CNN(nn.Module):
    """
    1D CNN with two heads:
      - multi class head: logits for C classes
      - binary head: probability for Target vs Non Target
    """
    def __init__(self, input_length: int, num_classes: int) -> None:
        super().__init__()
        kernel_size = 100
        pooling_size = 10

        self.conv1 = nn.Conv1d(1, 32, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(pooling_size)
        self.flatten = nn.Flatten()

        conv_out_len = (input_length - kernel_size + 1)
        pooled_len = conv_out_len // pooling_size
        feat_len = 32 * pooled_len

        self.fc1 = nn.Linear(feat_len, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc_multi = nn.Linear(32, num_classes)
        self.fc_binary = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = torch.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = self.flatten(x)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        features = torch.relu(self.fc3(x))

        multi_logits = self.fc_multi(features)
        binary_prob = torch.sigmoid(self.fc_binary(features))

        if return_features:
            return multi_logits, binary_prob, features
        return multi_logits, binary_prob


# ----------------------------- VAE ----------------------------------------------

class VAE(nn.Module):
    """
    1D VAE. Encoder reduces sequence to latent vector z. Decoder reconstructs sequence.
    """
    def __init__(self, input_length: int, latent_dim: int = 128) -> None:
        super().__init__()
        self.input_length = input_length

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Flatten()
        )
        enc_out_length = input_length // 8
        enc_out_size = 128 * enc_out_length

        self.fc_mu = nn.Sequential(nn.Linear(enc_out_size, latent_dim), nn.Dropout(0.3))
        self.fc_logvar = nn.Sequential(nn.Linear(enc_out_size, latent_dim), nn.Dropout(0.3))

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, enc_out_size)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, enc_out_length)),
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.ConvTranspose1d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar) + 1e-6
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc = self.encoder(x)
        mu = self.fc_mu(enc)
        logvar = torch.clamp(self.fc_logvar(enc), min=-10, max=10)
        z = self.reparameterize(mu, logvar)
        dec = self.fc_decode(z)
        x_rec = self.decoder(dec)
        return x_rec, mu, logvar


# ----------------------------- Data utils ---------------------------------------

def split_into_sequences(df: pd.DataFrame, sequence_length: int) -> np.ndarray:
    """
    Slice first column into windows and scale to range [-1, 1].
    """
    data = df.iloc[:, 0].to_numpy(dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0)
    mn, mx = data.min(), data.max()
    eps = 1e-12
    data = 2 * (data - mn) / (mx - mn + eps) - 1.0

    n = len(data) // sequence_length
    seqs = np.array([data[i*sequence_length:(i+1)*sequence_length] for i in range(n)], dtype=np.float32)
    return seqs


def init_weights(m: nn.Module) -> None:
    """
    Xavier init for Conv1d ConvTranspose1d Linear.
    """
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ----------------------------- Original task loader -----------------------------

def load_original_task_data(
    device: torch.device,
    sequence_length: int = 10000,
    manifest_path: str = "original_task_manifest.csv"
) -> DataLoader:
    """
    Load evaluation data for the original task by a manifest CSV.
    Manifest columns: filepath, multi_label, binary_label
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. "
            f"Create a CSV with columns: filepath, multi_label, binary_label."
        )

    def frame_df(df: pd.DataFrame) -> np.ndarray:
        arr = df.iloc[:, 0].to_numpy(dtype=np.float32)
        arr = np.nan_to_num(arr)
        n = len(arr) // sequence_length
        segs = np.array([arr[i*sequence_length:(i+1)*sequence_length] for i in range(n)], dtype=np.float32)
        # scale to [-1, 1] per file
        mn, mx = segs.min(), segs.max()
        eps = 1e-12
        segs = 2 * (segs - mn) / (mx - mn + eps) - 1.0
        return segs[:, None, :]

    X_list: List[np.ndarray] = []
    Y_multi_list: List[np.ndarray] = []
    Y_bin_list: List[np.ndarray] = []

    with open(manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row["filepath"]
            mlabel = int(row["multi_label"])
            blabel = float(row["binary_label"])
            if not os.path.exists(path):
                print(f"Warning missing file: {path}")
                continue
            df = pd.read_csv(path, usecols=[0])
            segs = frame_df(df)
            if segs.size == 0:
                continue
            X_list.append(segs)
            Y_multi_list.append(np.full(len(segs), mlabel, dtype=np.int64))
            Y_bin_list.append(np.full(len(segs), blabel, dtype=np.float32))

    if not X_list:
        raise RuntimeError("No data loaded from manifest entries")

    X = np.vstack(X_list)
    Ym = np.concatenate(Y_multi_list)
    Yb = np.concatenate(Y_bin_list)

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Ym, dtype=torch.long),
        torch.tensor(Yb, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=32, shuffle=False)


def evaluate_original_task(cnn_model: CNN, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Evaluate CNN on original task.
    Returns multi class accuracy and binary accuracy.
    """
    cnn_model.eval()
    cm_ok = 0
    cb_ok = 0
    total = 0
    with torch.no_grad():
        for x, y_multi, y_bin in loader:
            x = x.to(device)
            y_multi = y_multi.to(device)
            y_bin = y_bin.to(device)
            logits, p_bin = cnn_model(x)
            _, predm = torch.max(logits, 1)
            cm_ok += (predm == y_multi).sum().item()
            cb_ok += ((p_bin.squeeze() >= 0.5).float() == y_bin).sum().item()
            total += y_multi.size(0)
    return cm_ok / max(1, total), cb_ok / max(1, total)


# ----------------------------- Losses and training ------------------------------

def combined_loss(
    cnn_model: CNN,
    x_rec: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    lambda_feedback: float = 0.0,
    lambda_kl: float = 0.0,
    alpha: float = 1.2
) -> Tuple[torch.Tensor, float, float, float, float, float]:
    """
    Compute total VAE loss:
      - reconstruction (MSE + L1)
      - feature feedback from CNN
      - KL divergence
      - mean/std consistency
    """
    # Reconstruction
    rec = 0.7 * F.mse_loss(x_rec, x, reduction="mean") + 0.3 * F.l1_loss(x_rec, x, reduction="mean")

    # Match mean and std across time
    if x.size(0) > 1:
        mean_loss = F.mse_loss(x_rec.mean(dim=2), x.mean(dim=2))
        std_loss = F.l1_loss(x_rec.std(dim=2, unbiased=False), x.std(dim=2, unbiased=False))
    else:
        mean_loss = torch.tensor(0.0, device=x.device)
        std_loss = torch.tensor(0.0, device=x.device)

    # Feature feedback from CNN
    try:
        cnn_model.eval()
        with torch.no_grad():
            _, _, f_real = cnn_model(x, return_features=True)
        cnn_model.train()
        _, _, f_fake = cnn_model(x_rec, return_features=True)
        f_real = F.normalize(f_real, dim=1)
        f_fake = F.normalize(f_fake, dim=1)
        feat = 0.7 * F.mse_loss(f_fake, f_real, reduction="mean") + 0.3 * F.l1_loss(f_fake, f_real, reduction="mean")
    except Exception:
        feat = torch.tensor(0.0, device=x.device)

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total = alpha * rec + lambda_feedback * feat + lambda_kl * kl + mean_loss + std_loss
    return total, rec.item(), feat.item(), kl.item(), mean_loss.item(), std_loss.item()


def train_vae_with_adversarial_cnn(
    X: np.ndarray,
    cnn_model: CNN,
    device: torch.device,
    epochs: int = 250,
    latent_dim: int = 128,
    sequence_length: int = 10000,
    manifest_path: str = "original_task_manifest.csv"
):
    """
    Train VAE while adversarially finetuning CNN to detect reconstructions,
    with periodic evaluation on original task data.
    """
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(1))
    trn_size = int(0.7 * len(dataset))
    val_size = len(dataset) - trn_size
    trn_set, val_set = random_split(dataset, [trn_size, val_size])
    trn_loader = DataLoader(trn_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    vae = VAE(input_length=X.shape[1], latent_dim=latent_dim).to(device)
    vae.apply(init_weights)

    opt_vae = torch.optim.Adam(vae.parameters(), lr=1e-4, weight_decay=1e-5)
    opt_cnn = torch.optim.Adam(cnn_model.parameters(), lr=5e-5, weight_decay=1e-5)
    sch_vae = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_vae, mode="max", factor=0.5, patience=10)
    sch_cnn = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_cnn, mode="max", factor=0.1, patience=5)

    # Load original task for evaluation and CNN retention
    orig_loader = load_original_task_data(device, sequence_length, manifest_path)
    base_multi, base_bin = evaluate_original_task(cnn_model, orig_loader, device)
    print(f"Original task baseline. Multi: {base_multi:.4f}  Binary: {base_bin:.4f}")

    multi_thr = max(0.85, base_multi * 0.95)
    bin_thr = max(0.85, base_bin * 0.95)

    feedback_start_epoch = 10
    final_lambda_feedback = 0.2
    initial_lambda_feedback = 0.01
    kl_start_epoch = 10
    lambda_kl_initial = 1e-6
    lambda_kl_max = 5e-4

    perf_hist = {"multi": [base_multi], "bin": [base_bin]}
    window = 3
    viol = 0
    max_viol = 3
    recovery = False
    patience = 3
    early = 0

    def cyc(loader):
        while True:
            for b in loader:
                yield b

    orig_iter = cyc(orig_loader)

    for epoch in range(epochs):
        # schedule lambdas
        if epoch < feedback_start_epoch:
            lam_f = initial_lambda_feedback
        else:
            p = (epoch - feedback_start_epoch) / max(1, epochs - feedback_start_epoch)
            lam_f = initial_lambda_feedback + p * (final_lambda_feedback - initial_lambda_feedback)

        if epoch < kl_start_epoch:
            lam_kl = lambda_kl_initial
        else:
            p = (epoch - kl_start_epoch) / max(1, epochs - kl_start_epoch)
            lam_kl = lambda_kl_initial + p * (lambda_kl_max - lambda_kl_initial)

        # train
        vae.train()
        cnn_model.train()
        tr_loss = tr_rec = tr_feat = tr_kl = tr_mean = tr_std = 0.0

        for x_batch, in trn_loader:
            x = x_batch.to(device)

            # train VAE
            opt_vae.zero_grad()
            x_rec, mu, logvar = vae(x)
            total, rec, feat, kl, mloss, sloss = combined_loss(
                cnn_model, x_rec, x, mu, logvar, lambda_feedback=lam_f, lambda_kl=lam_kl
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5.0)
            opt_vae.step()

            tr_loss += total.item()
            tr_rec += rec
            tr_feat += feat
            tr_kl += kl
            tr_mean += mloss
            tr_std += sloss

            # train CNN adversarially and retain original task
            opt_cnn.zero_grad()
            real = x
            fake = x_rec.detach()
            cnn_in = torch.cat([real, fake], 0)
            cnn_lbl = torch.cat([
                torch.ones(real.size(0), 1, device=device),
                torch.zeros(fake.size(0), 1, device=device)
            ], 0)
            _, p_bin = cnn_model(cnn_in)
            det_loss = F.binary_cross_entropy(p_bin, cnn_lbl)

            try:
                o_x, o_y_m, o_y_b = next(orig_iter)
                o_x = o_x.to(device)
                o_y_m = o_y_m.to(device)
                o_y_b = o_y_b.to(device)
                logits, p = cnn_model(o_x)
                loss_m = F.cross_entropy(logits, o_y_m)
                loss_b = F.binary_cross_entropy(p.squeeze(), o_y_b)
                orig_loss = 0.7 * loss_m + 0.3 * loss_b
            except Exception:
                orig_loss = torch.tensor(0.0, device=device)

            # adaptive weights
            avg_m = sum(perf_hist["multi"][-window:]) / min(window, len(perf_hist["multi"]))
            avg_b = sum(perf_hist["bin"][-window:]) / min(window, len(perf_hist["bin"]))

            if recovery:
                w_det, w_orig = 0.1, 0.9
            elif avg_m < multi_thr or avg_b < bin_thr:
                w_det, w_orig = 0.3, 0.7
            elif epoch < 20:
                w_det, w_orig = 0.4, 0.6
            elif epoch < 50:
                w_det, w_orig = 0.5, 0.5
            else:
                w_det, w_orig = 0.55, 0.45

            cnn_loss = w_det * det_loss + w_orig * orig_loss
            cnn_loss.backward()
            torch.nn.utils.clip_grad_norm_(cnn_model.parameters(), max_norm=5.0)
            opt_cnn.step()

        # validation
        vae.eval()
        cnn_model.eval()
        with torch.no_grad():
            val_ok = tot = 0
            for x_batch, in DataLoader(val_set, batch_size=32, shuffle=False):
                x = x_batch.to(device)
                x_rec, _, _ = vae(x)
                # use binary head output directly
                p = cnn_model(x_rec)[1]
                val_ok += (p.squeeze() >= 0.5).float().sum().item()
                tot += x.size(0)
            val_acc = val_ok / max(1, tot)

        # evaluate on original task
        try:
            cur_m, cur_b = evaluate_original_task(cnn_model, orig_loader, device)
            perf_hist["multi"].append(cur_m)
            perf_hist["bin"].append(cur_b)
        except Exception:
            cur_m = perf_hist["multi"][-1]
            cur_b = perf_hist["bin"][-1]

        avg_m = sum(perf_hist["multi"][-window:]) / min(window, len(perf_hist["multi"]))
        avg_b = sum(perf_hist["bin"][-window:]) / min(window, len(perf_hist["bin"]))

        # violation logic
        if avg_m < multi_thr or avg_b < bin_thr:
            viol += 1
            if viol == max_viol and not recovery:
                recovery = True
                print("Entering recovery mode.")
        else:
            if viol > 0:
                print("Original task recovered.")
            viol = 0
            if recovery:
                print("Exiting recovery mode.")
                recovery = False

        # schedulers
        sch_vae.step(val_acc)
        sch_cnn.step(val_acc)

        # early stop
        if val_acc >= 0.98 and not recovery and avg_m >= 0.9 and avg_b >= 0.9:
            early += 1
            if early >= patience:
                print("Early stopping.")
                break
        else:
            early = 0

        print(f"Epoch {epoch+1}/{epochs} "
              f"TrainLoss {tr_loss/len(trn_loader):.4f} Rec {tr_rec/len(trn_loader):.4f} "
              f"Feat {tr_feat/len(trn_loader):.4f} KL {tr_kl/len(trn_loader):.4f} "
              f"ValAcc {val_acc:.4f} Orig Multi {cur_m:.4f} Orig Bin {cur_b:.4f} "
              f"lamF {lam_f:.4f} lamKL {lam_kl:.6f}")

    return vae, orig_loader


# ----------------------------- Validate through CNN -----------------------------

def evaluate_with_cnn(cnn_model: CNN, vae: VAE, val_loader: DataLoader, device: torch.device) -> float:
    """
    Accuracy of CNN binary head on VAE reconstructions. For quick monitoring.
    """
    cnn_model.eval()
    vae.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (x,) in val_loader:
            x = x.to(device)
            x_rec, _, _ = vae(x)
            p = cnn_model(x_rec)[1]
            correct += (p.squeeze() >= 0.5).float().sum().item()
            total += x.size(0)
    return correct / max(1, total)


# ----------------------------- Extract and save ---------------------------------

def extract_and_save_latent_and_signals(
    vae: VAE,
    dataset: TensorDataset,
    device: torch.device,
    output_file: str = "latent_and_signals.csv",
    num_samples: int = 100
) -> None:
    """
    Save a small sample of original signals, reconstructions and latent vectors.
    """
    vae.eval()
    n = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), n)

    with torch.no_grad(), open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "time_idx", "orig", "recon", "latent_dim", "latent_val"])
        for si, idx in enumerate(indices):
            x = dataset[idx][0].unsqueeze(0).to(device)
            enc = vae.encoder(x)
            mu = vae.fc_mu(enc).squeeze().cpu().numpy()
            x_rec, _, _ = vae(x)
            orig = x.squeeze().cpu().numpy()
            rec = x_rec.squeeze().cpu().numpy()
            T = max(len(orig), len(rec))
            for t in range(T):
                o = orig[t] if t < len(orig) else ""
                r = rec[t] if t < len(rec) else ""
                # store latent as a separate rows with dimension index
                writer.writerow([si, t, o, r, "", ""])
            for k, v in enumerate(mu):
                writer.writerow([si, "", "", "", k, v])

    print(f"Saved to {output_file}")


# ----------------------------- Main ---------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train VAE with adversarial CNN finetuning.")

    # Replace Demo data with true samples
    parser.add_argument("--true_csv", type=str, default="Demo_True.csv",
                        help="Path to CSV file containing 'true' samples for training the VAE "
                        "(demo filename shown; user must provide their own data).")

    parser.add_argument("--sequence_length", type=int, default=10000)
    parser.add_argument("--cnn_weights", type=str, default="./saved_models/Original_CNN.pth")
    parser.add_argument("--num_classes", type=int, default=3, help="set to original training class count")
    parser.add_argument("--manifest", type=str, default="original_task_manifest.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--out_latent_csv", type=str, default="latent_and_signals_review.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.true_csv):
        raise FileNotFoundError(
            f"{args.true_csv} not found. Provide a CSV with at least one numeric column."
        )
    df_true = pd.read_csv(args.true_csv, usecols=[0])
    X = split_into_sequences(df_true, args.sequence_length)
    if X.size == 0:
        raise RuntimeError("No sequences produced from true_csv. Check sequence_length and data length.")

    # build and load the CNN detector
    cnn = CNN(input_length=args.sequence_length, num_classes=args.num_classes).to(device)
    if os.path.exists(args.cnn_weights):
        state = torch.load(args.cnn_weights, map_location=device)
        missing, unexpected = cnn.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"Loaded with non strict. Missing keys: {len(missing)}  Unexpected keys: {len(unexpected)}")
    else:
        print(f"Warning CNN weights not found at {args.cnn_weights}. Using randomly initialized CNN.")

    # evaluate baseline on original task
    orig_loader = load_original_task_data(device, args.sequence_length, args.manifest)
    base_m, base_b = evaluate_original_task(cnn, orig_loader, device)
    print(f"Baseline. Multi {base_m:.4f}  Binary {base_b:.4f}")

    print("Start training VAE.")
    vae, final_loader = train_vae_with_adversarial_cnn(
        X, cnn, device, epochs=args.epochs, latent_dim=args.latent_dim,
        sequence_length=args.sequence_length, manifest_path=args.manifest
    )

    # final evaluation and saves
    if final_loader is not None:
        fin_m, fin_b = evaluate_original_task(cnn, final_loader, device)
        print(f"Final. Multi {fin_m:.4f}  Binary {fin_b:.4f}  "
              f"Delta Multi {fin_m - base_m:+.4f}  Delta Binary {fin_b - base_b:+.4f}")

        os.makedirs("./saved_models", exist_ok=True)
        torch.save(cnn.state_dict(), "./saved_models/Enhanced_CNN.pth")
        torch.save(vae.state_dict(), "./saved_models/VAE.pth")

        ds = TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(1))
        extract_and_save_latent_and_signals(vae, ds, device, output_file=args.out_latent_csv)
        print("Done.")
    else:
        print("Training ended without a final loader.")


if __name__ == "__main__":
    main()
