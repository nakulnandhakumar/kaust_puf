#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gray-box Attack on CNN Classifiers (Demo)
-----------------------------------------
This script implements a gray-box adversarial attack where the attacker
receives a quantized/noisy confidence value (not full logits) from the target CNN.
A VAE is trained to generate adversarial samples guided by these quantized confidences.

Key components:
- CNN detector (multi-class + binary head) -- used as the target.
- VAE generator -- trained to produce samples that achieve high reported confidence.
- GreyBoxConfidenceQuerySystem -- returns quantized/noisy confidences to the attacker.
- Training loop: VAE guided by reconstruction + confidence feedback + diversity terms.
"""

import os
import csv
import time
import random
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# ----------------------------- CNN Detector ------------------------------------

class CNN(nn.Module):
    """
    PUF CNN Detector with two heads:
      - multi-class head (logits)
      - binary head (sigmoid probability)
    Note: return_features / return_logits options allow flexible queries.
    """
    def __init__(self, input_length, num_classes):
        super().__init__()
        kernel_size = 100
        pooling_size = 10
        self.conv1 = nn.Conv1d(1, 32, kernel_size=kernel_size)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(pooling_size)
        # compute flattened feature length after conv+pool
        flat_len = 32 * ((input_length - kernel_size + 1) // pooling_size)
        self.fc1 = nn.Linear(flat_len, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc_multi = nn.Linear(32, num_classes)
        self.fc_binary = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, return_features=False, return_logits=False):
        """
        Forward pass.
        - If return_features True: also return internal features.
        - If return_logits True: return binary logits (pre-sigmoid) as third output when requested.
        """
        x = torch.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        features = torch.relu(self.fc3(x))

        multi_out = self.fc_multi(features)     # logits for multi-class
        bin_logits = self.fc_binary(features).squeeze(1)  # scalar logit per example
        bin_prob = torch.sigmoid(bin_logits)

        if return_features and return_logits:
            return multi_out, bin_prob, features, bin_logits
        if return_features:
            return multi_out, bin_prob, features
        if return_logits:
            return multi_out, bin_prob, bin_logits
        return multi_out, bin_prob


# ----------------------------- VAE Generator -----------------------------------

class VAE(nn.Module):
    """
    Small 1D VAE: encoder -> latent -> decoder.
    Designed to generate 1D sequences same length as input.
    """
    def __init__(self, input_length, latent_dim=128):
        super().__init__()
        self.input_length = input_length
        # encoder conv stack
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Flatten()
        )
        enc_out_len = input_length // 8  # 3 strides of 2
        enc_out_size = 128 * enc_out_len

        self.fc_mu = nn.Sequential(nn.Linear(enc_out_size, latent_dim), nn.Dropout(0.3))
        self.fc_logvar = nn.Sequential(nn.Linear(enc_out_size, latent_dim), nn.Dropout(0.3))

        # decoder
        self.fc_decode = nn.Linear(latent_dim, enc_out_size)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, enc_out_len)),
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(), nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(), nn.BatchNorm1d(32),
            nn.ConvTranspose1d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) + 1e-6
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc = self.encoder(x)
        mu = self.fc_mu(enc)
        logvar = self.fc_logvar(enc)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        z = self.reparameterize(mu, logvar)
        dec = self.fc_decode(z)
        x_rec = self.decoder(dec)
        return x_rec, mu, logvar


# ----------------------------- Data preparation ---------------------------------

def split_into_sequences(df, sequence_length):
    """
    Slice first column into non-overlapping sequences and normalize to [-1,1].
    Input: pandas DataFrame with numeric first column.
    """
    arr = df.iloc[:, 0].to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0)
    mn, mx = arr.min(), arr.max()
    eps = 1e-12
    arr_norm = 2.0 * (arr - mn) / (mx - mn + eps) - 1.0
    num = len(arr_norm) // sequence_length
    seqs = np.array([arr_norm[i*sequence_length:(i+1)*sequence_length] for i in range(num)], dtype=np.float32)
    return seqs  # shape (N, sequence_length)


def prepare_limited_dataset(real_data_ratio=0.2, fake_data_ratio=0.15, sequence_length=10000):
    """
    Prepare attacker dataset from limited real samples + optional fake contamination.
    - real_data_ratio: fraction of available real sequences to sample (0..1).
    - fake_data_ratio: fraction relative to sampled real used as contamination.
    Uses demo file names; replace with real paths when running experiments.
    """
    print("Loading real data (demo placeholder)...")
    df_real = pd.read_csv("Demo_True.csv")  # DEMO placeholder
    X_real_full = split_into_sequences(df_real, sequence_length)
    num_real_samples = max(1, int(len(X_real_full) * real_data_ratio))
    real_indices = random.sample(range(len(X_real_full)), num_real_samples)
    X_real_limited = X_real_full[real_indices]
    print(f"Selected {len(X_real_limited)} real samples (ratio={real_data_ratio})")

    print("Loading fake data (demo placeholder)...")
    df_fake = pd.read_csv("Demo_Fake.csv")  # DEMO placeholder
    X_fake_full = split_into_sequences(df_fake, sequence_length)
    num_fake = int(len(X_real_limited) * fake_data_ratio)
    num_fake = min(num_fake, len(X_fake_full))
    fake_indices = random.sample(range(len(X_fake_full)), num_fake) if num_fake > 0 else []
    X_fake_contamination = X_fake_full[fake_indices] if len(fake_indices) > 0 else np.empty((0, sequence_length), dtype=np.float32)
    print(f"Added {len(X_fake_contamination)} fake samples (contamination={fake_data_ratio})")

    X_attacker = np.vstack([X_real_limited, X_fake_contamination]) if X_fake_contamination.size else X_real_limited
    indices = np.random.permutation(len(X_attacker))
    X_attacker = X_attacker[indices]
    return X_attacker, len(X_real_limited), len(X_fake_contamination)


# --------------------- Grey-box query (quantized noisy confidence) ----------------

class GreyBoxConfidenceQuerySystem:
    """
    Grey-box query simulator.
    - Given generated samples, returns quantized noisy confidences in [0,1] (e.g., steps of 0.1).
    - Also returns a binary "success" mask based on a confidence threshold.
    - Stores successful samples/latents for replay.
    """
    def __init__(self, cnn_model, device, noise_std=0.05, quant_step=0.1, confidence_threshold=0.5, max_buffer=200):
        self.cnn_model = cnn_model
        self.device = device
        self.noise_std = noise_std
        self.quant_step = quant_step
        self.confidence_threshold = confidence_threshold
        self.query_count = 0
        self.successful_queries = 0
        self.query_history = []
        self.successful_samples = []
        self.successful_latents = []
        self.successful_confidences = []
        self.max_buffer = max_buffer

    def query_batch(self, generated_samples):
        """
        Query cnn with a batch of generated samples.
        Returns:
          quantized_confidences (Tensor CPU) -- shape (B,)
          predictions (Tensor CPU binary) -- 1 if quantized_conf >= threshold else 0
          success_rate (float)
        """
        self.cnn_model.eval()
        with torch.no_grad():
            x = generated_samples.to(self.device)
            _, bin_prob = self.cnn_model(x)  # shape (B,)
            # add small gaussian noise to simulate measurement uncertainty
            noisy = bin_prob + torch.randn_like(bin_prob) * self.noise_std
            noisy = torch.clamp(noisy, 0.0, 1.0)
            # quantize to steps (e.g., 0.0, 0.1, ..., 1.0)
            q = torch.round(noisy / self.quant_step) * self.quant_step
            q_cpu = q.detach().cpu()
            preds = (q_cpu >= self.confidence_threshold).float()
        batch_size = generated_samples.size(0)
        self.query_count += batch_size
        self.successful_queries += int(preds.sum().item())
        success_rate = float(preds.mean().item())
        self.query_history.append({'batch': len(self.query_history)+1, 'size': batch_size, 'rate': success_rate})
        return q_cpu, preds, success_rate

    def store_successful_samples(self, samples, latents, confidences):
        """
        Keep successful samples (based on returned quantized confidences).
        samples: tensor on CPU (B,1,L)
        latents: tensor on CPU (B, latent_dim)
        confidences: tensor on CPU (B,)
        """
        mask = (confidences >= self.confidence_threshold)
        if mask.sum().item() == 0:
            return 0
        for s, z, c in zip(samples[mask], latents[mask], confidences[mask]):
            self.successful_samples.append(s.clone())
            self.successful_latents.append(z.clone())
            self.successful_confidences.append(c.clone())
            if len(self.successful_samples) > self.max_buffer:
                self.successful_samples.pop(0)
                self.successful_latents.pop(0)
                self.successful_confidences.pop(0)
        return int(mask.sum().item())

    def get_statistics(self):
        overall_success = self.successful_queries / max(1, self.query_count)
        return {
            'total_queries': self.query_count,
            'successful_queries': self.successful_queries,
            'overall_success_rate': overall_success,
            'successful_samples_count': len(self.successful_samples),
            'confidence_threshold': self.confidence_threshold
        }


# ----------------------------- Loss: confidence-guided VAE -----------------------

def confidence_guided_vae_loss(x_rec, x, mu, logvar, confidences=None, successful_samples=None):
    """
    VAE loss combining:
      - reconstruction (MSE + L1)
      - KL divergence
      - statistical (mean/std) matching
      - frequency-domain magnitude matching
      - success-guidance (match stats of successful examples)
      - diversity penalty (penalize too-similar reconstructions)
    confidences: optionally the quantized confidences returned by query system (cpu Tensor)
    """
    device = x.device
    batch = x.size(0)

    # reconstruction
    rec = 0.7 * F.mse_loss(x_rec, x, reduction='mean') + 0.3 * F.l1_loss(x_rec, x, reduction='mean')

    # mean/std matching
    mean_loss = std_loss = torch.tensor(0.0, device=device)
    if batch > 1:
        mean_loss = F.mse_loss(x_rec.mean(dim=2), x.mean(dim=2).detach())
        std_loss = F.l1_loss(x_rec.std(dim=2, unbiased=False), x.std(dim=2, unbiased=False).detach())

    # frequency domain magnitude matching
    freq_loss = torch.tensor(0.0, device=device)
    if batch > 1:
        xr_fft = torch.fft.fft(x_rec.squeeze(1))
        x_fft = torch.fft.fft(x.squeeze(1)).detach()
        freq_loss = F.mse_loss(torch.abs(xr_fft), torch.abs(x_fft))

    # KL
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / max(1, batch)

    # success guidance: match mean/std of cached successful samples if available
    succ_loss = torch.tensor(0.0, device=device)
    if successful_samples and len(successful_samples) > 0:
        k = min(5, len(successful_samples))
        # stack few successful samples and compute simple mean-statistics match
        idxs = random.sample(range(len(successful_samples)), k)
        tgt = torch.stack([successful_samples[i] for i in idxs]).to(device)  # (k,1,L)
        tgt_mean = tgt.mean(dim=[0,2])  # mean over batch and time
        xr_mean = x_rec.mean(dim=[0,2])
        succ_loss = F.mse_loss(xr_mean, tgt_mean.detach())

    # diversity penalty
    div = torch.tensor(0.0, device=device)
    if batch > 1:
        flat = x_rec.view(batch, -1)
        sims = torch.mm(F.normalize(flat, dim=1), F.normalize(flat, dim=1).t())
        div = torch.relu(sims - 0.85).mean()

    # If confidences provided, upweight loss from low-confidence items (encourage stronger changes)
    conf_term = torch.tensor(0.0, device=device)
    if confidences is not None:
        # confidences should be cpu Tensor; move to device and align with batch length
        c = confidences.to(device)
        if c.numel() == batch:
            # smaller confidence -> larger weight
            weights = (1.0 - c) * 2.0 + 0.5
            weighted_rec = (weights * ((x_rec - x)**2).mean(dim=[1,2])).mean()
            conf_term = 1.0 * weighted_rec
        else:
            conf_term = 0.0

    total = rec + 1e-3 * kl + 0.3 * (mean_loss + std_loss) + 0.2 * freq_loss + 0.3 * succ_loss + 0.1 * div + conf_term
    return total, rec.item(), kl.item(), mean_loss.item(), std_loss.item(), freq_loss.item(), succ_loss.item(), div.item(), float(conf_term)


# ----------------------------- Training (gray-box) --------------------------------

def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def train_greybox_confidence_vae(X_attacker, query_system, device, epochs=50, latent_dim=128,
                                 target_success_rate=0.3):
    """
    Train VAE with confidence-guided loss and periodic querying of the grey-box oracle.
    Returns: trained VAE, training_history list, final statistics from query_system.
    """
    dataset = TensorDataset(torch.tensor(X_attacker, dtype=torch.float32).unsqueeze(1))
    train_size = int(0.8 * len(dataset))
    train_dataset, _ = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    input_len = X_attacker.shape[1]
    vae = VAE(input_length=input_len, latent_dim=latent_dim).to(device)
    vae.apply(init_weights)

    optimizer = optim.Adam(vae.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 120], gamma=0.7)

    training_history = []
    best_dyn = best_strict = best_avg_conf = 0.0

    queries_per_epoch = 100
    queries_per_batch = 25

    for epoch in range(epochs):
        vae.train()
        total_loss = 0.0
        for batch_idx, (x_batch,) in enumerate(train_loader):
            x = x_batch.to(device)
            # occasional replay from cached successful samples to stabilize
            if len(query_system.successful_samples) > 8:
                n_replay = min(8, len(query_system.successful_samples), x.size(0)//2)
                if n_replay > 0:
                    inds = random.sample(range(len(query_system.successful_samples)), n_replay)
                    replay = torch.stack([query_system.successful_samples[i] for i in inds]).to(device)
                    x = torch.cat([x[:x.size(0)-n_replay], replay], dim=0)

            optimizer.zero_grad()
            x_rec, mu, logvar = vae(x)

            # For first batch of epoch, query the oracle on reconstructions to obtain confidences
            if batch_idx == 0:
                with torch.no_grad():
                    try:
                        conf_q, preds, _ = query_system.query_batch(x_rec.detach())
                    except Exception:
                        conf_q = None
            else:
                conf_q = None

            loss_items = confidence_guided_vae_loss(
                x_rec, x, mu, logvar,
                confidences=conf_q,
                successful_samples=query_system.successful_samples
            )
            loss = loss_items[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))

        # after training epoch, perform a set of black-box queries with generated samples
        vae.eval()
        dyn_count = 0
        strict_count = 0
        sum_conf = 0.0
        num_batches = max(1, queries_per_epoch // queries_per_batch)
        with torch.no_grad():
            for _b in range(num_batches):
                # sample latents: mix random and successful-latent-based proposals
                z = torch.randn(queries_per_batch, latent_dim, device=device)
                if len(query_system.successful_latents) > 0:
                    m = min(queries_per_batch // 3, len(query_system.successful_latents))
                    idxs = random.sample(range(len(query_system.successful_latents)), m)
                    base_z = torch.stack([query_system.successful_latents[i] for i in idxs]).to(device)
                    noise = torch.randn_like(base_z) * random.choice([0.05, 0.1, 0.2])
                    z[:m] = base_z + noise

                gen = vae.decoder(vae.fc_decode(z))
                conf_batch, pred_batch, _ = query_system.query_batch(gen)
                dyn_count += int(pred_batch.sum().item())
                strict_count += int((conf_batch >= query_system.confidence_threshold).sum().item())
                sum_conf += float(conf_batch.sum().item())

                # store successful (quantized) samples for replay/guidance
                try:
                    query_system.store_successful_samples(gen.detach().cpu(), z.detach().cpu(), conf_batch)
                except Exception:
                    pass

        dyn_rate = dyn_count / queries_per_epoch
        strict_rate = strict_count / queries_per_epoch
        avg_conf = sum_conf / queries_per_epoch

        scheduler.step()
        training_history.append({
            'epoch': epoch+1,
            'avg_train_loss': avg_loss,
            'dyn_rate': dyn_rate,
            'strict_rate': strict_rate,
            'avg_confidence': avg_conf,
            'total_queries': query_system.query_count
        })

        best_dyn = max(best_dyn, dyn_rate)
        best_strict = max(best_strict, strict_rate)
        best_avg_conf = max(best_avg_conf, avg_conf)

        print(f"Epoch {epoch+1:3d} | Loss {avg_loss:.4f} | Dyn {dyn_rate:.3f} | Strict {strict_rate:.3f} | AvgConf {avg_conf:.3f} | Queries {query_system.query_count}")

        if strict_rate >= target_success_rate and epoch >= 10:
            print("Target strict success reached, stopping early.")
            break

    final_stats = query_system.get_statistics()
    print("Training finished. Final stats:", final_stats)
    return vae, training_history, final_stats


# ----------------------------- Utilities & Main --------------------------------

def test_greybox_model(model_path, X_attacker, device, model_name="CNN"):
    """
    High-level runner:
      - load model_path into CNN
      - create GreyBoxConfidenceQuerySystem
      - train greybox VAE
      - return stats & history
    """
    print(f"\n=== Testing {model_name} (Grey-box) ===")
    seq_len = X_attacker.shape[1]
    cnn_model = CNN(input_length=seq_len, num_classes=17).to(device)
    # load checkpoint; accept raw state_dict or checkpoint dicts
    ck = torch.load(model_path, map_location=device)
    if isinstance(ck, dict):
        if 'model_state_dict' in ck:
            cnn_model.load_state_dict(ck['model_state_dict'])
        elif 'state_dict' in ck:
            cnn_model.load_state_dict(ck['state_dict'])
        else:
            cnn_model.load_state_dict(ck)
    else:
        cnn_model.load_state_dict(ck)
    cnn_model.eval()

    with torch.no_grad():
        sample_probe = torch.tensor(X_attacker[:4], dtype=torch.float32).unsqueeze(1).to(device)
        _, conf = cnn_model(sample_probe)
        print("Probe confidences (demo):", [float(c) for c in conf.squeeze().tolist()])

    query_system = GreyBoxConfidenceQuerySystem(cnn_model, device, noise_std=0.05, quant_step=0.1, confidence_threshold=0.5, max_buffer=200)
    vae, history, stats = train_greybox_confidence_vae(X_attacker, query_system, device, epochs=50, latent_dim=128, target_success_rate=0.3)
    return stats, history


def save_greybox_results(original_stats, enhanced_stats, filename='greybox_results_demo.csv'):
    """Save summary of greybox attack results to CSV (simple summary)."""
    rows = [
        ['model', 'total_queries', 'success_rate', 'successful_samples'],
        ['original', original_stats.get('total_queries', 0), original_stats.get('overall_success_rate', 0.0), original_stats.get('successful_samples_count', 0)],
        ['enhanced', enhanced_stats.get('total_queries', 0), enhanced_stats.get('overall_success_rate', 0.0), enhanced_stats.get('successful_samples_count', 0)]
    ]
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Saved greybox summary to {filename}")


if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Prepare demo attacker dataset (only demo file names used)
    X_attacker, n_real, n_fake = prepare_limited_dataset(real_data_ratio=0.5, fake_data_ratio=0.1, sequence_length=10000)
    print(f"Attacker dataset prepared: {n_real} real samples, {n_fake} fake samples (total {len(X_attacker)})")

    # Demo model paths (replace these with your trained model files)
    original_model_path = "Original_model_path.pth"
    enhanced_model_path = "Enhanced_model_path.pth"

    # Test enhanced and original models (order does not matter)
    enhanced_stats, enhanced_history = test_greybox_model(enhanced_model_path, X_attacker, device, model_name="Enhanced")
    original_stats, original_history = test_greybox_model(original_model_path, X_attacker, device, model_name="Original")

    # Save simple summary
    save_greybox_results(original_stats, enhanced_stats, filename='greybox_results_demo.csv')

    print("Done. See greybox_results_demo.csv for a short summary.")
