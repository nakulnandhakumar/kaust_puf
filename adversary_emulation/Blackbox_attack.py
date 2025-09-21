#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Black-box Attack on CNN Classifiers (Demo)
------------------------------------------
This script simulates a black-box adversarial attack against PUF classifiers.
- A Variational Autoencoder (VAE) is trained to generate adversarial samples.
- The CNN detector is only accessible via queries (black-box setting).
- Attack effectiveness is evaluated against both an "Original" CNN
  and an "Enhanced" CNN for robustness comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
import random
import time
from collections import defaultdict, deque
import torch.fft

# ----------------------------- CNN Detector ------------------------------------

class CNN(nn.Module):
    """PUF CNN Detector with multi-class and binary heads."""
    def __init__(self, input_length, num_classes):
        super(CNN, self).__init__()
        kernel_size, pooling_size = 100, 10
        self.conv1 = nn.Conv1d(1, 32, kernel_size=kernel_size)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(pooling_size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * ((input_length - kernel_size + 1) // pooling_size), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)
        self.fc_binary = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, return_features=False):
        x = torch.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        features = torch.relu(self.fc3(x))

        multi_class_output = self.fc4(features)
        binary_output = torch.sigmoid(self.fc_binary(features))

        if return_features:
            return multi_class_output, binary_output, features
        else:
            return multi_class_output, binary_output


# ----------------------------- VAE Generator -----------------------------------

class VAE(nn.Module):
    """Variational Autoencoder used to generate adversarial samples."""
    def __init__(self, input_length, latent_dim=128):
        super(VAE, self).__init__()
        self.input_length = input_length
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2), nn.ReLU(), nn.BatchNorm1d(128),
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
        encoded = self.encoder(x)
        mu, logvar = self.fc_mu(encoded), self.fc_logvar(encoded)
        logvar = torch.clamp(logvar, min=-10, max=10)
        z = self.reparameterize(mu, logvar)
        z_decoded = self.fc_decode(z)
        return self.decoder(z_decoded), mu, logvar


# ----------------------------- Data Preparation --------------------------------

def split_into_sequences(df, sequence_length):
    """Split long 1D series into normalized sequences of fixed length."""
    data = df.iloc[:, 0].values
    data = np.nan_to_num(data, nan=0.0)
    data_min, data_max = data.min(), data.max()
    data = 2 * (data - data_min) / (data_max - data_min) - 1
    num_sequences = len(data) // sequence_length
    return np.array([data[i * sequence_length:(i + 1) * sequence_length] for i in range(num_sequences)])


def prepare_limited_dataset(real_data_ratio=1, fake_data_ratio=0, sequence_length=10000):
    """Prepare attacker dataset with limited real samples and optional fake contamination."""
    print("Loading real data (legitimate PUF responses)...")
    df_real = pd.read_csv("Demo_True.csv")
    X_real_full = split_into_sequences(df_real, sequence_length)

    num_real_samples = int(len(X_real_full) * real_data_ratio)
    real_indices = random.sample(range(len(X_real_full)), num_real_samples)
    X_real_limited = X_real_full[real_indices]

    print(f"Selected {len(X_real_limited)} real samples ({real_data_ratio * 100}%)")
    print("Loading fake data for contamination...")

    df_fake = pd.read_csv("Demo_Fake.csv")
    X_fake_full = split_into_sequences(df_fake, sequence_length)

    num_fake_samples = int(len(X_real_limited) * fake_data_ratio)
    fake_indices = random.sample(range(len(X_fake_full)), num_fake_samples)
    X_fake_contamination = X_fake_full[fake_indices]

    print(f"Added {len(X_fake_contamination)} fake samples ({fake_data_ratio * 100}% of real data)")

    X_attacker = np.vstack([X_real_limited, X_fake_contamination])
    indices = np.random.permutation(len(X_attacker))
    return X_attacker[indices], len(X_real_limited), len(X_fake_contamination)


# ----------------------------- Black-box Query ---------------------------------

class TrueBlackBoxQuerySystem:
    """Query interface to CNN detector in a black-box setting."""
    def __init__(self, cnn_model, device):
        self.cnn_model = cnn_model
        self.device = device
        self.query_count = 0
        self.successful_queries = 0
        self.query_history = []
        self.successful_samples, self.successful_latents = [], []
        self.max_success_buffer = 200

    def query_batch(self, generated_samples):
        """Send a batch of generated samples to CNN, return predictions and success rate."""
        self.cnn_model.eval()
        with torch.no_grad():
            generated_samples = generated_samples.to(self.device)
            _, binary_output = self.cnn_model(generated_samples)
            predictions = (binary_output > 0.5).float()
            self.query_count += generated_samples.size(0)
            self.successful_queries += predictions.sum().item()
            success_rate = predictions.mean().item()
            self.query_history.append({'batch_size': generated_samples.size(0), 'success_rate': success_rate})
            return predictions, success_rate

    def store_successful_samples(self, samples, latents, predictions):
        """Cache successful adversarial samples for replay and guidance."""
        success_mask = predictions.squeeze() == 1
        if success_mask.sum() > 0:
            for sample, latent in zip(samples[success_mask], latents[success_mask]):
                self.successful_samples.append(sample.cpu().clone())
                self.successful_latents.append(latent.cpu().clone())
                if len(self.successful_samples) > self.max_success_buffer:
                    self.successful_samples.pop(0)
                    self.successful_latents.pop(0)
            print(f"Stored {success_mask.sum().item()} successful samples (total={len(self.successful_samples)})")
            return success_mask.sum().item()
        return 0

    def get_statistics(self):
        overall_success_rate = self.successful_queries / max(self.query_count, 1)
        return {
            'total_queries': self.query_count,
            'successful_queries': self.successful_queries,
            'overall_success_rate': overall_success_rate,
            'successful_samples_count': len(self.successful_samples)
        }


# ----------------------------- Loss Function -----------------------------------

def enhanced_vae_loss_with_frequency(x_reconstructed, x, mu, logvar, successful_samples=None):
    """Enhanced VAE loss with reconstruction, KL, frequency, guidance, and diversity terms."""
    device, batch_size = x.device, x.size(0)
    # Reconstruction
    reconstruction_loss = 0.7 * F.mse_loss(x_reconstructed, x) + 0.3 * F.l1_loss(x_reconstructed, x)
    # Statistical consistency
    mean_loss = std_loss = torch.tensor(0.0, device=device)
    if batch_size > 1:
        mean_loss = F.mse_loss(x_reconstructed.mean(dim=2), x.mean(dim=2).detach())
        std_loss = F.l1_loss(x_reconstructed.std(dim=2, unbiased=False), x.std(dim=2, unbiased=False).detach())
    # Frequency-domain loss
    frequency_loss = torch.tensor(0.0, device=device)
    if batch_size > 1:
        x_fft, xr_fft = torch.fft.fft(x.squeeze(1)), torch.fft.fft(x_reconstructed.squeeze(1))
        frequency_loss = F.mse_loss(torch.abs(xr_fft), torch.abs(x_fft).detach())
        phase_loss = F.mse_loss(torch.sin(torch.angle(xr_fft)), torch.sin(torch.angle(x_fft)).detach())
        frequency_loss += 0.1 * phase_loss
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    # Success-guided loss
    success_guidance_loss = torch.tensor(0.0, device=device)
    if successful_samples:
        targets = torch.stack(random.sample(successful_samples, min(5, len(successful_samples)))).to(device)
        success_guidance_loss = (F.mse_loss(x_reconstructed.mean(), targets.mean().detach()) +
                                 F.mse_loss(x_reconstructed.std(), targets.std().detach()))
    # Diversity loss
    diversity_loss = torch.tensor(0.0, device=device)
    if batch_size > 1:
        flattened = x_reconstructed.view(batch_size, -1)
        sims = torch.mm(F.normalize(flattened, dim=1), F.normalize(flattened, dim=1).t())
        diversity_loss = torch.relu(sims - 0.8).mean()
    # Weighted total
    total_loss = (reconstruction_loss + 0.001 * kl_loss +
                  0.5 * (mean_loss + std_loss) + 0.3 * frequency_loss +
                  0.4 * success_guidance_loss + 0.1 * diversity_loss)
    return total_loss


# ----------------------------- Training Loop -----------------------------------

def train_true_blackbox_vae(X_attacker, query_system, device, epochs=200, latent_dim=128, target_success_rate=0.3, total_queries=200000):
    """Train VAE adversary with black-box queries to CNN detector."""
    dataset = TensorDataset(torch.tensor(X_attacker, dtype=torch.float32).unsqueeze(1))
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    vae = VAE(input_length=X_attacker.shape[1], latent_dim=latent_dim).to(device)
    vae.apply(lambda m: nn.init.xavier_uniform_(m.weight) if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)) else None)

    optimizer = optim.Adam(vae.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 200], gamma=0.5)

    best_success_rate = 0.0
    for epoch in range(epochs):
        vae.train()
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            xr, mu, logvar = vae(x)
            loss = enhanced_vae_loss_with_frequency(xr, x, mu, logvar, query_system.successful_samples)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
        # Black-box queries
        with torch.no_grad():
            z = torch.randn(500, latent_dim).to(device)
            xr = vae.decoder(vae.fc_decode(z))
            preds, success_rate = query_system.query_batch(xr)
            query_system.store_successful_samples(xr, z, preds)
        scheduler.step()
        best_success_rate = max(best_success_rate, success_rate)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: loss={loss.item():.4f}, success_rate={success_rate:.3f}")
        if success_rate >= target_success_rate:
            break
    return vae, best_success_rate


# ----------------------------- Evaluation --------------------------------------

def evaluate_attack_effectiveness(vae, cnn_model, device, num_test_samples=500):
    """Evaluate success rate of adversarial VAE against CNN detector."""
    vae.eval()
    with torch.no_grad():
        z = torch.randn(num_test_samples, 128).to(device)
        xr = vae.decoder(vae.fc_decode(z))
        _, binary_output = cnn_model(xr)
        preds = (binary_output > 0.5).float()
    return preds.mean().item()


def test_single_model(model_path, X_attacker, device, model_name="CNN", epochs=200):
    """Run attack against one CNN model (original or enhanced)."""
    print(f"\nTesting {model_name} CNN: {model_path}")
    cnn_model = CNN(input_length=X_attacker.shape[1], num_classes=17).to(device)
    cnn_model.load_state_dict(torch.load(model_path, map_location=device))
    cnn_model.eval()
    query_system = TrueBlackBoxQuerySystem(cnn_model, device)
    vae, best_success = train_true_blackbox_vae(X_attacker, query_system, device, epochs=epochs)
    final_success = evaluate_attack_effectiveness(vae, cnn_model, device)
    return {'model_name': model_name, 'best_success': best_success, 'final_success': final_success}


# ----------------------------- Main --------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Prepare demo attacker dataset
    X_attacker, num_real, num_fake = prepare_limited_dataset(real_data_ratio=1, fake_data_ratio=0)
    print(f"Prepared attacker dataset: {num_real} real, {num_fake} fake samples")

    # Paths to demo CNN models
    original_cnn_path, enhanced_cnn_path = "Original_model_path.pth", "Enhanced_model_path.pth"

    # Attack both models
    results_original = test_single_model(original_cnn_path, X_attacker, device, model_name="Original")
    results_enhanced = test_single_model(enhanced_cnn_path, X_attacker, device, model_name="Enhanced")

    print("\nAttack Results Summary:")
    print(results_original)
    print(results_enhanced)
