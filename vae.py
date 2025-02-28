#!/ibex/user/nandhan/mambaforge/bin/python3

'''
Header: vae.py
The code snippet below is a Variational Autoencoder (VAE) model that is used to generate synthetic data
replicating chaotic 1D laser signals input to the model. The VAE will serve as the
basis for a generator in a Generative adversarial training framework model meant to generate
data for adversial training of the puf_classifier model.
'''

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
import umap
import seaborn as sns

# ----------------------------- CNN_Detector Architecture --------------------------------

# Detector: 1D_CNN
class CNN(nn.Module):
    def __init__(self, input_length, num_classes):
        super(CNN, self).__init__()
        kernel_size = 100
        pooling_size = 10
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
        features = torch.relu(self.fc3(x))  # Feature layer

        multi_class_output = self.fc4(features)
        binary_output = torch.sigmoid(self.fc_binary(features))

        if return_features:
            return multi_class_output, binary_output, features
        else:
            return multi_class_output, binary_output


# ----------------------------- VAE Architecture --------------------------------
# Define the Variational Autoencoder (VAE) model
class VAE(nn.Module):
    def __init__(self, input_length, latent_dim=128):
        super(VAE, self).__init__()
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
        enc_out_length = input_length // (2 * 2 * 2)
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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) + 1e-6 
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)

        # cut logvar
        logvar = torch.clamp(logvar, min=-10, max=10)

        z = self.reparameterize(mu, logvar)
        z_decoded = self.fc_decode(z)
        x_reconstructed = self.decoder(z_decoded)
        return x_reconstructed, mu, logvar

# ----------------------------- VAE Data Preparation --------------------------------

# Helper function to scale and process "real" data into sequences of size 10000
def split_into_sequences(df, sequence_length):
    data = df.iloc[:, 0].values
    data = np.nan_to_num(data, nan=0.0)

    # Scaling data to be between -1 and 1
    data_min = data.min()
    data_max = data.max()
    data = 2 * (data - data_min) / (data_max - data_min) - 1

    # Splitting into sequences
    num_sequences = len(data) // sequence_length
    sequences = np.array([data[i * sequence_length:(i + 1) * sequence_length] for i in range(num_sequences)])

    return sequences


# ----------------------------- Classification Feedback + KL Loss --------------------------------

def combined_loss(cnn_model, x_reconstructed, x, mu, logvar, lambda_feedback=0.0, lambda_kl=0.0, alpha=1.2):
    # Reconstruction Loss (MSE + L1)
    reconstruction_loss = 0.7 * F.mse_loss(x_reconstructed, x, reduction='mean') + 0.3 * F.l1_loss(x_reconstructed, x, reduction='mean')

    # Dynamic adjust mean and mu
    try:
        if x.size(0) > 1:  
            target_mean = x.mean(dim=2).detach()
            generated_mean = x_reconstructed.mean(dim=2)
            mean_loss = F.mse_loss(generated_mean, target_mean)

            target_std = x.std(dim=2, unbiased=False).detach()
            generated_std = x_reconstructed.std(dim=2, unbiased=False)
            std_loss = F.l1_loss(generated_std, target_std)
        else:
            mean_loss = torch.tensor(0.0, device=x.device)
            std_loss = torch.tensor(0.0, device=x.device)
    except Exception as e:
        print(f"Warning: Mean/Std computation failed. Error: {e}")
        mean_loss = torch.tensor(0.0, device=x.device)
        std_loss = torch.tensor(0.0, device=x.device)

    # Feature matching loss
    cnn_model.eval()
    with torch.no_grad():
        _, _, features_real = cnn_model(x, return_features=True)
    _, _, features_fake = cnn_model(x_reconstructed, return_features=True)

    # 特征归一化和匹配损失
    features_real_norm = F.normalize(features_real, dim=1)
    features_fake_norm = F.normalize(features_fake, dim=1)
    feature_loss = 0.7 * F.mse_loss(features_fake_norm, features_real_norm, reduction='mean') + 0.3 * F.l1_loss(features_fake_norm, features_real_norm, reduction='mean')

    # KL Divergence Loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    # Total Loss
    total_loss = alpha * reconstruction_loss + lambda_feedback * feature_loss + lambda_kl * kl_loss + std_loss + mean_loss

    return total_loss, reconstruction_loss.item(), feature_loss.item(), kl_loss.item(), mean_loss.item(), std_loss.item()


# ----------------------------- VAE Init --------------------------------
def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# ----------------------------- VAE Training --------------------------------

def train_vae_with_adversarial_cnn(X, cnn_model, device, epochs=250, latent_dim=128):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(1))

    # Split train/val
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    input_length = X.shape[1]
    vae = VAE(input_length=input_length, latent_dim=latent_dim).to(device)
    vae.apply(init_weights)

    vae_optimizer = optim.Adam(vae.parameters(), lr=1e-4, weight_decay=1e-5)
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler_vae = optim.lr_scheduler.ReduceLROnPlateau(vae_optimizer, mode="max", factor=0.5, patience=10)
    scheduler_cnn = optim.lr_scheduler.ReduceLROnPlateau(cnn_optimizer, mode="max", factor=0.1, patience=5)

    feedback_start_epoch = 10
    final_lambda_feedback = 0.2
    initial_lambda_feedback = 0.01
    kl_target = latent_dim / 2 + 5
    kl_start_epoch = 10
    max_lambda_kl = 0.001

    patience = 3
    val_acc_threshold = 0.98
    early_stop_counter = 0

    lambda_kl_initial = 1e-6

    for epoch in range(epochs):
        if epoch < feedback_start_epoch:
            lambda_feedback = 0.01
        else:
            progress = (epoch - feedback_start_epoch) / (epochs - feedback_start_epoch)
            lambda_feedback = initial_lambda_feedback + progress * (final_lambda_feedback - initial_lambda_feedback)

        if epoch < kl_start_epoch:
            lambda_kl = lambda_kl_initial
        else:
            progress = (epoch - kl_start_epoch) / (epochs - kl_start_epoch)
            lambda_kl = lambda_kl_initial + progress * (max_lambda_kl - lambda_kl_initial)

        vae.train()
        cnn_model.train()
        total_train_loss = 0
        total_train_recon = 0
        total_train_class = 0
        total_train_kl = 0
        total_train_mean = 0
        total_train_std = 0

        for batch in train_loader:
            x = batch[0].to(device)

            # Train VAE
            vae_optimizer.zero_grad()
            x_reconstructed, mu, logvar = vae(x)
            vae_loss, recon_loss_val, class_loss_val, kl_loss_val, mean_loss_val, std_loss_val = combined_loss(
                cnn_model, x_reconstructed, x, mu, logvar,
                lambda_feedback=lambda_feedback,
                lambda_kl=lambda_kl
            )
            vae_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5.0)
            vae_optimizer.step()

            total_train_loss += vae_loss.item()
            total_train_recon += recon_loss_val
            total_train_class += class_loss_val
            total_train_kl += kl_loss_val
            total_train_mean += mean_loss_val
            total_train_std += std_loss_val

            # CNN training set generator
            real_data = x
            fake_data = x_reconstructed.detach()

            cnn_input = torch.cat([real_data, fake_data], dim=0)
            cnn_labels = torch.cat([
                torch.ones(real_data.size(0), 1, device=device), 
                torch.zeros(fake_data.size(0), 1, device=device)
            ], dim=0)

            # Train CNN
            cnn_optimizer.zero_grad()
            _, binary_output = cnn_model(cnn_input)
            cnn_loss = F.binary_cross_entropy(binary_output, cnn_labels)
            cnn_loss.backward()
            torch.nn.utils.clip_grad_norm_(cnn_model.parameters(), max_norm=5.0)
            cnn_optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_recon = total_train_recon / len(train_loader)
        avg_train_class = total_train_class / len(train_loader)
        avg_train_kl = total_train_kl / len(train_loader)
        avg_train_mean = total_train_mean / len(train_loader)
        avg_train_std = total_train_std / len(train_loader)

        vae.eval()
        cnn_model.eval()
        total_val_loss = 0
        gen_mean_total = 0  
        real_mean_total = 0 
        gen_std_total = 0  
        real_std_total = 0  
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                x_reconstructed, mu, logvar = vae(x)
                val_loss, _, _, _, mean_loss_val, std_loss_val = combined_loss(
                    cnn_model, x_reconstructed, x, mu, logvar,
                    lambda_feedback=lambda_feedback,
                    lambda_kl=lambda_kl
                )
                total_val_loss += val_loss.item()

                gen_mean_total += x_reconstructed.mean().item()
                real_mean_total += x.mean().item()
                gen_std_total += x_reconstructed.std().item()
                real_std_total += x.std().item()
                num_batches += 1

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = evaluate_with_cnn(cnn_model, vae, val_loader, device)
        scheduler_vae.step(val_accuracy)
        scheduler_cnn.step(val_accuracy)
        current_lr = vae_optimizer.param_groups[0]['lr']

        avg_gen_mean = gen_mean_total / num_batches
        avg_real_mean = real_mean_total / num_batches
        avg_gen_std = gen_std_total / num_batches
        avg_real_std = real_std_total / num_batches

        # Early stopping logic
        if val_accuracy >= val_acc_threshold and avg_train_kl < kl_target:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping criteria met. Training stopped.")
                break
        else:
            early_stop_counter = 0

        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Recon: {avg_train_recon:.4f}, Class: {avg_train_class:.4f}, KL: {avg_train_kl:.4f}, Mean Loss: {avg_train_mean:.4f}, Std Loss: {avg_train_std:.4f} "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
              f"Generated Mean: {avg_gen_mean:.4f}, Real Mean: {avg_real_mean:.4f}, "
              f"Generated Std: {avg_gen_std:.4f}, Real Std: {avg_real_std:.4f}, "
              f"Lambda_feedback: {lambda_feedback:.4f}, Lambda_kl: {lambda_kl:.4f}, "
              f"Current LR: {current_lr:.10f}, CNN Loss: {cnn_loss.item():.4f}")

    return vae

# --------------------- Validate through CNN_classifier ------------------------
def evaluate_with_cnn(cnn_model, vae, val_loader, device):
    cnn_model.eval()
    vae.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            x_reconstructed, _, _ = vae(x)
            outputs = cnn_model(x_reconstructed)[1]
            predictions = torch.sigmoid(outputs).cpu().numpy()
            correct += (predictions > 0.5).sum()
            total += x.size(0)

    accuracy = correct / total
    return accuracy

# ----------------------------- Plotting Function --------------------------------

def plot_original_and_generated(vae, dataset, device):
    vae.eval()
    with torch.no_grad():
        random_indices = random.sample(range(len(dataset)), 5)
        plt.figure(figsize=(10, 15))
        for i, idx in enumerate(random_indices):
            original_sample = dataset[idx][0].unsqueeze(0).to(device)
            generated_sample, _, _ = vae(original_sample)
            original_sample = original_sample.squeeze().cpu().numpy()
            generated_sample = generated_sample.squeeze().cpu().numpy()
            plt.subplot(5, 2, 2 * i + 1)
            plt.plot(original_sample, color='blue', linewidth=1)
            plt.ylim(-1, 1)
            plt.title(f'Original Sample {i + 1}')
            plt.subplot(5, 2, 2 * i + 2)
            plt.plot(generated_sample, color='green', linewidth=1)
            plt.ylim(-1, 1)
            plt.title(f'Generated Sample {i + 1}')
        plt.tight_layout()
        plt.show()


# ----------------------------- Extract Signals --------------------------------

def extract_and_save_latent_and_signals(vae, dataset, device, output_file='latent_and_signals.csv', num_samples=100):
    """
    Extract latent space representations, original signals, and reconstructed signals for a random set of samples,
    and save them to a CSV file.

    Args:
        vae: Trained VAE model.
        dataset: TensorDataset containing the input sequences.
        device: The device (CPU or CUDA) to run the computations on.
        output_file: Name of the output CSV file.
        num_samples: Number of random samples to extract.
    """
    vae.eval()

    # Randomly select num_samples indices
    indices = random.sample(range(len(dataset)), num_samples)

    with torch.no_grad():
        # Prepare lists to store data
        all_original_signals = []
        all_reconstructed_signals = []
        all_latent_vectors = []

        for idx in indices:
            # Get the original sample
            original_sample = dataset[idx][0].unsqueeze(0).to(device)  # Shape: (1, 1, sequence_length)

            # Pass the original sample through the VAE to get the latent vector and reconstructed sample
            encoded = vae.encoder(original_sample)
            mu = vae.fc_mu(encoded)
            latent_vector = mu.squeeze().cpu().numpy()

            reconstructed_sample, _, _ = vae(original_sample)

            # Convert tensors to numpy arrays
            original_signal = original_sample.squeeze().cpu().numpy()
            reconstructed_signal = reconstructed_sample.squeeze().cpu().numpy()

            # Store data
            all_original_signals.append(original_signal)
            all_reconstructed_signals.append(reconstructed_signal)
            all_latent_vectors.extend(latent_vector)  # Flatten and extend the latent vectors list

        # Save to CSV
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Write headers
            writer.writerow(['Sample_Index', 'Data_Point', 'Original_Signal', 'Reconstructed_Signal', 'Latent_Vector'])

            # Write data rows for each sample
            for sample_idx, (original_signal, reconstructed_signal) in enumerate(
                    zip(all_original_signals, all_reconstructed_signals)):
                max_length = max(len(original_signal), len(reconstructed_signal))

                for i in range(max_length):
                    orig_val = original_signal[i] if i < len(original_signal) else ''
                    recon_val = reconstructed_signal[i] if i < len(reconstructed_signal) else ''
                    latent_val = all_latent_vectors[sample_idx * num_samples + i] if sample_idx * num_samples + i < len(
                        all_latent_vectors) else ''

                    writer.writerow([sample_idx, i, orig_val, recon_val, latent_val])

    print(f"{num_samples} latent vectors, original signals, and reconstructed signals saved to {output_file}")

def plot_umap_distribution(vae, dataset, device, num_samples=100, save_path=None):
    vae.eval()
    all_original_signals = []
    all_generated_signals = []

    indices = random.sample(range(len(dataset)), num_samples)
    with torch.no_grad():
        for idx in indices:
            original_sample = dataset[idx][0].unsqueeze(0).to(device)  # (1, 1, sequence_length)
            generated_sample, _, _ = vae(original_sample)

            all_original_signals.append(original_sample.squeeze().cpu().numpy())
            all_generated_signals.append(generated_sample.squeeze().cpu().numpy())

    all_original_signals = np.array(all_original_signals)
    all_generated_signals = np.array(all_generated_signals)

    original_flattened = all_original_signals.reshape(num_samples, -1)
    generated_flattened = all_generated_signals.reshape(num_samples, -1)

    combined_data = np.vstack((original_flattened, generated_flattened))
    labels = ['Original'] * num_samples + ['Generated'] * num_samples

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(combined_data)

    df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    df['Type'] = labels

    if save_path:
        df.to_csv(save_path, index=False)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x='UMAP1',
        y='UMAP2',
        hue='Type',
        palette={'Original': 'blue', 'Generated': 'green'},
        alpha=0.7,
        s=60
    )
    plt.title('UMAP Distribution of Original and Generated Signals', fontsize=16)
    plt.xlabel('UMAP1', fontsize=14)
    plt.ylabel('UMAP2', fontsize=14)
    plt.legend(title='Signal Type', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------------------------- MAIN --------------------------------------------

if __name__ == '__main__':
    # Check for GPU support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CSV files into DataFrames and combine them into one dataframe
    df1 = pd.read_csv('Ture_1.csv', usecols=[4])
    df2 = pd.read_csv('Ture_2.csv', usecols=[4])
    df3 = pd.read_csv('Ture_3.csv', usecols=[4])
    df4 = pd.read_csv('Ture_4.csv', usecols=[4])

     # Define sequence length
    sequence_length = 10000

    X1 = split_into_sequences(d1, sequence_length)
    X2 = split_into_sequences(d2, sequence_length)
    X3 = split_into_sequences(d3, sequence_length)
    X4 = split_into_sequences(d4, sequence_length)


    # Split the dataframe into scaled sequences of size sequence_length
    X = np.vstack([X1, X2, X3, X4])

    # Load CNN-classifier weights
    cnn_model = CNN(input_length=sequence_length, num_classes=17).to(device)
    cnn_model.load_state_dict(torch.load('/saved_models/puf_classifier_save_model.pth', map_location=device))
    cnn_model.eval()

    # Train VAE with new loss function
    vae = train_vae_with_adversarial_cnn(X, cnn_model, device, epochs=100, latent_dim=128)

    # Save VAE model
    torch.save(vae.state_dict(), '/saved_models/vae.pth')

    # Draw comparison figures for original and generated sequences
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(1))
    plot_original_and_generated(vae, dataset, device)
    plot_umap_distribution(vae, dataset, device, num_samples=100, save_path='umap.csv')

    # Save data
    extract_and_save_latent_and_signals(vae, dataset, device, output_file='latent_and_signals.csv')

