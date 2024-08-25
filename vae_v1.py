#!/ibex/user/nandhan/mambaforge/bin/python3

'''
Header: vae_v1.0.py
The code snippet below is a Variational Autoencoder (VAE) model that is used to generate synthetic data
replicating chaotic 1D laser signals input to the model. The VAE will serve as the
basis for a generator in a Generative Adversarial Network (GAN) model meant to generate
data for adversial training of the puf_classifier model.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import torch.nn.functional as F

# ----------------------------- VAE Architecture --------------------------------

# Define the Variational Autoencoder (VAE) model
class VAE(nn.Module):
    def __init__(self, input_length, latent_dim=20):
        super(VAE, self).__init__()
        self.input_length = input_length
        self.conv_kernel_size1 = 101
        self.conv_kernel_size2 = 11
        self.deconv_kernel_size1 = 11
        self.deconv_kernel_size2 = 200
        self.pooling_size = 10
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=self.conv_kernel_size1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(self.pooling_size),
            nn.Conv1d(32, 64, kernel_size=self.conv_kernel_size2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(self.pooling_size),
            nn.Flatten()
        )
        self.fc1 = nn.Linear(64 * ((((input_length-self.conv_kernel_size1+1) // self.pooling_size) -self.conv_kernel_size2+1) // self.pooling_size), latent_dim)
        self.fc2 = nn.Linear(64 * ((((input_length-self.conv_kernel_size1+1) // self.pooling_size) - self.conv_kernel_size2+1) // self.pooling_size), latent_dim)
        self.fc3 = nn.Linear(latent_dim, 64 * ((((input_length-self.conv_kernel_size1+1) // self.pooling_size) - self.conv_kernel_size2+1) // self.pooling_size))
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=self.deconv_kernel_size1, stride=10),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.ConvTranspose1d(32, 1, kernel_size=self.deconv_kernel_size2, stride=10),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc1(x)
        logvar = self.fc2(x)
        z = self.reparameterize(mu, logvar)
        z = self.fc3(z).view(x.size(0), 64, ((((self.input_length-self.conv_kernel_size1+1) // self.pooling_size) - self.conv_kernel_size2+1) // self.pooling_size))
        return self.decoder(z), mu, logvar


# ----------------------------- VAE Training --------------------------------

# Train the VAE model
def train_vae(X):
    # Create dataset
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(1))
    
    # Split dataset into train and validation sets
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Define VAE parameters, instantiate the model, and define the optimizer
    input_length = X.shape[1]
    latent_dim = 20
    epochs = 50
    vae = VAE(input_length=input_length, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=0.0001, weight_decay=1e-5)
    
    # Training and validation loop
    for epoch in range(epochs):
        # Training loop
        vae.train()
        total_train_loss = 0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            
            # Forward pass
            x_reconstructed, mu, logvar = vae(x)
            
            # Compute the loss
            recon_loss = F.mse_loss(x_reconstructed, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            
            # Backpropagation and clipping the gradients
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation Loop
        vae.eval()
        total_val_loss = 0
        total_val_mse = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                
                # Forward pass
                x_reconstructed, mu, logvar = vae(x)
                
                # Compute the loss
                recon_loss = F.mse_loss(x_reconstructed, x, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl_loss
                
                 # Compute MSE
                x_reconstructed_np = x_reconstructed.cpu().numpy().flatten()
                x_np = x.cpu().numpy().flatten()
                batch_val_mse = mse(x_reconstructed_np, x_np)
                
                # Total loss and SSIM
                total_val_mse += batch_val_mse
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_mse = total_val_mse / len(val_loader)
        
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Validation Loss: {avg_val_loss:.4f}, '
              f'Validation MSE: {avg_val_mse:.4f}')
    
    # Return the model
    return vae

# Function to calculate the similiarity between two 1D signals through MSE
def mse(signal1, signal2):
    return np.mean((signal1 - signal2) ** 2)
        
# ----------------------------- VAE Data Preparation --------------------------------

# Helper function to scale and process "real" data into sequences of size 10000
def split_into_sequences(df, sequence_length):
    data = df['Intensity'].values
    
    # Scaling data to be between -1 and 1
    data_min = data.min()
    data_max = data.max()
    data = 2 * (data - data_min) / (data_max - data_min) - 1
    
    # Splitting into sequences
    num_sequences = len(data) // sequence_length
    sequences = np.array([data[i*sequence_length:(i+1)*sequence_length] for i in range(num_sequences)])
    
    return sequences



# ----------------------------- MAIN --------------------------------------------

if __name__ == '__main__':
    # Check for GPU support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CSV files into DataFrames and combine them into one dataframe
    df7_1 = pd.read_csv('puf_dataset_07_14/2Can-D7-50mA-long1.csv')
    df7_2 = pd.read_csv('puf_dataset_07_14/2Can-D7-50mA-long2.csv')
    df7_3 = pd.read_csv('puf_dataset_07_14/2Can-D7-50mA-long3.csv')
    df7_4 = pd.read_csv('puf_dataset_07_14/2Can-D7-50mA-long4.csv')
    df7 = pd.concat([df7_1, df7_2, df7_3, df7_4], axis=0)
    df7 = df7.reset_index(drop=True)

    # Define sequence length
    sequence_length = 10000

    # Split the dataframe into scaled sequences of size sequence_length
    X = split_into_sequences(df7, sequence_length)

    # Train the VAE and save the model
    vae = train_vae(X)
    torch.save(vae, 'saved_models/vae_v1.pth')
