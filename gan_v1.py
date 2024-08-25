#!/ibex/user/nandhan/mambaforge/bin/python3

'''
Header: gan_v1.0.py
The code snippet below is a Generative Adversarial Network (GAN) model that is used to generate synthetic data
replicating chaotic 1D laser signals input to the model. The trained VAE from vae_v1.0.py will serve as the
generator in this GAN model. The GAN will generate data for adversial training of the puf_classifier model.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.linalg import sqrtm
import numpy as np
import pandas as pd
import torch.nn.functional as F
from vae_v1 import VAE


# ----------------------------- GAN Architecture --------------------------------

# Define the Generator model
class GAN_Generator(nn.Module):
    def __init__(self, input_length, latent_dim=20):
        super(GAN_Generator, self).__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim
        
        # Define fully connected layer similar to VAE layer that prepares latent space for decoder
        self.kernel_size1 = 101
        self.kernel_size2 = 11
        self.pooling_size = 10
        self.fc = nn.Linear(latent_dim, 64 * ((((self.input_length-self.kernel_size1+1) // self.pooling_size) - self.kernel_size2+1) // self.pooling_size))
        
        # Load in pretrained VAE model and get decoder
        pretrained_vae = torch.load('saved_models/vae_v1.pth', map_location=device)
        self.generator = pretrained_vae.decoder

    def forward(self, z):
        z = self.fc(z).view(z.size(0), 64, ((((self.input_length-self.kernel_size1+1) // self.pooling_size) - self.kernel_size2+1) // self.pooling_size))
        return self.generator(z)
    

# Define the Discriminator model
class GAN_Discriminator(nn.Module):
    def __init__(self, input_length):
        super(GAN_Discriminator, self).__init__()
        self.input_length = input_length
        self.kernel_size = 100
        self.pooling_size = 10
        self.conv1 = nn.Conv1d(1, 32, kernel_size=self.kernel_size)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(self.pooling_size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * ((self.input_length - self.kernel_size + 1) // self.pooling_size), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    
# ----------------------------- GAN Training --------------------------------

# Define the GAN training function
def train_gan(X):
    # Create dataset
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    
    # Split dataset into train and validation sets
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Define GAN parameters
    latent_dim = 20
    input_length = X.shape[1]
    num_epochs = 100
    latent_dim = 20
    lr = 0.0002
    
    # Instantiate the models and move them to the appropriate device
    generator = GAN_Generator(latent_dim=latent_dim, input_length=input_length).to(device)
    discriminator = GAN_Discriminator(input_length=input_length).to(device)
    
    # Loss and Optimizers
    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)
    
    # Prepare for FID evaluation
    real_data_for_fid = []
    fake_data_for_fid = []
    
    for epoch in range(num_epochs):
        # Training loop
        generator.train()
        discriminator.train()
        total_loss_d_train = 0
        total_loss_g_train = 0
        
        for data in train_loader:
            real_data = data[0]
            batch_size = real_data.size(0)
            real_data = real_data.view(batch_size, 1, -1).float().to(device)
            
            # Collect real data for FID computation
            if epoch == 0:
                real_data_for_fid.append(real_data.cpu().numpy())
            
            # Train Discriminator
            optimizer_d.zero_grad()
            labels_real = torch.ones(batch_size, 1).to(device)
            labels_fake = torch.zeros(batch_size, 1).to(device)
            
            outputs_real = discriminator(real_data)
            loss_real = criterion(outputs_real, labels_real)
            
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_data = generator(z)
            
            outputs_fake = discriminator(fake_data.detach())
            loss_fake = criterion(outputs_fake, labels_fake)
            
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()
            
            total_loss_d_train += loss_d.item()
            
            # Train Generator
            optimizer_g.zero_grad()
            outputs = discriminator(fake_data)
            loss_g = criterion(outputs, labels_real)
            loss_g.backward()
            optimizer_g.step()
            
            total_loss_g_train += loss_g.item()
        
        avg_loss_d_train = total_loss_d_train / len(train_loader)
        avg_loss_g_train = total_loss_g_train / len(train_loader)
        
        # Validation loop
        generator.eval()
        discriminator.eval()
        total_loss_d_val = 0
        total_loss_g_val = 0
        
        with torch.no_grad():
            for data in val_loader:
                real_data = data[0]
                batch_size = real_data.size(0)
                real_data = real_data.view(batch_size, 1, -1).float().to(device)
                
                # Validate Discriminator
                labels_real = torch.ones(batch_size, 1).to(device)
                labels_fake = torch.zeros(batch_size, 1).to(device)
                
                outputs_real = discriminator(real_data)
                loss_real = criterion(outputs_real, labels_real)
                
                z = torch.randn(batch_size, latent_dim).to(device)
                fake_data = generator(z)
                
                outputs_fake = discriminator(fake_data)
                loss_fake = criterion(outputs_fake, labels_fake)
                
                loss_d = loss_real + loss_fake
                total_loss_d_val += loss_d.item()
                
                # Validate Generator
                outputs = discriminator(fake_data)
                loss_g = criterion(outputs, labels_real)
                total_loss_g_val += loss_g.item()
        
        avg_loss_d_val = total_loss_d_val / len(val_loader)
        avg_loss_g_val = total_loss_g_val / len(val_loader)
        
        # Calculate FID score
        if epoch % 10 == 0:  # Calculate FID every 10 epochs
            with torch.no_grad():
                z = torch.randn(len(real_data_for_fid[0]), latent_dim).to(device)
                fake_data_for_fid = generator(z).cpu().numpy()
            
            real_data_for_fid = np.vstack(real_data_for_fid)  # Convert list to numpy array
            
            # Flatten the data if needed
            if real_data_for_fid.ndim == 3:
                num_samples = real_data_for_fid.shape[0]
                num_features = np.prod(real_data_for_fid.shape[1:])
                real_data_for_fid = real_data_for_fid.reshape(num_samples, num_features)

            if fake_data_for_fid.ndim == 3:
                num_samples = fake_data_for_fid.shape[0]
                num_features = np.prod(fake_data_for_fid.shape[1:])
                fake_data_for_fid = fake_data_for_fid.reshape(num_samples, num_features)
            fid_score = calculate_fid_1d(real_data_for_fid, fake_data_for_fid)
            print(f'Epoch [{epoch+1}/{num_epochs}] - FID: {fid_score:.4f}')
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - '
              f'Train Loss D: {avg_loss_d_train:.4f}, '
              f'Train Loss G: {avg_loss_g_train:.4f}, '
              f'Val Loss D: {avg_loss_d_val:.4f}, '
              f'Val Loss G: {avg_loss_g_val:.4f}')

    return generator, discriminator
    
# Function for calculating FID scores for 1D data
def calculate_fid_1d(real_data, fake_data, device=None):
    # Move data to the specified device
    real_data = np.asarray(real_data)
    fake_data = np.asarray(fake_data)

    # Function to calculate mean and covariance
    def calculate_mean_and_covariance(data):
        mu = np.mean(data, axis=0)
        sigma = np.cov(data, rowvar=False)
        return mu, sigma

    # Calculate mean and covariance for real and fake data
    mu_real, sigma_real = calculate_mean_and_covariance(real_data)
    mu_fake, sigma_fake = calculate_mean_and_covariance(fake_data)

    # Calculate the FID score
    ssdiff = np.sum((mu_real - mu_fake) ** 2)
    
    # Calculate covariance and if the result has an imaginary part, discard it (due to numerical errors)
    covmean, _ = sqrtm(np.dot(sigma_real, sigma_fake), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = ssdiff + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    return fid

    
# ----------------------------- GAN Dataset Preparation --------------------------------

# Helper function to process "real" data into sequences of size 10000
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



# ----------------------------- MAIN -----------------------------------------------
if __name__ == '__main__':
    # Check for GPU support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CSV files into DataFrames and combine into one dataframe
    df7_1 = pd.read_csv('puf_dataset_07_14/2Can-D7-50mA-long1.csv')
    df7_2 = pd.read_csv('puf_dataset_07_14/2Can-D7-50mA-long2.csv')
    df7_3 = pd.read_csv('puf_dataset_07_14/2Can-D7-50mA-long3.csv')
    df7_4 = pd.read_csv('puf_dataset_07_14/2Can-D7-50mA-long4.csv')
    df7 = pd.concat([df7_1, df7_2, df7_3, df7_4], axis=0)
    df7 = df7.reset_index(drop=True)

    # Define sequence length
    sequence_length = 10000

    # Split each DataFrame
    X = split_into_sequences(df7, sequence_length)

    # Train the GAN and save the generator model
    gan_generator, gan_discriminator = train_gan(X)
    torch.save(gan_generator, 'saved_models/gan_generator_v1.pth')
