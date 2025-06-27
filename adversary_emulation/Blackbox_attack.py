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


# ----------------------------- CNN_Detector Architecture--------------------------------

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
        features = torch.relu(self.fc3(x))

        multi_class_output = self.fc4(features)
        binary_output = torch.sigmoid(self.fc_binary(features))

        if return_features:
            return multi_class_output, binary_output, features
        else:
            return multi_class_output, binary_output


# ----------------------------- VAE Architecture --------------------------------

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
        logvar = torch.clamp(logvar, min=-10, max=10)
        z = self.reparameterize(mu, logvar)
        z_decoded = self.fc_decode(z)
        x_reconstructed = self.decoder(z_decoded)
        return x_reconstructed, mu, logvar


# ----------------------------- Data Preparation Functions --------------------------------

def split_into_sequences(df, sequence_length):
    
    data = df.iloc[:, 0].values
    data = np.nan_to_num(data, nan=0.0)
    data_min = data.min()
    data_max = data.max()
    data = 2 * (data - data_min) / (data_max - data_min) - 1
    num_sequences = len(data) // sequence_length
    sequences = np.array([data[i * sequence_length:(i + 1) * sequence_length] for i in range(num_sequences)])

    return sequences


def prepare_limited_dataset(real_data_ratio=1, fake_data_ratio=0, sequence_length=10000):
    print("Loading real data (legitimate PUF responses)...")
    df1_real = pd.read_csv('File_Path_True')
 
    X1_real = split_into_sequences(df1_real, sequence_length)
    X_real_full = X1_real

    num_real_samples = int(len(X_real_full) * real_data_ratio)
    real_indices = random.sample(range(len(X_real_full)), num_real_samples)
    X_real_limited = X_real_full[real_indices]

    print(f"Selected {len(X_real_limited)} real samples ({real_data_ratio * 100}% of original)")
    print("Loading fake data for contamination...")
    
    df1_fake = pd.read_csv("File_Path_Fake")

    X1_fake = split_into_sequences(df1_fake, sequence_length)
    X_fake_full = X1_fake

    num_fake_samples = int(len(X_real_limited) * fake_data_ratio)
    fake_indices = random.sample(range(len(X_fake_full)), num_fake_samples)
    X_fake_contamination = X_fake_full[fake_indices]

    print(f"Added {len(X_fake_contamination)} fake samples for contamination ({fake_data_ratio * 100}% of real data)")

    X_attacker = np.vstack([X_real_limited, X_fake_contamination])
    indices = np.random.permutation(len(X_attacker))
    X_attacker = X_attacker[indices]

    print(f"Total attacker dataset size: {len(X_attacker)} samples")

    return X_attacker, len(X_real_limited), len(X_fake_contamination)


# ----------------------------- Black-box query--------------------------------

class TrueBlackBoxQuerySystem:

    def __init__(self, cnn_model, device):
        self.cnn_model = cnn_model
        self.device = device
        self.query_count = 0
        self.successful_queries = 0
        self.query_history = []

        self.successful_samples = []
        self.successful_latents = []
        self.max_success_buffer = 200

    def query_batch(self, generated_samples):

        self.cnn_model.eval()
        with torch.no_grad():
            generated_samples = generated_samples.to(self.device)

            _, binary_output = self.cnn_model(generated_samples)
            predictions = (binary_output > 0.5).float()

            batch_size = generated_samples.size(0)
            self.query_count += batch_size
            self.successful_queries += predictions.sum().item()
            success_rate = predictions.mean().item()

            self.query_history.append({
                'query_batch': len(self.query_history) + 1,
                'batch_size': batch_size,
                'success_rate': success_rate,
                'total_queries': self.query_count
            })

            return predictions, success_rate

    def store_successful_samples(self, samples, latents, predictions):

        success_mask = predictions.squeeze() == 1
        if success_mask.sum() > 0:
            successful_samples = samples[success_mask]
            successful_latents = latents[success_mask]

            for sample, latent in zip(successful_samples, successful_latents):
                self.successful_samples.append(sample.cpu().clone())
                self.successful_latents.append(latent.cpu().clone())

                if len(self.successful_samples) > self.max_success_buffer:
                    self.successful_samples.pop(0)
                    self.successful_latents.pop(0)

            print(f"Found {success_mask.sum().item()} new successful samples! Total: {len(self.successful_samples)}")
            return success_mask.sum().item()
        return 0

    def get_statistics(self):
        overall_success_rate = self.successful_queries / max(self.query_count, 1)
        return {
            'total_queries': self.query_count,
            'successful_queries': self.successful_queries,
            'overall_success_rate': overall_success_rate,
            'query_efficiency': overall_success_rate,
            'successful_samples_count': len(self.successful_samples)
        }


# ----------------------------- VAE-loss-blackbox --------------------------------

def enhanced_vae_loss_with_frequency(x_reconstructed, x, mu, logvar, successful_samples=None):

    device = x.device
    batch_size = x.size(0)
    reconstruction_loss = 0.7 * F.mse_loss(x_reconstructed, x, reduction='mean') + 0.3 * F.l1_loss(x_reconstructed, x,
                                                                                                   reduction='mean')
    mean_loss = torch.tensor(0.0, device=device)
    std_loss = torch.tensor(0.0, device=device)

    try:
        if batch_size > 1:
            target_mean = x.mean(dim=2).detach()
            generated_mean = x_reconstructed.mean(dim=2)
            mean_loss = F.mse_loss(generated_mean, target_mean)
            target_std = x.std(dim=2, unbiased=False).detach()
            generated_std = x_reconstructed.std(dim=2, unbiased=False)
            std_loss = F.l1_loss(generated_std, target_std)
    except:
        pass

    frequency_loss = torch.tensor(0.0, device=device)
    try:
        if batch_size > 1:
            x_fft = torch.fft.fft(x.squeeze(1))
            x_recon_fft = torch.fft.fft(x_reconstructed.squeeze(1))
            x_magnitude = torch.abs(x_fft)
            x_recon_magnitude = torch.abs(x_recon_fft)
            frequency_loss = F.mse_loss(x_recon_magnitude, x_magnitude.detach())
            x_phase = torch.angle(x_fft)
            x_recon_phase = torch.angle(x_recon_fft)
            phase_loss = F.mse_loss(torch.sin(x_recon_phase), torch.sin(x_phase.detach()))
            frequency_loss += 0.1 * phase_loss
    except:
        pass

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    success_guidance_loss = torch.tensor(0.0, device=device)
    if successful_samples is not None and len(successful_samples) > 0:
        try:
            num_targets = min(5, len(successful_samples))
            target_indices = random.sample(range(len(successful_samples)), num_targets)
            targets = torch.stack([successful_samples[i] for i in target_indices]).to(device)

            target_mean_stats = targets.mean(dim=2).mean(dim=0)
            target_std_stats = targets.std(dim=2).mean(dim=0)

            current_mean_stats = x_reconstructed.mean(dim=2).mean(dim=0)
            current_std_stats = x_reconstructed.std(dim=2).mean(dim=0)

            success_guidance_loss = (F.mse_loss(current_mean_stats, target_mean_stats.detach()) +
                                     F.mse_loss(current_std_stats, target_std_stats.detach()))
        except:
            pass

    diversity_loss = torch.tensor(0.0, device=device)
    if batch_size > 1:
        try:
            flattened = x_reconstructed.view(batch_size, -1)
            similarities = torch.mm(F.normalize(flattened, dim=1), F.normalize(flattened, dim=1).t())
            diversity_loss = torch.relu(similarities - 0.8).mean()
        except:
            pass

    total_loss = (1.0 * reconstruction_loss +
                  0.001 * kl_loss +
                  0.5 * (mean_loss + std_loss) +
                  0.3 * frequency_loss +
                  0.4 * success_guidance_loss +
                  0.1 * diversity_loss)

    return total_loss, reconstruction_loss.item(), kl_loss.item(), mean_loss.item(), std_loss.item(), frequency_loss.item()


# ----------------------------- VAE_init --------------------------------

def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ----------------------------- VAE_training (Black-box) --------------------------------

def train_true_blackbox_vae(X_attacker, query_system, device, epochs=200, latent_dim=128, target_success_rate=0.3, total_queries=200000):

    dataset = TensorDataset(torch.tensor(X_attacker, dtype=torch.float32).unsqueeze(1))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_length = X_attacker.shape[1]
    vae = VAE(input_length=input_length, latent_dim=latent_dim).to(device)
    vae.apply(init_weights)

    initial_lr = 2e-3
    optimizer = optim.Adam(vae.parameters(), lr=initial_lr, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[50, 100, 200, 300],
        gamma=0.5
    )

    adaptive_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.8, patience=20, min_lr=1e-5
    )

    total_query_budget = total_queries
    queries_per_epoch = total_query_budget // epochs

    print(f"True black-box VAE training started")
    print(f"Query budget allocation: {queries_per_epoch} queries per epoch for {epochs} epochs")
    print(f"Total planned queries: {queries_per_epoch * epochs}")
    print(f"Learning rate schedule: {initial_lr} -> milestones at [50,100,200,300]")

    training_history = []
    best_success_rate = 0.0

    min_epochs = 200
    patience_threshold = 0.001

    for epoch in range(epochs):
        vae.train()
        total_train_loss = 0
        total_recon_loss = 0
        total_freq_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            x = batch[0].to(device)

            if len(query_system.successful_samples) > 5 and random.random() < 0.3:
                num_replay = min(8, len(query_system.successful_samples), x.size(0) // 2)
                replay_indices = random.sample(range(len(query_system.successful_samples)), num_replay)
                replay_samples = torch.stack([query_system.successful_samples[i] for i in replay_indices]).to(device)
                x = torch.cat([x[:batch_size - num_replay], replay_samples], dim=0)

            optimizer.zero_grad()
            x_reconstructed, mu, logvar = vae(x)

            loss, recon_loss, kl_loss, mean_loss, std_loss, freq_loss = enhanced_vae_loss_with_frequency(
                x_reconstructed, x, mu, logvar, query_system.successful_samples
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            total_recon_loss += recon_loss
            total_freq_loss += freq_loss

        avg_train_loss = total_train_loss / len(train_loader)

        current_success_rate = 0.0
        new_successful_count = 0

        vae.eval()
        with torch.no_grad():
            query_latents = torch.randn(queries_per_epoch, latent_dim).to(device)

            if len(query_system.successful_latents) > 0:
                mutation_count = min(queries_per_epoch // 3, len(query_system.successful_latents))
                if mutation_count > 0:
                    base_indices = random.sample(range(len(query_system.successful_latents)), mutation_count)
                    base_latents = torch.stack([query_system.successful_latents[i] for i in base_indices]).to(device)

                    noise_scale = 0.5 + 0.01 * epoch 
                    noise = torch.randn_like(base_latents) * noise_scale
                    mutated_latents = base_latents + noise

                    query_latents[:mutation_count] = mutated_latents

            query_samples = vae.fc_decode(query_latents)
            query_samples = vae.decoder(query_samples)

            predictions, success_rate = query_system.query_batch(query_samples)
            current_success_rate = success_rate
            
            new_successful_count = query_system.store_successful_samples(
                query_samples, query_latents, predictions
            )

        scheduler.step() 
        if current_success_rate > 0:
            adaptive_scheduler.step(current_success_rate) 

        current_lr = optimizer.param_groups[0]['lr']

        if current_success_rate > best_success_rate:
            best_success_rate = current_success_rate

        stats = query_system.get_statistics()
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'success_rate': current_success_rate,
            'total_queries': stats['total_queries'],
            'overall_success_rate': stats['overall_success_rate'],
            'successful_samples_count': stats['successful_samples_count'],
            'learning_rate': current_lr
        })

        if epoch % 20 == 0 or current_success_rate > 0 or new_successful_count > 0:
            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Loss: {avg_train_loss:.4f} (Recon: {total_recon_loss / len(train_loader):.4f}, "
                  f"Freq: {total_freq_loss / len(train_loader):.4f}) | "
                  f"Success Rate: {current_success_rate:.3f} (Best: {best_success_rate:.3f}) | "
                  f"New Success: {new_successful_count} | "
                  f"Total Success: {stats['successful_samples_count']} | "
                  f"Queries: {stats['total_queries']} | "
                  f"LR: {current_lr:.6f}")

        should_stop = False

        if current_success_rate >= target_success_rate and epoch >= min_epochs:
            print(f"Target success rate {target_success_rate} achieved after {min_epochs} epochs!")
            should_stop = True
        elif current_success_rate > patience_threshold and epoch >= epochs - 10:
            print(f"Late stage with some success, considering early stop.")
            should_stop = True

        if should_stop:
            break

    final_stats = query_system.get_statistics()
    print(f"\nTraining completed after {epoch + 1} epochs!")
    print(f"Best success rate achieved: {best_success_rate:.3f}")
    print(f"Final statistics: {final_stats}")

    return vae, training_history, final_stats


# ----------------------------- Validation --------------------------------

def evaluate_attack_effectiveness(vae, cnn_model, device, num_test_samples=500):
    vae.eval()
    cnn_model.eval()

    with torch.no_grad():
        latent_vectors = torch.randn(num_test_samples, 128).to(device)
        generated_samples = vae.fc_decode(latent_vectors)
        generated_samples = vae.decoder(generated_samples)

        _, binary_output = cnn_model(generated_samples)
        predictions = (binary_output > 0.5).float()
        success_rate = predictions.mean().item()

    return success_rate, generated_samples


def test_single_model(model_path, X_attacker, device, model_name="CNN", epochs=200, total_queries=200000):
    print(f"\n{'=' * 80}")
    print(f"TESTING {model_name.upper()} MODEL: {model_path}")
    print(f"{'=' * 80}")

    sequence_length = X_attacker.shape[1]

    cnn_model = CNN(input_length=sequence_length, num_classes=17).to(device)
    cnn_model.load_state_dict(torch.load(model_path, map_location=device))
    cnn_model.eval()

    query_system = TrueBlackBoxQuerySystem(cnn_model, device)

    print(f"Training true black-box VAE against {model_name} CNN...")
    vae, training_history, final_stats = train_true_blackbox_vae(
        X_attacker, query_system, device, epochs=epochs, total_queries=total_queries, target_success_rate=0.3
    )

    final_success_rate, generated_samples = evaluate_attack_effectiveness(vae, cnn_model, device, num_test_samples=500)

    results = {
        'model_name': model_name,
        'model_path': model_path,
        'total_queries': final_stats['total_queries'],
        'training_success_rate': final_stats['overall_success_rate'],
        'final_test_success_rate': final_success_rate,
        'query_efficiency': final_stats['query_efficiency'],
        'training_epochs': len(training_history),
        'successful_samples_found': final_stats['successful_samples_count']
    }

    print(f"\n{model_name.upper()} CNN Attack Results:")
    print(f"Total queries used: {results['total_queries']}")
    print(f"Training success rate: {results['training_success_rate']:.3f}")
    print(f"Final test success rate: {results['final_test_success_rate']:.3f}")
    print(f"Query efficiency: {results['query_efficiency']:.3f}")
    print(f"Successful samples found: {results['successful_samples_found']}")
    print(f"Training epochs: {results['training_epochs']}")

    return results, training_history


# ----------------------------- Results --------------------------------

def save_detailed_results(original_results, enhanced_results, original_history, enhanced_history,
                          filename='results.csv'):
    results_data = [
        ['Original_CNN', original_results['total_queries'], original_results['final_test_success_rate']],
        ['Enhanced_CNN', enhanced_results['total_queries'], enhanced_results['final_test_success_rate']]
    ]

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Queries', 'Success_Rate'])
        writer.writerows(results_data)

    print(f"Results saved to {filename}")


# ----------------------------- main --------------------------------

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    REAL_DATA_RATIO = 1
    FAKE_DATA_RATIO = 0 

    X_attacker, num_real, num_fake = prepare_limited_dataset(
        real_data_ratio=REAL_DATA_RATIO,
        fake_data_ratio=FAKE_DATA_RATIO
    )
    print(f"Dataset prepared: {num_real} real samples, {num_fake} fake samples")

    original_cnn_path = 'Original_model_path.pth'
    enhanced_cnn_path = 'Enhanced_model_path.pth'

    original_results, original_history = test_single_model(
        original_cnn_path, X_attacker, device, "Original", epochs=200, total_queries=200000
    )

    enhanced_results, enhanced_history = test_single_model(
        enhanced_cnn_path, X_attacker, device, "Enhanced", epochs=200, total_queries=200000
    )

    print(f"\n{'=' * 20} VISUALIZATION {'=' * 20}")
    save_detailed_results(original_results, enhanced_results, original_history, enhanced_history)

