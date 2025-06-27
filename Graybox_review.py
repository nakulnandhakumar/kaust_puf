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


# ----------------------------- CNN_Detector Architecture --------------------------------

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

    def forward(self, x, return_features=False, return_logits=False):
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
        logits = self.fc_binary(features).squeeze()
        binary_output = torch.sigmoid(logits)

        if return_features and return_logits:
            return multi_class_output, binary_output, features, logits
        elif return_features:
            return multi_class_output, binary_output, features
        elif return_logits:
            return multi_class_output, binary_output, logits
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


def prepare_limited_dataset(real_data_ratio=0.2, fake_data_ratio=0.15, sequence_length=10000):
    print("Loading real data (legitimate PUF responses)...")
    # Load real data
    df1_real = pd.read_csv('File_Path_True')

    # Process real data
    X1_real = split_into_sequences(df1_real, sequence_length)
    X_real_full = X1_real

    # Sample limited real data
    num_real_samples = int(len(X_real_full) * real_data_ratio)
    real_indices = random.sample(range(len(X_real_full)), num_real_samples)
    X_real_limited = X_real_full[real_indices]

    print(f"Selected {len(X_real_limited)} real samples ({real_data_ratio * 100}% of original)")

    print("Loading fake data for contamination...")
    # Load fake data
    df1_fake = pd.read_csv('File_Path_Fake')

    # Process fake data
    X1_fake = split_into_sequences(df1_fake, sequence_length)

    X_fake_full = X1_fake

    # Sample fake data for contamination
    num_fake_samples = int(len(X_real_limited) * fake_data_ratio)
    fake_indices = random.sample(range(len(X_fake_full)), num_fake_samples)
    X_fake_contamination = X_fake_full[fake_indices]

    print(f"Added {len(X_fake_contamination)} fake samples for contamination ({fake_data_ratio * 100}% of real data)")

    # Combine and shuffle
    X_attacker = np.vstack([X_real_limited, X_fake_contamination])
    indices = np.random.permutation(len(X_attacker))
    X_attacker = X_attacker[indices]

    print(f"Total attacker dataset size: {len(X_attacker)} samples")

    return X_attacker, len(X_real_limited), len(X_fake_contamination)


# --------------------- Gray-box query (leakaged quantized confidence ratio) ---------------------------

class GreyBoxConfidenceQuerySystem:

    def __init__(self, cnn_model, device, query_budget=None):
        self.cnn_model = cnn_model
        self.device = device
        self.query_count = 0
        self.query_budget = None
        self.successful_queries = 0
        self.query_history = []

        self.confidence_threshold = 0.5
        self.noise_std = 0.05
        self.successful_samples = []
        self.successful_latents = []
        self.successful_confidences = []
        self.max_success_buffer = 100

    def query_batch(self, generated_samples):

        batch_size = generated_samples.size(0)

        self.cnn_model.eval()
        with torch.no_grad():
            x = generated_samples.detach().clone().to(self.device)

            _, binary_prob = self.cnn_model(x)

            noise = torch.randn_like(binary_prob) * self.noise_std
            noisy_conf = torch.clamp(binary_prob + noise, 0.0, 1.0)

            quantized_confidences = torch.round(noisy_conf * 10) / 10
            predictions = (quantized_confidences >= self.confidence_threshold).float()

            quantized_confidences = quantized_confidences.detach().cpu()
            predictions = predictions.detach().cpu()

        self.query_count += batch_size
        self.successful_queries += int(predictions.sum().item())
        success_rate = predictions.mean().item()
        self.query_history.append({
            'query_batch': len(self.query_history) + 1,
            'batch_size': batch_size,
            'success_rate': success_rate,
            'total_queries': self.query_count
        })

        return quantized_confidences, predictions, success_rate

    def store_successful_samples(self, samples, latents, confidences):
        success_mask = confidences > self.confidence_threshold

        if success_mask.sum().item() > 0:
            successful_samples = samples[success_mask]
            successful_latents = latents[success_mask]
            successful_confidences = confidences[success_mask]

            for sample, latent, conf in zip(successful_samples, successful_latents, successful_confidences):
                self.successful_samples.append(sample.cpu().clone())
                self.successful_latents.append(latent.cpu().clone())
                self.successful_confidences.append(conf.cpu().clone())

                if len(self.successful_samples) > self.max_success_buffer:
                    self.successful_samples.pop(0)
                    self.successful_latents.pop(0)
                    self.successful_confidences.pop(0)

            return success_mask.sum().item()
        return 0

    def get_statistics(self):
        overall_success_rate = self.successful_queries / max(self.query_count, 1)
        return {
            'total_queries': self.query_count,
            'successful_queries': self.successful_queries,
            'overall_success_rate': overall_success_rate,
            'query_efficiency': overall_success_rate,
            'successful_samples_count': len(self.successful_samples),
            'confidence_threshold': self.confidence_threshold,  # 添加这行
            'avg_confidence': 0.5  # 添加这行，提供默认值
        }


def test_greybox_model(model_path, X_attacker, device, model_name="CNN"):
    print("\n" + "=" * 60)
    print(f"Testing {model_name} CNN with Grey-box Attack")
    print("=" * 60)

    sequence_length = X_attacker.shape[1]
    cnn_model = CNN(input_length=sequence_length, num_classes=17).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            cnn_model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            cnn_model.load_state_dict(checkpoint['state_dict'])
        else:
            cnn_model.load_state_dict(checkpoint)
    else:
        cnn_model.load_state_dict(checkpoint)
    cnn_model.eval()

    with torch.no_grad():
        probe = torch.tensor(X_attacker[:5], dtype=torch.float32).unsqueeze(1).to(device)
        _, probe_conf = cnn_model(probe)
        print("Sanity confidences (first 5 real seq):",
              ", ".join(f"{c.item():.3f}" for c in probe_conf.squeeze()))

    query_system = GreyBoxConfidenceQuerySystem(cnn_model, device, query_budget=None)  # 设置合理预算
    print("Training grey-box VAE.")
    vae, training_history, final_stats = train_greybox_confidence_vae(
        X_attacker, query_system, device,
        epochs=50, target_success_rate=0.3
    )

    original_results = final_stats
    original_history = training_history
    original_confidences = [e['avg_confidence'] for e in training_history]

    return original_results, original_history, original_confidences


# -------------------------- VAE-Loss-guided-by-confidence-ratio ------------------------------

def confidence_guided_vae_loss(x_reconstructed, x, mu, logvar,
                               confidences=None, successful_samples=None,
                               recon_weight=1.0, kl_weight=1e-3,
                               conf_weight=5.0):

    device = x.device
    batch_size = x.size(0)

    if confidences is not None:
        confidences = confidences.to(device)

    rec_mse = F.mse_loss(x_reconstructed, x, reduction='none')
    rec_l1 = F.l1_loss(x_reconstructed, x, reduction='none')
    rec_per_sample = (0.7 * rec_mse + 0.3 * rec_l1).mean(dim=[1, 2])  # shape = [B]
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    mean_loss = torch.tensor(0.0, device=device)
    std_loss = torch.tensor(0.0, device=device)
    if batch_size > 1:
        tgt_mean = x.mean(dim=2).detach()
        gen_mean = x_reconstructed.mean(dim=2)
        mean_loss = F.mse_loss(gen_mean, tgt_mean)
        tgt_std = x.std(dim=2, unbiased=False).detach()
        gen_std = x_reconstructed.std(dim=2, unbiased=False)
        std_loss = F.l1_loss(gen_std, tgt_std)

    frequency_loss = torch.tensor(0.0, device=device)
    if batch_size > 1:
        x_fft = torch.fft.fft(x.squeeze(1))
        x_recon_fft = torch.fft.fft(x_reconstructed.squeeze(1))
        frequency_loss = F.mse_loss(
            x_recon_fft.abs(),
            x_fft.abs().detach()
        )

    success_loss = torch.tensor(0.0, device=device)
    if successful_samples and len(successful_samples) > 0:
        num_tgt = min(5, len(successful_samples))
        idxs = random.sample(range(len(successful_samples)), num_tgt)
        tgt_samps = torch.stack(
            [successful_samples[i] for i in idxs]
        ).to(device)
        tgt_stats = tgt_samps.mean(dim=2).mean(dim=0)
        cur_stats = x_reconstructed.mean(dim=2).mean(dim=0)
        success_loss = F.mse_loss(cur_stats, tgt_stats.detach())

    diversity_loss = torch.tensor(0.0, device=device)
    if batch_size > 1:
        flat = x_reconstructed.view(batch_size, -1)
        sims = torch.mm(
            F.normalize(flat, 1),
            F.normalize(flat, 1).t()
        )
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        diversity_loss = torch.relu(sims[mask] - 0.8).mean()

    if confidences is not None:
        conf_weights = torch.where(confidences > 0.6,
                                   torch.ones_like(confidences) * 0.5,
                                   torch.ones_like(confidences) * 2.0)

        weighted_recon = (rec_per_sample * conf_weights.to(rec_per_sample.device)).mean()
        conf_term = torch.tensor(0.0, device=x.device)
    else:
        conf_term = torch.tensor(0.0, device=x.device)
        weighted_recon = rec_per_sample.mean()

    total_loss = (
            weighted_recon
            + kl_weight * kl_loss
            + 0.3 * (mean_loss + std_loss)
            + 0.2 * frequency_loss
            + 0.3 * success_loss
            + 0.05 * diversity_loss
            + conf_term
    )

    return (total_loss,
            weighted_recon.item(),
            kl_loss.item(),
            mean_loss.item(),
            std_loss.item(),
            frequency_loss.item(),
            success_loss.item(),
            diversity_loss.item(),
            conf_term.item())


# ----------------------------- VAE-init --------------------------------

def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ----------------------------- Graybox-training --------------------------------

def train_greybox_confidence_vae(X_attacker, query_system, device,
                                 epochs=50, latent_dim=128,
                                 target_success_rate=0.3):

    dataset = TensorDataset(torch.tensor(X_attacker, dtype=torch.float32).unsqueeze(1))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, _ = random_split(dataset, [train_size, val_size])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_length = X_attacker.shape[1]
    vae = VAE(input_length=input_length, latent_dim=latent_dim).to(device)
    vae.apply(init_weights)

    initial_lr = 2e-3
    optimizer = optim.Adam(vae.parameters(), lr=initial_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 60, 120, 160], gamma=0.7
    )

    gen_conf_weight = 3.0
    queries_per_epoch = 100
    queries_per_batch = 25

    training_history = []
    best_strict_succ = 0.0
    best_dyn_succ = 0.0
    best_avg_conf = 0.0

    for epoch in range(epochs):
        vae.train()
        total_train_loss = 0.0

        for batch_idx, (x_batch,) in enumerate(train_loader):
            x = x_batch.to(device)

            if len(query_system.successful_samples) > 5:
                num_replay = min(16,
                                 len(query_system.successful_samples),
                                 x.size(0) // 2)
                replay_inds = random.sample(
                    range(len(query_system.successful_samples)),
                    num_replay
                )
                replay_samples = torch.stack(
                    [query_system.successful_samples[i] for i in replay_inds]
                ).to(device)
                x = torch.cat([x[:batch_size - num_replay], replay_samples], dim=0)

            optimizer.zero_grad()
            x_rec, mu, logvar = vae(x)

            if batch_idx == 0:
                try:
                    with torch.no_grad():
                        conf_rec_quantized, _, _ = query_system.query_batch(x_rec.detach())
                except Exception as e:
                    print(f"Query failed: {e}")
                    conf_rec_quantized = None
            else:
                conf_rec_quantized = None

            if epoch < 20 and epoch % 10 == 0:
                z_rand = torch.randn_like(mu[:5])
                gen_rand = vae.decoder(vae.fc_decode(z_rand))
                with torch.no_grad():
                    conf_rand_quantized, _, _ = query_system.query_batch(gen_rand.detach())
                diversity_penalty = -0.1 * conf_rand_quantized.var()
                gen_conf_loss = diversity_penalty.to(device)
            else:
                gen_conf_loss = torch.tensor(0.0, device=device)

            loss, recon_l, kl_l, mean_l, std_l, freq_l, succ_l, div_l, conf_term = \
                confidence_guided_vae_loss(
                    x_rec, x, mu, logvar,
                    confidences=conf_rec_quantized,  # 使用量化的无梯度置信度
                    successful_samples=query_system.successful_samples
                )

            total_loss = loss + gen_conf_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += total_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        vae.eval()
        dyn_success_count = 0
        strict_success_count = 0
        sum_confidence = 0.0

        actual_queries = queries_per_epoch
        num_batches = max(1, actual_queries // queries_per_batch)

        with torch.no_grad():
            for batch_num in range(num_batches):
                try:
                    z = torch.randn(queries_per_batch, latent_dim, device=device)
                    if len(query_system.successful_latents) > 0:
                        m = min(queries_per_batch // 3,
                                len(query_system.successful_latents))
                        base_inds = random.sample(
                            range(len(query_system.successful_latents)), m
                        )
                        base_z = torch.stack(
                            [query_system.successful_latents[i]
                             for i in base_inds]
                        ).to(device)
                        noise = torch.randn_like(base_z) * random.choice([0.1, 0.2, 0.3])
                        z[:m] = base_z + noise

                    gen = vae.decoder(vae.fc_decode(z))
                    conf_batch, pred_dyn, _ = query_system.query_batch(gen)

                    dyn_success_count += int(pred_dyn.sum().item())
                    strict_success_count += int((conf_batch > 0.5).float().sum().item())
                    sum_confidence += conf_batch.sum().item()
                    dynamic_threshold = max(0.2, 0.5 - (50 - epoch) * 0.006) if epoch < 50 else 0.5
                    mask_success = (conf_batch > dynamic_threshold)
                    if mask_success.any():
                        good_x = gen[mask_success].detach().cpu()
                        good_z = z[mask_success].detach().cpu()
                        query_system.store_successful_samples(
                            good_x, good_z, conf_batch[mask_success]
                        )
                except Exception as e:
                    if "budget exhausted" in str(e):
                        print(f"Query budget exhausted at epoch {epoch + 1}")
                        break
                    else:
                        raise e

        succ_dyn = dyn_success_count / queries_per_epoch
        succ_0_5 = strict_success_count / queries_per_epoch
        avg_conf = sum_confidence / queries_per_epoch

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        best_dyn_succ = max(best_dyn_succ, succ_dyn)
        best_strict_succ = max(best_strict_succ, succ_0_5)
        best_avg_conf = max(best_avg_conf, avg_conf)

        stats = query_system.get_statistics()
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'dyn_success_rate': succ_dyn,
            'success_rate': succ_0_5,
            'avg_confidence': avg_conf,
            'total_queries': stats['total_queries'],
            'overall_success_rate': stats['overall_success_rate'],
            'successful_samples_count': stats['successful_samples_count'],
            'learning_rate': current_lr,
            'confidence_threshold': stats['confidence_threshold']
        })

        print(f"Epoch {epoch + 1:3d}  Loss: {avg_train_loss:6.4f} "
              f"SuccDyn: {succ_dyn:5.3f}  Succ0.5: {succ_0_5:5.3f} "
              f"AvgConf: {avg_conf:5.3f} "
              f"Queries: {stats['total_queries']:6d}/{query_system.query_budget}")

        if succ_0_5 >= target_success_rate and epoch >= 50:
            print(f"Target strict success rate reached at epoch {epoch + 1}")
            break

    final_stats = query_system.get_statistics()
    print(f"\nTraining completed! Best SuccDyn: {best_dyn_succ:.3f}, "
          f"Best Succ0.5: {best_strict_succ:.3f}, Best AvgConf: {best_avg_conf:.3f}")
    print(f"\n=== QUERY STATISTICS ===")
    print(f"Total queries used: {query_system.query_count}")
    print(f"Successful queries: {query_system.successful_queries}")
    print(f"Overall success rate: {query_system.successful_queries / max(1, query_system.query_count):.4f}")

    return vae, training_history, final_stats


def test_simple_attacks(cnn_model, X_real, device, model_name="CNN"):
    print(f"\n=== Testing {model_name} against simple attacks ===")
    cnn_model.eval()

    test_samples = X_real[:500] 

    with torch.no_grad():
        noise_samples = torch.randn(500, X_real.shape[1])
        noise_samples = noise_samples.unsqueeze(1).to(device)
        _, noise_conf = cnn_model(noise_samples)
        noise_success = (noise_conf > 0.5).float().mean().item()

        real_tensor = torch.tensor(test_samples, dtype=torch.float32)
        gaussian_noise = torch.randn_like(real_tensor) * 0.2
        gauss_samples = (real_tensor + gaussian_noise).unsqueeze(1).to(device)
        _, gauss_conf = cnn_model(gauss_samples)
        gauss_success = (gauss_conf > 0.5).float().mean().item()

        scale_factor = 1.5
        scaled_samples = (real_tensor * scale_factor).unsqueeze(1).to(device)
        _, scale_conf = cnn_model(scaled_samples)
        scale_success = (scale_conf > 0.5).float().mean().item()

        real_samples = real_tensor.unsqueeze(1).to(device)
        _, real_conf = cnn_model(real_samples)
        real_accuracy = (real_conf > 0.5).float().mean().item()

    results = {
        'random_noise_success': noise_success,
        'gaussian_mix_success': gauss_success,
        'scaling_success': scale_success,
        'real_data_accuracy': real_accuracy
    }

    print(f"Real data accuracy: {real_accuracy:.3f}")
    print(f"Random noise attack success: {noise_success:.3f}")
    print(f"Gaussian mix attack success: {gauss_success:.3f}")
    print(f"Scaling attack success: {scale_success:.3f}")

    return results


def save_greybox_results(original_results, enhanced_results, filename='greybox_results.csv'):
    results = {
        'Original_CNN': original_results,
        'Enhanced_CNN': enhanced_results
    }

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Total_Queries', 'Success_Rate', 'Avg_Confidence'])

        for model_name, data in results.items():
            writer.writerow([
                model_name,
                data['total_queries'],
                data['overall_success_rate'],
                data.get('avg_confidence', 0.5)
            ])

    print(f"Results saved to {filename}")


# ----------------------------- main --------------------------------

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_attacker, num_real, num_fake = prepare_limited_dataset(
        real_data_ratio=0.5,
        fake_data_ratio=0.1
    )
    print(f"Dataset: {num_real} real + {num_fake} fake = {len(X_attacker)} total")

    original_cnn_path = 'Original_model_path.pth'
    enhanced_cnn_path = 'Enhanced_model_path.pth'

    print("Testing Enhanced CNN...")
    enhanced_results, enhanced_history, _ = test_greybox_model(
        enhanced_cnn_path, X_attacker, device, "Enhanced"
    )

    print("Testing Original CNN...")
    original_results, original_history, _ = test_greybox_model(
        original_cnn_path, X_attacker, device, "Original"
    )

    save_greybox_results(original_results, enhanced_results)

    print(f"\nResults Summary:")
    print(f"Original CNN - Success Rate: {original_results['overall_success_rate']:.3f}")
    print(f"Enhanced CNN - Success Rate: {enhanced_results['overall_success_rate']:.3f}")
    print("Analysis completed!")