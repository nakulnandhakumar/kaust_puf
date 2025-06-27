
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import torch.nn.functional as F
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


# ----------------------------- Original task --------------------------------

def load_original_task_data(device, sequence_length=10000):
    def create_segments_like_original(df, seq_len):
        arr = df.iloc[:, 0].to_numpy(dtype=np.float32)
        arr = np.nan_to_num(arr)
        n = len(arr) // seq_len
        segments = np.array([
            arr[i * seq_len:(i + 1) * seq_len]
            for i in range(n)
        ], dtype=np.float32)
        return segments[:, None, :]

    file_info = [
        ("Demo_Nontarget_1.csv", 0, 0.0),
        ("Demo_Nontarget_2.csv", 1, 0.0),
        ...
        ("Demo_Target.csv", n, 1.0)
    ]

    train_segments = []
    train_multi_labels = []
    train_binary_labels = []
    val_segments = []
    val_multi_labels = []
    val_binary_labels = []

    TRAIN_VAL_RATIO = 0.7

    print("Loading original task data using file-by-file approach...")

    for path, multi_label, binary_label in file_info:
        try:
            print(f"Processing {path}...")
            df = pd.read_csv(path, usecols=[4])
            segs = create_segments_like_original(df, sequence_length)

            if len(segs) == 0:
                print(f"  Warning: No segments created from {path}")
                del df
                continue
            mn, mx = segs.min(), segs.max()
            if mx - mn > 0:
                segs = 2 * (segs - mn) / (mx - mn) - 1
            else:
                print(f"  Warning: Constant values in {path}, skipping normalization")
            n = len(segs)
            split = int(TRAIN_VAL_RATIO * n)

            if split > 0:
                train_segments.append(segs[:split])
                train_multi_labels.append(np.full(split, multi_label, dtype=np.int64))
                train_binary_labels.append(np.full(split, binary_label, dtype=np.float32))

            if n - split > 0:
                val_segments.append(segs[split:])
                val_multi_labels.append(np.full(n - split, multi_label, dtype=np.int64))
                val_binary_labels.append(np.full(n - split, binary_label, dtype=np.float32))

            print(f"  Loaded {len(segs)} segments, train: {split}, val: {n - split}")

            del df, segs

        except Exception as e:
            print(f"  Error loading {path}: {e}")
            continue

    if not train_segments and not val_segments:
        raise RuntimeError("No data could be loaded from any file")
    if train_segments:
        X_train = np.vstack(train_segments)
        Y_multi_train = np.concatenate(train_multi_labels)
        Y_binary_train = np.concatenate(train_binary_labels)
    else:
        X_train = np.array([]).reshape(0, 1, sequence_length)
        Y_multi_train = np.array([])
        Y_binary_train = np.array([])

    if val_segments:
        X_val = np.vstack(val_segments)
        Y_multi_val = np.concatenate(val_multi_labels)
        Y_binary_val = np.concatenate(val_binary_labels)
    else:
        X_val = np.array([]).reshape(0, 1, sequence_length)
        Y_multi_val = np.array([])
        Y_binary_val = np.array([])

    del train_segments, train_multi_labels, train_binary_labels
    del val_segments, val_multi_labels, val_binary_labels

    print(f"Total: Train {len(X_train)}, Val {len(X_val)}")
    if len(X_val) > 0:
        dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(Y_multi_val, dtype=torch.long),
            torch.tensor(Y_binary_val, dtype=torch.float32)
        )
        del X_val, Y_multi_val, Y_binary_val
    elif len(X_train) > 0:
        print("Warning: No validation data, using subset of training data for evaluation")
        subset_size = min(1000, len(X_train))
        indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_subset = X_train[indices]
        Y_multi_subset = Y_multi_train[indices]
        Y_binary_subset = Y_binary_train[indices]

        dataset = TensorDataset(
            torch.tensor(X_subset, dtype=torch.float32),
            torch.tensor(Y_multi_subset, dtype=torch.long),
            torch.tensor(Y_binary_subset, dtype=torch.float32)
        )
        del X_subset, Y_multi_subset, Y_binary_subset
    else:
        raise RuntimeError("No data available for evaluation")

    if 'X_train' in locals():
        del X_train, Y_multi_train, Y_binary_train

    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    print("Original task data loaded successfully")
    return test_loader


def evaluate_original_task(cnn_model, test_loader, device):
    cnn_model.eval()
    correct_multi = 0
    correct_binary = 0
    total = 0

    with torch.no_grad():
        for inputs, multi_labels, binary_labels in test_loader:
            inputs = inputs.to(device)
            multi_labels = multi_labels.to(device)
            binary_labels = binary_labels.to(device)

            multi_outputs, binary_outputs = cnn_model(inputs)

            _, predicted_multi = torch.max(multi_outputs, 1)
            correct_multi += (predicted_multi == multi_labels).sum().item()

            predicted_binary = (binary_outputs.squeeze() >= 0.5).float()
            correct_binary += (predicted_binary == binary_labels).sum().item()

            total += multi_labels.size(0)

    multi_accuracy = correct_multi / total if total > 0 else 0.0
    binary_accuracy = correct_binary / total if total > 0 else 0.0
    return multi_accuracy, binary_accuracy

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
    reconstruction_loss = 0.7 * F.mse_loss(x_reconstructed, x, reduction='mean') + 0.3 * F.l1_loss(x_reconstructed, x,
                                                                                                   reduction='mean')

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

    try:
        cnn_model.eval()
        with torch.no_grad():
            _, _, features_real = cnn_model(x, return_features=True)

        original_training_mode = cnn_model.training
        cnn_model.train()
        _, _, features_fake = cnn_model(x_reconstructed, return_features=True)
        cnn_model.train(original_training_mode)

        if features_real.size() == features_fake.size() and features_real.numel() > 0:
            features_real_norm = F.normalize(features_real, dim=1)
            features_fake_norm = F.normalize(features_fake, dim=1)
            feature_loss = 0.7 * F.mse_loss(features_fake_norm, features_real_norm, reduction='mean') + 0.3 * F.l1_loss(
                features_fake_norm, features_real_norm, reduction='mean')
        else:
            feature_loss = torch.tensor(0.0, device=x.device)
    except Exception as e:
        print(f"Warning: Feature computation failed. Error: {e}")
        feature_loss = torch.tensor(0.0, device=x.device)

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
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

    try:
        print("Loading original task data...")
        original_task_loader = load_original_task_data(device)
        current_multi_accuracy, current_binary_accuracy = evaluate_original_task(cnn_model, original_task_loader,
                                                                                 device)
        print(
            f"Initial original task - Multi-class accuracy: {current_multi_accuracy:.4f}, Binary accuracy: {current_binary_accuracy:.4f}")

        multi_task_threshold = max(0.85, current_multi_accuracy * 0.95)
        binary_task_threshold = max(0.85, current_binary_accuracy * 0.95)

        print(f"Original task thresholds - Multi: {multi_task_threshold:.4f}, Binary: {binary_task_threshold:.4f}")

    except Exception as e:
        print(f"Error: Could not load original task data: {e}")
        print("Training cannot continue without original task data!")
        return None, None

    feedback_start_epoch = 10
    final_lambda_feedback = 0.2
    initial_lambda_feedback = 0.01
    kl_target = latent_dim / 2 + 5
    kl_start_epoch = 10
    max_lambda_kl = 0.0005

    patience = 3
    val_acc_threshold = 0.98
    early_stop_counter = 0

    original_task_violation_count = 0
    max_violations = 3
    performance_recovery_mode = False

    lambda_kl_initial = 1e-6

    performance_history = {
        'multi_accuracy': [current_multi_accuracy],
        'binary_accuracy': [current_binary_accuracy]
    }
    window_size = 3

    def create_infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch

    original_task_infinite_iter = create_infinite_loader(original_task_loader)

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

        for batch_idx, batch in enumerate(train_loader):
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
            cnn_optimizer.zero_grad()

            real_data = x
            fake_data = x_reconstructed.detach()
            cnn_input = torch.cat([real_data, fake_data], dim=0)
            cnn_labels = torch.cat([
                torch.ones(real_data.size(0), 1, device=device),
                torch.zeros(fake_data.size(0), 1, device=device)
            ], dim=0)

            _, binary_output = cnn_model(cnn_input)
            detection_loss = F.binary_cross_entropy(binary_output, cnn_labels)

            original_task_loss = torch.tensor(0.0, device=device)
            try:
                orig_batch = next(original_task_infinite_iter)
                orig_inputs, orig_multi_labels, orig_binary_labels = orig_batch
                orig_inputs = orig_inputs.to(device)
                orig_multi_labels = orig_multi_labels.to(device)
                orig_binary_labels = orig_binary_labels.to(device)

                orig_multi_output, orig_binary_output = cnn_model(orig_inputs)
                orig_multi_loss = F.cross_entropy(orig_multi_output, orig_multi_labels)
                orig_binary_loss = F.binary_cross_entropy(orig_binary_output.squeeze(), orig_binary_labels)
                original_task_loss = 0.7 * orig_multi_loss + 0.3 * orig_binary_loss

            except Exception as e:
                print(f"Warning: Original task loss computation failed: {e}")
                original_task_loss = torch.tensor(0.0, device=device)

            avg_multi_accuracy = sum(performance_history['multi_accuracy'][-window_size:]) / min(window_size, len(
                performance_history['multi_accuracy']))
            avg_binary_accuracy = sum(performance_history['binary_accuracy'][-window_size:]) / min(window_size,
                                                                                                   len(
                                                                                                       performance_history[
                                                                                                           'binary_accuracy']))

            if performance_recovery_mode:
                detection_weight = 0.1
                original_weight = 0.9
            elif avg_multi_accuracy < multi_task_threshold or avg_binary_accuracy < binary_task_threshold:
                detection_weight = 0.3
                original_weight = 0.7
            elif epoch < 20:
                detection_weight = 0.4
                original_weight = 0.6
            elif epoch < 50:
                detection_weight = 0.5
                original_weight = 0.5
            else:
                detection_weight = 0.55
                original_weight = 0.45

            total_cnn_loss = detection_weight * detection_loss + original_weight * original_task_loss
            total_cnn_loss.backward()
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

        try:
            current_multi_accuracy, current_binary_accuracy = evaluate_original_task(cnn_model, original_task_loader,
                                                                                     device)

            performance_history['multi_accuracy'].append(current_multi_accuracy)
            performance_history['binary_accuracy'].append(current_binary_accuracy)

            if len(performance_history['multi_accuracy']) > window_size * 2:
                performance_history['multi_accuracy'] = performance_history['multi_accuracy'][-window_size * 2:]
                performance_history['binary_accuracy'] = performance_history['binary_accuracy'][-window_size * 2:]

        except Exception as e:
            print(f"Warning: Could not evaluate original task: {e}")
            current_multi_accuracy = performance_history['multi_accuracy'][-1]
            current_binary_accuracy = performance_history['binary_accuracy'][-1]

        recent_multi_avg = sum(performance_history['multi_accuracy'][-window_size:]) / min(window_size, len(
            performance_history['multi_accuracy']))
        recent_binary_avg = sum(performance_history['binary_accuracy'][-window_size:]) / min(window_size, len(
            performance_history['binary_accuracy']))

        if recent_multi_avg < multi_task_threshold or recent_binary_avg < binary_task_threshold:
            original_task_violation_count += 1
            print(
                f"VIOLATION {original_task_violation_count}/{max_violations} - Multi: {recent_multi_avg:.4f} < {multi_task_threshold:.4f}, Binary: {recent_binary_avg:.4f} < {binary_task_threshold:.4f}")

            if original_task_violation_count == max_violations:
                if not performance_recovery_mode:
                    print(f"ENTERING PERFORMANCE RECOVERY MODE!")
                    performance_recovery_mode = True
                    lambda_feedback *= 0.5
                    print(f"   Reduced lambda_feedback to {lambda_feedback:.4f}")
        else:
            if original_task_violation_count > 0:
                print(f"Original task performance recovered!")
            original_task_violation_count = 0
            if performance_recovery_mode:
                print(f"EXITING PERFORMANCE RECOVERY MODE!")
                performance_recovery_mode = False

        if performance_recovery_mode and epoch > 0 and epoch % 10 == 0:
            if recent_multi_avg < multi_task_threshold * 0.8 or recent_binary_avg < binary_task_threshold * 0.8:
                print(f"CRITICAL: Original task performance severely degraded. Stopping training.")
                print(f"   Multi: {recent_multi_avg:.4f} << {multi_task_threshold:.4f}")
                print(f"   Binary: {recent_binary_avg:.4f} << {binary_task_threshold:.4f}")
                break

        scheduler_vae.step(val_accuracy)
        scheduler_cnn.step(val_accuracy)
        current_lr = vae_optimizer.param_groups[0]['lr']

        avg_gen_mean = gen_mean_total / num_batches
        avg_real_mean = real_mean_total / num_batches
        avg_gen_std = gen_std_total / num_batches
        avg_real_std = real_std_total / num_batches

        if (val_accuracy >= 0.98 and
                avg_train_kl < kl_target and
                recent_multi_avg >= 0.9 and
                recent_binary_avg >= 0.9 and
                not performance_recovery_mode):
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping criteria met. Training stopped.")
                break
        else:
            early_stop_counter = 0

        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Recon: {avg_train_recon:.4f}, Class: {avg_train_class:.4f}, KL: {avg_train_kl:.4f}, Mean Loss: {avg_train_mean:.4f}, Std Loss: {avg_train_std:.4f} "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
              f"Multi Task Acc: {current_multi_accuracy:.4f} (avg: {recent_multi_avg:.4f}), "
              f"Binary Task Acc: {current_binary_accuracy:.4f} (avg: {recent_binary_avg:.4f}), "
              f"Generated Mean: {avg_gen_mean:.4f}, Real Mean: {avg_real_mean:.4f}, "
              f"Generated Std: {avg_gen_std:.4f}, Real Std: {avg_real_std:.4f}, "
              f"Lambda_feedback: {lambda_feedback:.4f}, Lambda_kl: {lambda_kl:.4f}, "
              f"Current LR: {current_lr:.10f}, Weights(D/O): {detection_weight:.2f}/{original_weight:.2f}")

    return vae, original_task_loader

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

# ----------------------------- Extract Signals --------------------------------

def extract_and_save_latent_and_signals(vae, dataset, device, output_file='latent_and_signals.csv', num_samples=100):
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
            original_sample = dataset[idx][0].unsqueeze(0).to(device)
            encoded = vae.encoder(original_sample)
            mu = vae.fc_mu(encoded)
            latent_vector = mu.squeeze().cpu().numpy()

            reconstructed_sample, _, _ = vae(original_sample)

            original_signal = original_sample.squeeze().cpu().numpy()
            reconstructed_signal = reconstructed_sample.squeeze().cpu().numpy()

            all_original_signals.append(original_signal)
            all_reconstructed_signals.append(reconstructed_signal)
            all_latent_vectors.extend(latent_vector)

        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(['Sample_Index', 'Data_Point', 'Original_Signal', 'Reconstructed_Signal', 'Latent_Vector'])
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

# ----------------------------- MAIN --------------------------------------------

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    df1 = pd.read_csv('Demo_True.csv')
    sequence_length = 10000
    X1 = split_into_sequences(df1, sequence_length)
    X = X1

    # Load CNN model
    # num_classes should be equal to the number of CRPs
    cnn_model = CNN(input_length=sequence_length, num_classes=1).to(device)
    cnn_model.load_state_dict(torch.load('./saved_models/Original_CNN.pth', map_location=device))

    # Evaluate baseline performance
    original_task_loader = load_original_task_data(device)
    baseline_multi, baseline_binary = evaluate_original_task(cnn_model, original_task_loader, device)
    print(f"Baseline - Multi: {baseline_multi:.4f}, Binary: {baseline_binary:.4f}")

    # Train VAE
    print("Starting VAE training...")
    vae, final_loader = train_vae_with_adversarial_cnn(X, cnn_model, device, epochs=100, latent_dim=128)

    if final_loader is not None:
        # Final evaluation
        final_multi, final_binary = evaluate_original_task(cnn_model, final_loader, device)
        print(f"Final - Multi: {final_multi:.4f}, Binary: {final_binary:.4f}")
        print(f"Change - Multi: {final_multi - baseline_multi:+.4f}, Binary: {final_binary - baseline_binary:+.4f}")

        # Save models
        torch.save(cnn_model.state_dict(), './saved_models/Enhanced_CNN.pth')
        torch.save(vae.state_dict(), './saved_models/VAE.pth')

        # Save results
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(1))
        extract_and_save_latent_and_signals(vae, dataset, device, output_file='latent_and_signals_review.csv')
        print("Training completed and models saved.")
    else:
        print("Training failed - no evaluation data available")

