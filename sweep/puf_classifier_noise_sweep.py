#!/ibex/user/nandhan/mambaforge/bin/python3

'''
Header: puf_classifier_v1.5.py
The code snippet below is an extension of the code snippet from puf_classifier_v1.4.py. The code snippet
below is trained on new data measured from new devices on July 14th 2024. It has the same architecture
as the model in puf_classifier_v1.4.py but the data is different.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd

# ----------------------------- Neural Network Architecture --------------------

# Define the 1D CNN model
# 1. Conv1d layer with 32 filters and kernel size of 100
# 2. Batch normalization layer
# 3. Max pooling layer with kernel size of 10
# 4. Flatten layer to convert the 3D tensor to a 1D tensor
# 5. Fully connected layer with 128 output features
# 6. Fully connected layer with 64 output features
# 7. Fully connected layer with 32 output features
# 8. Fully connected layer with 1 output feature (for binary classification)
 

class CNN(nn.Module):
    def __init__(self, input_length):
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

# ----------------------------- Dataset Processing --------------------------------

# Helper function to create vectors with intensity data measurements from CSV file
def create_dataset(df, sequence_size=10000):
    # Convert the single column into a NumPy array
    data = df['Intensity'].values.astype(np.float32)

    # Calculate the number of samples
    num_samples = len(data) // sequence_size

    # Split the data into samples
    X = np.array([data[i * sequence_size:(i + 1) * sequence_size] for i in range(num_samples) if len(data[i * sequence_size:(i + 1) * sequence_size]) == sequence_size])

    # Adding channel dimension
    X = np.expand_dims(X, axis=1)  # Shape: (num_samples, 1, sequence_size)
    
    # Calculate the minimum and maximum values across the entire X array
    # Scale X to range [-1, 1]
    min_val = np.min(X)
    max_val = np.max(X)
    X_scaled = 2 * (X - min_val) / (max_val - min_val) - 1

    return X_scaled

# Function to add Gaussian noise to the data
def add_gaussian_noise(data, noise_percentage=0):
    if noise_percentage == 0:
        return data
    # Compute the standard deviation of the data
    data_std = np.std(data)
    
    # Convert percentage to a fraction
    noise_std = (noise_percentage / 100.0) * data_std

    # Generate Gaussian noise
    noise = np.random.normal(0, noise_std, data.shape).astype(np.float32)
    
    # Add noise to the data
    noisy_data = data + noise
    
    return noisy_data



# ----------------------------- Training and Evaluation ----------------------------

# Distribution size is the number of intensity data measurements in each sample
def train_and_evaluate(X, Y):
    # Debug statement
    print(f"Training with distribution size: {sequence_size}")

    # Create TensorDataset
    dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))

    # Split dataset into train and validation sets
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Instantiate the model, loss function, and optimizer
    input_length = X.shape[2]
    model = CNN(input_length=input_length).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = torch.round(torch.sigmoid(outputs))
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_accuracy = 100 * correct_train / total_train
        
        # Validation loop
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = torch.round(torch.sigmoid(outputs))
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Print the epoch number, training loss, training accuracy, validation loss, and validation accuracy
        print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {train_loss / len(train_loader):.4f}, "
          f"Train Accuracy: {train_accuracy:.2f}%",
          f"Validation Loss: {val_loss/len(val_loader):.4f}, "
          f"Validation Accuracy: {100 * correct / total:.2f}%")
        
    val_accuracy = 100 * correct / total
    
    # Return the results    
    result =  {
        "Distribution Size": sequence_size,
        "Train Loss": train_loss / len(train_loader),
        "Train Accuracy": train_accuracy,
        "Validation Loss": val_loss / len(val_loader),
        "Validation Accuracy": val_accuracy,
    }
    
    return result, model
    



# ----------------------------- Extra Validation ---------------------------------

def extra_validation(model, test_data, test_labels, test2_data, test2_labels, test3_data, test3_labels):
    # Create PyTorch tensor for the real and fake test sets
    test_tensor = torch.tensor(test_data).to(device)
    test2_tensor = torch.tensor(test2_data).to(device)
    test3_tensor = torch.tensor(test3_data).to(device)

    # Set the model to evaluation mode
    model.eval()

    # Make predictions for each of the distributions on probe-again device 7 data (real)
    predictions_test = []
    with torch.no_grad():
        for i in range(len(test_data)):
            output = model(test_tensor[i].unsqueeze(0))
            predicted = torch.round(torch.sigmoid(output))
            predictions_test.append(predicted.item())

    accuracy_test = 100 * np.sum(np.array(predictions_test) == test_labels) / len(test_data)
    
    # Make predictions for each of the distributions on probe-again device 10 data (fake)
    predictions_test2 = []
    with torch.no_grad():
        for i in range(len(test2_data)):
            output = model(test2_tensor[i].unsqueeze(0))
            predicted = torch.round(torch.sigmoid(output))
            predictions_test2.append(predicted.item())
            
    accuracy_test2 = 100 * np.sum(np.array(predictions_test2) == test2_labels) / len(test2_data)

    # Make predictions for each of the distributions on cut data from device 7 (real)
    predictions_test3 = []
    with torch.no_grad():
        for i in range(len(test3_data)):
            output = model(test3_tensor[i].unsqueeze(0))
            predicted = torch.round(torch.sigmoid(output))
            predictions_test3.append(predicted.item())

    accuracy_test3 = 100 * np.sum(np.array(predictions_test3) == test3_labels) / len(test3_data)
    
    # Return the results    
    return {
        "Accuracy on Device 7 Probe-Again": accuracy_test,
        "Accuracy on Device 10 Probe-Again": accuracy_test2,
        "Accuracy on Device 7 Cut": accuracy_test3
    }



    
# ----------------------------- MAIN ----------------------------------------------

# Check for GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters for training and evaluation
sequence_size = 10000
noise_percentages = np.arange(0.0, 100.0, 1.0)
results = []

# Load the CSV file with measurements from devices into Pandas DataFrames
df1 = pd.read_csv("puf_dataset_07_14/2Can-D1-50mA.csv")
df2 = pd.read_csv("puf_dataset_07_14/2Can-D2-50mA.csv")
df3 = pd.read_csv("puf_dataset_07_14/2Can-D3-50mA.csv")
df4 = pd.read_csv("puf_dataset_07_14/2Can-D4-50mA.csv")
df5 = pd.read_csv("puf_dataset_07_14/2Can-D5-50mA.csv")
df7_1 = pd.read_csv("puf_dataset_07_14/2Can-D7-50mA-long1.csv")
df7_2 = pd.read_csv("puf_dataset_07_14/2Can-D7-50mA-long2.csv")
df7_3 = pd.read_csv("puf_dataset_07_14/2Can-D7-50mA-long3.csv")
df7_4 = pd.read_csv("puf_dataset_07_14/2Can-D7-50mA-long4.csv")
df7 = pd.concat([df7_1, df7_2, df7_3, df7_4], axis=0)
df7 = df7.reset_index(drop=True)
df8 = pd.read_csv("puf_dataset_07_14/2Can-D8-50mA-w.csv")
df10_1 = pd.read_csv("puf_dataset_07_14/2Can-D10-50mA.csv")
df10_2 = pd.read_csv("puf_dataset_07_14/2Can-D10-50mA-long2.csv")
df10_3 = pd.read_csv("puf_dataset_07_14/2Can-D10-50mA-long3.csv")
df10_4 = pd.read_csv("puf_dataset_07_14/2Can-D10-50mA-long4.csv")
df10 = pd.concat([df10_1, df10_2, df10_3, df10_4], axis=0)
df10 = df10.reset_index(drop=True)
df10_test = pd.read_csv("puf_dataset_07_14/p-2Can-D10-50mA.csv")
df7_test = pd.read_csv("puf_dataset_07_14/p-2Can-D7-50mA.csv")

# Split the data from CSV files into samples and add channel dimension
# Shape of final NumPy array for data from a CSV file: (num_samples, 1, sequence_size)
# Generate labels for CSV files (1 for "real" and 0 for "fake" distributions)
# For this model, device 7 is real, all other devices are fake
X1 = create_dataset(df1, sequence_size)  
X2 = create_dataset(df2, sequence_size)
X3 = create_dataset(df3, sequence_size)
X4 = create_dataset(df4, sequence_size)
X5 = create_dataset(df5, sequence_size)
X7 = create_dataset(df7, sequence_size)
X8 = create_dataset(df8, sequence_size)
X10 = create_dataset(df10, sequence_size)

Y1 = np.zeros(len(X1)).astype(np.float32)
Y2 = np.zeros(len(X2)).astype(np.float32)
Y3 = np.zeros(len(X3)).astype(np.float32)
Y4 = np.zeros(len(X4)).astype(np.float32)
Y5 = np.zeros(len(X5)).astype(np.float32)
Y7 = np.ones(len(X7)).astype(np.float32)
Y8 = np.zeros(len(X8)).astype(np.float32)
Y10 = np.zeros(len(X10)).astype(np.float32)

# Create datasets for extra validation
# Use the p-2Can-D7-50mA.csv file as the test set to see if it recognizes the real device that has been reprobed
# Test the trained model on data from the probed again device 10 and see if it recognizes it as fake
# Also reserve some data from device 7 for testing, see if it recognizes the real device
probe_again_dev7_data = create_dataset(df7_test, sequence_size)
probe_again_dev7_labels = np.ones(len(probe_again_dev7_data)).astype(np.float32)
probe_again_dev10_data = create_dataset(df10_test, sequence_size)
probe_again_dev10_labels = np.zeros(len(probe_again_dev10_data)).astype(np.float32)
dev7_cut_data = X7[-1000:]
X7 = X7[:-1000]
dev7_cut_labels = Y7[-1000:]
Y7 = Y7[:-1000]

# Concatenate data from different CSV files
X_dataset = np.concatenate((X1, X2, X3, X4, X5, X7, X8, X10), axis=0).astype(np.float32)
Y_dataset = np.concatenate((Y1, Y2, Y3, Y4, Y5, Y7, Y8, Y10), axis=0).astype(np.float32)

# Sweep through different noise percentages
for noise_percentage in noise_percentages:
    # Add noise to validation data
    noisy_probe_again_dev7_data = add_gaussian_noise(probe_again_dev7_data, noise_percentage)
    noisy_probe_again_dev10_data = add_gaussian_noise(probe_again_dev10_data, noise_percentage)
    noisy_dev7_cut_data = add_gaussian_noise(dev7_cut_data, noise_percentage)
    noisy_X_dataset = add_gaussian_noise(X_dataset, noise_percentage)
    
    # Evaluate the model on the noisy data
    result, model = train_and_evaluate(noisy_X_dataset, Y_dataset)
    extra_validation_result = extra_validation(model, noisy_probe_again_dev7_data, probe_again_dev7_labels, 
                                               noisy_probe_again_dev10_data, probe_again_dev10_labels, noisy_dev7_cut_data, dev7_cut_labels)
    
    # Add the noise percentage to the results
    extra_validation_result["Noise Percentage"] = noise_percentage
    results.append(extra_validation_result)

# Save results to a CSV file
results_df = pd.DataFrame(results, columns=[
    "Noise Percentage",
    'Accuracy on Device 10 Probe-Again', 
    'Accuracy on Device 7 Cut', 
    'Accuracy on Device 7 Probe-Again'
])
results_df.to_csv("sweep/noise_sweep_results/results.csv", index=False)
print("Results saved to results.csv")