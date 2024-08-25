#!/ibex/user/nandhan/mambaforge/bin/python3

'''
Header: puf_classifier_distribution_sweep.py
This is an extension of the puf_classifier_v1.5.py script. This script constructs
the same 1D CNN model as the one in puf_classifier_v1.5.py, but sweeps over 
multiple different sizes of distributions of intensity data measurements. The
accuracy of the model is recorded for each distribution size to be graphed later.
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

# ----------------------------- Dataset Preparation ----------------------------

# Function to add Gaussian noise to the data
def add_gaussian_noise(data, noise_percentage):
    # Compute the standard deviation of the data
    data_std = np.std(data)
    
    # Convert percentage to a fraction
    noise_std = (noise_percentage / 100.0) * data_std

    # Generate Gaussian noise
    noise = np.random.normal(0, noise_std, data.shape)
    
    # Add noise to the data
    noisy_data = data + noise
    
    return noisy_data

# Helper function to create vectors with intensity data measurements from CSV file
def create_datasets(df, sequence_size):
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

# Helper function to split dataset into train and validation sets and add noise to validation set
def split_and_add_noise(X, Y, val_size, noise_percentage):
    # Split the data into training and validation sets
    train_size = len(X) - val_size
    train_X, val_X = X[:train_size], X[train_size:]
    train_Y, val_Y = Y[:train_size], Y[train_size:]
    
    # Add Gaussian noise to the validation set
    if noise_percentage > 0:
        val_X = add_gaussian_noise(val_X, noise_percentage).astype(np.float32)
    
    return train_X, train_Y, val_X, val_Y



# Check for GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# Initialize array to store results of sweep
results = []
count = 0

# ----------------------------- Training and Evaluation -----------------------------

# Distribution size is the number of intensity data measurements in each sample
def train_and_evaluate(sequence_size, noise_percentage=0.0):
    # Debug statement
    print(f"Training with noise percentage: {noise_percentage}%")
    
    # Split the data from CSV files into samples and add channel dimension
    # Shape of final NumPy array for data from a CSV file: (num_samples, 1, sequence_size)
    X1 = create_datasets(df1, sequence_size)  
    X2 = create_datasets(df2, sequence_size)
    X3 = create_datasets(df3, sequence_size)
    X4 = create_datasets(df4, sequence_size)
    X5 = create_datasets(df5, sequence_size)
    X7 = create_datasets(df7, sequence_size)
    X8 = create_datasets(df8, sequence_size)
    X10 = create_datasets(df10, sequence_size)

    Y1 = np.zeros(len(X1)).astype(np.float32)
    Y2 = np.zeros(len(X2)).astype(np.float32)
    Y3 = np.zeros(len(X3)).astype(np.float32)
    Y4 = np.zeros(len(X4)).astype(np.float32)
    Y5 = np.zeros(len(X5)).astype(np.float32)
    Y7 = np.ones(len(X7)).astype(np.float32)
    Y8 = np.zeros(len(X8)).astype(np.float32)
    Y10 = np.zeros(len(X10)).astype(np.float32)

    # Use the p-2Can-D7-50mA.csv file as the test set to see if it recognizes the real device that has been reprobed
    test = create_datasets(df7_test, sequence_size)
    test = add_gaussian_noise(test, noise_percentage).astype(np.float32)
    test_labels = np.ones(len(test)).astype(np.float32)

    # Test the trained model on data from the probed again device 10 and see if it recognizes it as fake
    # Since this device is fake, the model should predict a label of 0
    test2 = create_datasets(df10_test, sequence_size)
    test2 = add_gaussian_noise(test2, noise_percentage).astype(np.float32)
    test2_labels = np.zeros(len(test2)).astype(np.float32)

    # Also reserve some data from device 7 for testing
    test3 = X7[-100:]
    test3 = add_gaussian_noise(test3, noise_percentage).astype(np.float32)
    X7 = X7[:-100]
    test3_labels = Y7[-100:]
    Y7 = Y7[:-100]

    # Concatenate data from different CSV files
    X_combined = np.concatenate((X1, X2, X3, X4, X5, X7, X8, X10), axis=0).astype(np.float32)
    Y_combined = np.concatenate((Y1, Y2, Y3, Y4, Y5, Y7, Y8, Y10), axis=0).astype(np.float32)

    # Split dataset into train and validation sets and add noise to validation set
    val_size = int(0.3 * len(X_combined))
    train_X, train_Y, val_X, val_Y = split_and_add_noise(X_combined, Y_combined, val_size, noise_percentage)

    # Create TensorDataset
    train_dataset = TensorDataset(torch.tensor(train_X), torch.tensor(train_Y))
    val_dataset = TensorDataset(torch.tensor(val_X), torch.tensor(val_Y))

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Instantiate the model, loss function, and optimizer
    input_length = X_combined.shape[2]
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
        
    val_accuracy = 100 * correct / total


    # ----------------------------- Extra Validation -------------------------------

    # Create PyTorch tensor for the real and fake test sets
    test_tensor = torch.tensor(test).to(device)
    test2_tensor = torch.tensor(test2).to(device)
    test3_tensor = torch.tensor(test3).to(device)

    # Set the model to evaluation mode
    model.eval()

    # Make predictions for each of the distributions on probe-again device 7 data (real)
    predictions_test = []
    with torch.no_grad():
        for i in range(len(test)):
            output = model(test_tensor[i].unsqueeze(0))
            predicted = torch.round(torch.sigmoid(output))
            predictions_test.append(predicted.item())

    accuracy_test = 100 * np.sum(np.array(predictions_test) == test_labels) / len(test)
    
    # Make predictions for each of the distributions on probe-again device 10 data (fake)
    predictions_test2 = []
    with torch.no_grad():
        for i in range(len(test2)):
            output = model(test2_tensor[i].unsqueeze(0))
            predicted = torch.round(torch.sigmoid(output))
            predictions_test2.append(predicted.item())
            
    accuracy_test2 = 100 * np.sum(np.array(predictions_test2) == test2_labels) / len(test2)

    # Make predictions for each of the distributions on cut data from device 7 (real)
    predictions_test3 = []
    with torch.no_grad():
        for i in range(len(test3)):
            output = model(test3_tensor[i].unsqueeze(0))
            predicted = torch.round(torch.sigmoid(output))
            predictions_test3.append(predicted.item())

    accuracy_test3 = 100 * np.sum(np.array(predictions_test3) == test3_labels) / len(test3)
    
    # Return the results    
    return {
        "Distribution Size": sequence_size,
        "Noise Percentage": noise_percentage,
        "Train Loss": train_loss / len(train_loader),
        "Train Accuracy": train_accuracy,
        "Validation Loss": val_loss / len(val_loader),
        "Validation Accuracy": val_accuracy,
        "Accuracy on Device 10 Probe-Again": accuracy_test2,
        "Accuracy on Device 7 Cut": accuracy_test3,
        "Accuracy on Device 7 Probe-Again": accuracy_test
    }
    

# ----------------------------- Save Results to CSV File -------------------------    

# Sweeping over noise percentages from 0% to 25% in steps of 0.1%
results = []
sequence_size = 10000
noise_percentage = 0.0
for sequence_size in range(8700, 10001, 50):
    result = train_and_evaluate(sequence_size, noise_percentage)
    results.append(result)

# Save results to a CSV file
results_df = pd.DataFrame(results, columns=[
    "Distribution Size",
    "Noise Percentage",
    "Train Loss",
    "Train Accuracy",
    "Validation Loss",
    "Validation Accuracy",
    'Accuracy on Device 10 Probe-Again', 
    'Accuracy on Device 7 Cut', 
    'Accuracy on Device 7 Probe-Again'
])
results_df.to_csv("sweep/sequence_size_sweep_results/results6.csv", index=False)
print("Results saved to results6.csv")