'''
Header: puf_classifier_v2.py
This code trains a 1D CNN model to classify the data from the CSV files as "real" or "fake" based on the device
the data was measured from. One device is arbitrarily chosen as the "real" device and the rest are considered "fake"
devices. The model is trained on data from 7 devices and tested on data from an 8th device. The model is trained
using the CrossEntropy loss function and the Adam optimizer. The model is trained for 10 epochs and the training
and validation accuracy are printed after each epoch. The model is then tested on data from the 8th device and the
accuracy is printed.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd

# ----------------------------- Neural Network Architecture --------------------

# Define the 1D CNN model

# Model 1st Iteration:
# 1. Conv1d layer with 32 filters and kernel size of 100
# 2. Batch normalization layer
# 3. Max pooling layer with kernel size of 10
# 4. Flatten layer to convert the 3D tensor to a 1D tensor
# 5. Fully connected layer with 128 output features
# 6. Fully connected layer with 64 output features
# 7. Fully connected layer with 32 output features
# 8. Fully connected layer with 2 output features (for binary classification)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        kernel_size = 3
        pooling_size = 2
        self.conv1 = nn.Conv1d(1, 32, kernel_size=kernel_size)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(pooling_size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * ((input_length - kernel_size + 1) // pooling_size), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)
    
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

# Model 2nd Iteration:
# 1. Conv1d layer with 32 filters, kernel size of 10, and padding of 5
# 2. Batch normalization layer
# 3. Max pooling layer with kernel size of 4
# 4. Conv1d layer with 64 filters, kernel size of 10, and padding of 5
# 5. Batch normalization layer
# 6. Max pooling layer with kernel size of 4
# 7. Flatten layer to convert the 3D tensor to a 1D tensor
# 8. Fully connected layer with 128 output features
# 9. Fully connected layer with 64 output features
# 10. Fully connected layer with 32 output features
# 11. Fully connected layer with 2 output features (for binary classification)
class CNN_2(nn.Module):
    def __init__(self):
        conv_kernel_size=3
        pool_kernel_size=2
        conv2_kernel_size=3
        pool2_kernel_size=2
        super(CNN_2, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=conv_kernel_size, padding=conv_kernel_size//2)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(pool_kernel_size)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=conv2_kernel_size, padding=conv2_kernel_size//2)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(pool2_kernel_size)
        self.flatten = nn.Flatten()
        fc_input_size = 64 * ( ( (input_length // pool_kernel_size)  //  pool2_kernel_size ) )
        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.batch_norm2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

# ----------------------------- Dataset Preparation ----------------------------

# Helper function to create vectors with intensity data measurements from CSV file
def create_datasets(df, distribution_size):
    # Convert the single column into a NumPy array
    data = df['Intensity'].values.astype(np.float32)

    # Calculate the number of samples
    num_samples = len(data) // distribution_size

    # Split the data into samples
    X = np.array([data[i * distribution_size:(i + 1) * distribution_size] for i in range(num_samples) if len(data[i * distribution_size:(i + 1) * distribution_size]) == distribution_size])
    
    # Calculate the minimum and maximum values across the entire X array
    # Scale X_combined to range [-1, 1]
    min_val = np.min(X)
    max_val = np.max(X)
    X_scaled = 2 * (X - min_val) / (max_val - min_val) - 1

    # Adding channel dimension
    X_scaled = np.expand_dims(X_scaled, axis=1)  # Shape: (num_samples, 1, distribution_size)
    
    # Initialize a new array to hold the histogram counts with shape (num_samples, 1, bins)
    bins = 50
    X_counts = np.zeros((num_samples, 1, bins))

    # Iterate over each sample and compute the histogram counts
    for i in range(num_samples):
        counts, _ = np.histogram(X[i, 0], bins=bins, range=(-1, 1), density=True)
        X_counts[i, 0] = counts

    # Confirm the X_counts contains only floating-point values
    X_counts = X_counts.astype(np.float32)
    
    return X_counts

# Load the CSV file with measurements from device 1 and 2 into dataframes
df1 = pd.read_csv("puf_dataset_07_08/Can-D1-50mA2.csv")
df2 = pd.read_csv("puf_dataset_07_08/Can-D2-50mA2.csv")
df3 = pd.read_csv("puf_dataset_07_08/Can-D3-50mA.csv")
df4 = pd.read_csv("puf_dataset_07_08/Can-D4-50mA.csv")
df5 = pd.read_csv("puf_dataset_07_08/Can-D5-50mA.csv")
df6 = pd.read_csv("puf_dataset_07_08/Can-D6-50mA.csv")
df7 = pd.read_csv("puf_dataset_07_08/Can-D7-50mA.csv")

# Number of contiguous intensity data measurements in each distribution
distribution_size = 10000

# Split the data from CSV files into samples and add channel dimension
# Shape of final NumPy array for data from a CSV file: (num_samples, 1, distribution_size)
X1 = create_datasets(df1, distribution_size)  
X2 = create_datasets(df2, distribution_size)
X3 = create_datasets(df3, distribution_size)  
X4 = create_datasets(df4, distribution_size)
X5 = create_datasets(df5, distribution_size)
X6 = create_datasets(df6, distribution_size)
X7 = create_datasets(df7, distribution_size)

# Generate labels for CSV files (1 for "real" and 0 for "fake" distributions)
Y1 = np.ones(len(X1)).astype(np.int64)
Y2 = np.zeros(len(X2)).astype(np.int64)
Y3 = np.zeros(len(X3)).astype(np.int64)
Y4 = np.zeros(len(X4)).astype(np.int64)
Y5 = np.zeros(len(X5)).astype(np.int64)
Y6 = np.zeros(len(X6)).astype(np.int64)
Y7 = np.zeros(len(X7)).astype(np.int64)

# Remove the last 20 samples from X1 and last 20 labels from Y1 and place value from X1 in separate array for testing
test = X1[-10:]
X1 = X1[:-10]
test_labels = Y1[-10:]
Y1 = Y1[:-10]

# Concatenate data from different CSV files
X_combined = np.concatenate((X1, X2, X3, X4, X5, X6, X7), axis=0)
Y_combined = np.concatenate((Y1, Y2, Y3, Y4, Y5, Y6, Y7), axis=0)

# Create TensorDataset
dataset = TensorDataset(torch.tensor(X_combined), torch.tensor(Y_combined))

# Split dataset into train and validation sets
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)



# ----------------------------- Training the Model -----------------------------

# Instantiate the model, loss function, and optimizer
input_length = X_combined.shape[2]
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
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
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {train_loss / len(train_loader):.4f}, "
          f"Train Accuracy: {train_accuracy:.2f}%",
          f"Validation Loss: {val_loss/len(val_loader):.4f}, "
          f"Validation Accuracy: {100 * correct / total:.2f}%")


# ----------------------------- Extra Validation -------------------------------

# Test the trained model on data from Can-D8-50mA.csv
df8 = pd.read_csv("puf_dataset_07_08/Can-D8-50mA.csv")
X8 = create_datasets(df8, distribution_size)

# Since all of the distributions from device 8 are fake, create labels with zeros
Y8 = np.zeros(len(X8)).astype(np.int64)

# Create tensors from the NumPy arrays
X8_tensor = torch.tensor(X8)
test_tensor = torch.tensor(test)

# Set the model to evaluation mode
model.eval()

# Make predictions for each of the distributions from X8_tensor using the trained model and calculate accuracy
predictions_dev8 = []
with torch.no_grad():
    for i in range(len(X8_tensor)):
        output = model(X8_tensor[i].unsqueeze(0))
        _, predicted = torch.max(output.data, 1)
        predictions_dev8.append(predicted.item())
        
accuracy_dev8 = 100 * np.sum(np.array(predictions_dev8) == Y8) / len(X8_tensor)
print(f"Accuracy on Can-D8-50mA.csv: {accuracy_dev8:.2f}%")

# Make predictions for each of the distributions from X1_tensor using the trained model and calculate accuracy
predictions_dev1 = []
with torch.no_grad():
    for i in range(len(test_tensor)):
        output = model(test_tensor[i].unsqueeze(0))
        _, predicted = torch.max(output.data, 1)
        predictions_dev1.append(predicted.item())

accuracy_dev1 = 100 * np.sum(np.array(predictions_dev1) == test_labels) / len(test_tensor)
print(f"Accuracy on test set Device 1: {accuracy_dev1:.2f}%")
print(predictions_dev1)
