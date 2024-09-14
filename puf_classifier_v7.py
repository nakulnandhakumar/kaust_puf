#!/ibex/user/nandhan/mambaforge/bin/python3

'''
Header: puf_classifier_v7.py
The code snippet below is an extension of the code snippet from puf_classifier_v5.py. The code snippet
below is trained on all devices from the first two sets of experiments. The code snippet below is trained
using the CrossEntropyLoss() loss function which applies the softmax function to the output layer neurons
for multiple class classification problems. The goal of the model in this file is to classify the data as 
beloning to a certain device.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
def create_dataset(df, sequence_size):
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



# ----------------------------- Training and Evaluation ----------------------------

# Distribution size is the number of intensity data measurements in each sample
def train_and_evaluate(X, Y):
    # Debug statement
    print(f"Training with sequence size: {sequence_size}")

    # Create TensorDataset
    dataset = TensorDataset(torch.tensor(X), torch.tensor(Y).long())

    # Split dataset into train and validation sets
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Instantiate the model, loss function, and optimizer
    num_classes = len(np.unique(Y))
    input_length = X.shape[2]
    model = CNN(input_length=input_length, num_classes=num_classes).to(device)
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
            inputs, labels = inputs.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
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
                inputs, labels = inputs.to(device), labels.to(device).long()
                outputs = model(inputs).squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
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

def extra_validation(model, test_data, test_labels):
    # Create PyTorch tensor for the real and fake test sets
    test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)

    # Set the model to evaluation mode
    model.eval()
    
    # Make predictions for each of the distributions on device 5 data (fake)
    predictions_test = []
    with torch.no_grad():
        for i in range(len(test_data)):
            output = model(test_tensor[i].unsqueeze(0))
            _, predicted = torch.max(output, 1)
            predictions_test.append(predicted.item())
            
    accuracy_test = 100 * np.sum(np.array(predictions_test) == test_labels.cpu().numpy()) / len(test_data)
    
    # Set the model back to training mode
    model.train()
    
    # Return the results    
    return accuracy_test


def extra_validation_preds(model, test_data):
    # Create PyTorch tensor for the real and fake test sets
    test_tensor = torch.tensor(test_data).to(device)

    # Set the model to evaluation mode
    model.eval()
    
    # Make predictions for each of the distributions on device 5 data (fake)
    predictions_test = []
    with torch.no_grad():
        for i in range(len(test_data)):
            output = model(test_tensor[i].unsqueeze(0))
            _, predicted = torch.max(output, 1)
            predictions_test.append(predicted.item())
            
    # Convert predictions_test to a NumPy array
    predictions_test = np.array(predictions_test)
    
    # Set the model back to training mode
    model.train()
    
    # Return the results    
    return predictions_test


# Method to plot the confusion matrices
def plot_confusion_matrix(cm, title, filename, save_dir):
    plt.figure(figsize=(8, 6))
    
    # Define vmax and vmin for the color scale of the confusion matrix
    vmin = 0
    vmax = cm.max()
    
    # Plot the confusion matrix with larger annotations and axis labels
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=cm_labels_str, yticklabels=cm_labels_str, vmin=vmin, vmax=vmax,
                annot_kws={"size": 14},  # Increase the font size for the annotations
                linewidths=0.5, linecolor='black')
    
    # Set the title and axis labels with larger font size
    plt.title(title, fontsize=18)
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('Actual', fontsize=16)
    
    # Adjust the tick labels size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the figure as a PNG file
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    
# ----------------------------- MAIN ----------------------------------------------
if __name__ == "__main__":
    # Check for GPU support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parameters for training and evaluation
    sequence_size = 10000
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
    df10_p2 = pd.read_csv("puf_dataset_07_14/p-2Can-D10-50mA.csv")
    
    df1_old = pd.read_csv("puf_dataset_07_08/Can-D1-50mA2.csv")
    df2_old = pd.read_csv("puf_dataset_07_08/Can-D2-50mA2.csv")
    df3_old = pd.read_csv("puf_dataset_07_08/Can-D3-50mA.csv")
    df4_old = pd.read_csv("puf_dataset_07_08/Can-D4-50mA.csv")
    df5_old = pd.read_csv("puf_dataset_07_08/Can-D5-50mA.csv")
    df6_old = pd.read_csv("puf_dataset_07_08/Can-D6-50mA.csv")
    df7_old = pd.read_csv("puf_dataset_07_08/Can-D7-50mA.csv")
    df8_old = pd.read_csv("puf_dataset_07_08/Can-D8-50mA.csv")
    df10_old = pd.read_csv("puf_dataset_07_08/Can-D10-50mA.csv")

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
    X10_p2 = create_dataset(df10_p2, sequence_size)
    X1_old = create_dataset(df1_old, sequence_size)
    X2_old = create_dataset(df2_old, sequence_size)
    X3_old = create_dataset(df3_old, sequence_size)
    X4_old = create_dataset(df4_old, sequence_size)
    X5_old = create_dataset(df5_old, sequence_size)
    X6_old = create_dataset(df6_old, sequence_size)
    X7_old = create_dataset(df7_old, sequence_size)
    X10_old = create_dataset(df10_old, sequence_size)

    Y1 = np.full(len(X1), 0).astype(np.float32)
    Y2 = np.full(len(X2), 1).astype(np.float32)
    Y3 = np.full(len(X3), 2).astype(np.float32)
    Y4 = np.full(len(X4), 3).astype(np.float32)
    Y5 = np.full(len(X5), 4).astype(np.float32)
    Y7 = np.full(len(X7), 5).astype(np.float32)
    Y8 = np.full(len(X8), 6).astype(np.float32)
    Y10 = np.full(len(X10), 7).astype(np.float32)
    Y10_p2 = np.full(len(X10_p2), 8).astype(np.float32)
    Y1_old = np.full(len(X1_old), 9).astype(np.float32)
    Y2_old = np.full(len(X2_old), 10).astype(np.float32)
    Y3_old = np.full(len(X3_old), 11).astype(np.float32)
    Y4_old = np.full(len(X4_old), 12).astype(np.float32)
    Y5_old = np.full(len(X5_old), 13).astype(np.float32)
    Y6_old = np.full(len(X6_old), 14).astype(np.float32)
    Y7_old = np.full(len(X7_old), 15).astype(np.float32)
    Y10_old = np.full(len(X10_old), 16).astype(np.float32)

    # Create datasets for extra validation
    # Cut data from seen devices 1,2,3,4,5,7,8 and see if the model correctly predicts the right device
    dev1_cut_data = X1[-200:]
    X1 = X1[:-200]
    dev1_cut_labels = Y1[-200:]
    Y1 = Y1[:-200]
    
    dev2_cut_data = X2[-200:]
    X2 = X2[:-200]
    dev2_cut_labels = Y2[-200:]
    Y2 = Y2[:-200]
    
    dev3_cut_data = X3[-200:]
    X3 = X3[:-200]
    dev3_cut_labels = Y3[-200:]
    Y3 = Y3[:-200]
    
    dev4_cut_data = X4[-200:]
    X4 = X4[:-200]
    dev4_cut_labels = Y4[-200:]
    Y4 = Y4[:-200]
    
    dev5_cut_data = X5[-200:]
    X5 = X5[:-200]
    dev5_cut_labels = Y5[-200:]
    Y5 = Y5[:-200]
    
    dev7_cut_data = X7[-200:]
    X7 = X7[:-200]
    dev7_cut_labels = Y7[-200:]
    Y7 = Y7[:-200]
    
    dev8_cut_data = X8[-200:]
    X8 = X8[:-200]
    dev8_cut_labels = Y8[-200:]
    Y8 = Y8[:-200]
    
    dev10_cut_data = X10[-200:]
    X10 = X10[:-200]
    dev10_cut_labels = Y10[-200:]
    Y10 = Y10[:-200]
    
    dev10_p2_cut_data = X10_p2[-200:]
    X10_p2 = X10_p2[:-200]
    dev10_p2_cut_labels = Y10_p2[-200:]
    Y10_p2 = Y10_p2[:-200]
    
    dev1_old_cut_data = X1_old[-200:]
    X1_old = X1_old[:-200]
    dev1_old_cut_labels = Y1_old[-200:]
    Y1_old = Y1_old[:-200]
    
    dev2_old_cut_data = X2_old[-200:]
    X2_old = X2_old[:-200]
    dev2_old_cut_labels = Y2_old[-200:]
    Y2_old = Y2_old[:-200]
    
    dev3_old_cut_data = X3_old[-200:]
    X3_old = X3_old[:-200]
    dev3_old_cut_labels = Y3_old[-200:]
    Y3_old = Y3_old[:-200]
    
    dev4_old_cut_data = X4_old[-200:]
    X4_old = X4_old[:-200]
    dev4_old_cut_labels = Y4_old[-200:]
    Y4_old = Y4_old[:-200]
    
    dev5_old_cut_data = X5_old[-200:]
    X5_old = X5_old[:-200]
    dev5_old_cut_labels = Y5_old[-200:]
    Y5_old = Y5_old[:-200]
    
    dev6_old_cut_data = X6_old[-200:]
    X6_old = X6_old[:-200]
    dev6_old_cut_labels = Y6_old[-200:]
    Y6_old = Y6_old[:-200]
    
    dev7_old_cut_data = X7_old[-200:]
    X7_old = X7_old[:-200]
    dev7_old_cut_labels = Y7_old[-200:]
    Y7_old = Y7_old[:-200]
    
    dev10_old_cut_data = X10_old[-200:]
    X10_old = X10_old[:-200]
    dev10_old_cut_labels = Y10_old[-200:]
    Y10_old = Y10_old[:-200]
    
    holdout_data = np.concatenate((dev1_cut_data, dev2_cut_data, dev3_cut_data, dev4_cut_data, dev5_cut_data, dev7_cut_data,
                                   dev8_cut_data, dev10_cut_data, dev10_p2_cut_data, dev1_old_cut_data, dev2_old_cut_data, dev3_old_cut_data, 
                                   dev4_old_cut_data, dev5_old_cut_data, dev6_old_cut_data, dev7_old_cut_data, dev10_old_cut_data), axis=0).astype(np.float32)
    holdout_labels = np.concatenate((dev1_cut_labels, dev2_cut_labels, dev3_cut_labels, dev4_cut_labels, 
                                     dev5_cut_labels, dev7_cut_labels, dev8_cut_labels, dev10_cut_labels, dev10_p2_cut_labels, dev1_old_cut_labels, dev2_old_cut_labels, 
                                     dev3_old_cut_labels, dev4_old_cut_labels, dev5_old_cut_labels,
                                     dev6_old_cut_labels, dev7_old_cut_labels,
                                     dev10_old_cut_labels), axis=0)

    # Concatenate data from different CSV files
    X_dataset = np.concatenate((X1, X2, X3, X4, X5, X7, X8, X10, X10_p2, X1_old, 
                                X2_old, X3_old, X4_old, X5_old, X6_old, X7_old, X10_old), axis=0)
    Y_dataset = np.concatenate((Y1, Y2, Y3, Y4, Y5, Y7, Y8, Y10, Y10_p2, Y1_old, 
                                Y2_old, Y3_old, Y4_old, Y5_old, Y6_old, Y7_old, Y10_old), axis=0)

    # Train and evaluate the model
    result, model = train_and_evaluate(X_dataset, Y_dataset)
    
    # Print the results
    print("\nResults:")
    for key, value in result.items():
        print(f"{key}: {value}")

    # Save the model
    torch.save(model, "saved_models/puf_classifier_v7.pth")

    # Perform extra validation and print the results
    holdout_accuracy = extra_validation(model, holdout_data, holdout_labels)
    print(f"Validation accuracy on cut holdout data: {holdout_accuracy:.2f}%")
    
    # Generate confusion matrix with dimensions being the devices
    # Generate predictions for the holdout set
    list_of_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    cm_label_mapping = {0: 'D0', 1: 'D1', 2: 'D2', 3: 'D3', 4: 'D4', 5: 'D5', 6: 'D6', 7: 'D7', 8: 'D8', 9: 'D9', 
                        10: 'D10', 11: 'D11', 12: 'D12', 13: 'D13', 14: 'D14', 15: 'D15', 16: 'D16'}
    cm_labels_str = [cm_label_mapping[label] for label in list_of_labels]
    preds = extra_validation_preds(model, holdout_data)

    # True labels for the holdout set
    labels = holdout_labels 

    # Generate confusion matrix for each set
    cm = confusion_matrix(labels, preds, labels=list_of_labels)

    # Plotting and saving the confusion matrices
    save_directory = 'figures/confusion_matrices'
    plot_confusion_matrix(cm, "Confusion Matrix - Multiclass Model", "cm_multiclass_model.png", save_directory) 