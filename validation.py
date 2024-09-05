import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from puf_classifier_v6 import CNN
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

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

def extra_validation(model, test_data, test_labels):
    # Create PyTorch tensor for the real and fake test sets
    test_tensor = torch.tensor(test_data).to(device)

    # Set the model to evaluation mode
    model.eval()
    
    # Make predictions for each of the distributions on device 5 data (fake)
    predictions_test = []
    with torch.no_grad():
        for i in range(len(test_data)):
            output = model(test_tensor[i].unsqueeze(0))
            predicted = torch.round(torch.sigmoid(output))
            predictions_test.append(predicted.item())
            
    accuracy_test = 100 * np.sum(np.array(predictions_test) == test_labels) / len(test_data)
    
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
            predicted = torch.round(torch.sigmoid(output))
            predictions_test.append(predicted.item())
            
    # Convert predictions_test to a NumPy array
    predictions_test = np.array(predictions_test)
    
    # Set the model back to training mode
    model.train()
    
    # Return the results    
    return predictions_test


# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load in puf_classifier_v5 and v6 PyTorch model
model_v5 = torch.load("saved_models/puf_classifier_v5.pth", map_location=device)
model_v6 = torch.load("saved_models/puf_classifier_v6.pth", map_location=device)


# Load in all data required for ezxtra validation and confusion matrix generation
df0 = pd.read_csv("puf_dataset_08_19/50mA.csv")
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
df50_point_1mA = pd.read_csv("puf_dataset_08_19/50.1mA.csv")
df50_point_2mA = pd.read_csv("puf_dataset_08_19/50.2mA.csv")
df50_point_3mA = pd.read_csv("puf_dataset_08_19/50.3mA.csv")
df50_point_4mA = pd.read_csv("puf_dataset_08_19/50.4mA.csv")
df50_point_5mA = pd.read_csv("puf_dataset_08_19/50.5mA.csv")
df51mA = pd.read_csv("puf_dataset_08_19/51mA.csv")
df52mA = pd.read_csv("puf_dataset_08_19/52mA.csv")
df53mA = pd.read_csv("puf_dataset_08_19/53mA.csv")
df54mA = pd.read_csv("puf_dataset_08_19/54mA.csv")
df55mA = pd.read_csv("puf_dataset_08_19/55mA.csv")
df56mA = pd.read_csv("puf_dataset_08_19/56mA.csv")
df57mA = pd.read_csv("puf_dataset_08_19/57mA.csv")
df58mA = pd.read_csv("puf_dataset_08_19/58mA.csv")
df59mA = pd.read_csv("puf_dataset_08_19/59mA.csv")
df60mA = pd.read_csv("puf_dataset_08_19/60mA.csv")
df1_07_08 = pd.read_csv("puf_dataset_07_08/Can-D1-50mA2.csv")
df2_07_08 = pd.read_csv("puf_dataset_07_08/Can-D2-50mA2.csv")
df3_07_08 = pd.read_csv("puf_dataset_07_08/Can-D3-50mA.csv")
df4_07_08 = pd.read_csv("puf_dataset_07_08/Can-D4-50mA.csv")

sequence_size = 10000
X0 = create_dataset(df0, sequence_size)
X1 = create_dataset(df1, sequence_size)  
X2 = create_dataset(df2, sequence_size)
X3 = create_dataset(df3, sequence_size)
X4 = create_dataset(df4, sequence_size)
X5 = create_dataset(df5, sequence_size)
X7 = create_dataset(df7, sequence_size)
X8 = create_dataset(df8, sequence_size)
X50_point_1mA = create_dataset(df50_point_1mA, sequence_size)
X50_point_2mA = create_dataset(df50_point_2mA, sequence_size)
X50_point_3mA = create_dataset(df50_point_3mA, sequence_size)
X50_point_4mA = create_dataset(df50_point_4mA, sequence_size)
X50_point_5mA = create_dataset(df50_point_5mA, sequence_size)
X51mA = create_dataset(df51mA, sequence_size)
X52mA = create_dataset(df52mA, sequence_size)
X53mA = create_dataset(df53mA, sequence_size)
X54mA = create_dataset(df54mA, sequence_size)
X55mA = create_dataset(df55mA, sequence_size)
X56mA = create_dataset(df56mA, sequence_size)
X57mA = create_dataset(df57mA, sequence_size)
X58mA = create_dataset(df58mA, sequence_size)
X59mA = create_dataset(df59mA, sequence_size)
X60mA = create_dataset(df60mA, sequence_size)
X1_unseen = create_dataset(df1_07_08, sequence_size)
X2_unseen = create_dataset(df2_07_08, sequence_size)
X3_unseen = create_dataset(df3_07_08, sequence_size)
X4_unseen = create_dataset(df4_07_08, sequence_size)

Y0 = np.ones(len(X0)).astype(np.float32)
Y1 = np.zeros(len(X1)).astype(np.float32)
Y2 = np.zeros(len(X2)).astype(np.float32)
Y3 = np.zeros(len(X3)).astype(np.float32)
Y4 = np.zeros(len(X4)).astype(np.float32)
Y5 = np.zeros(len(X5)).astype(np.float32)
Y7 = np.ones(len(X7)).astype(np.float32)
Y8 = np.zeros(len(X8)).astype(np.float32)
Y50_point_1mA = np.zeros(len(X50_point_1mA)).astype(np.float32)
Y50_point_2mA = np.zeros(len(X50_point_2mA)).astype(np.float32)
Y50_point_3mA = np.zeros(len(X50_point_3mA)).astype(np.float32)
Y50_point_4mA = np.zeros(len(X50_point_4mA)).astype(np.float32)
Y50_point_5mA = np.zeros(len(X50_point_5mA)).astype(np.float32)
Y51mA = np.zeros(len(X51mA)).astype(np.float32)
Y52mA = np.zeros(len(X52mA)).astype(np.float32)
Y53mA = np.zeros(len(X53mA)).astype(np.float32)
Y54mA = np.zeros(len(X54mA)).astype(np.float32)
Y55mA = np.zeros(len(X55mA)).astype(np.float32)
Y56mA = np.zeros(len(X56mA)).astype(np.float32)
Y57mA = np.zeros(len(X57mA)).astype(np.float32)
Y58mA = np.zeros(len(X58mA)).astype(np.float32)
Y59mA = np.zeros(len(X59mA)).astype(np.float32)
Y60mA = np.zeros(len(X60mA)).astype(np.float32)
Y1_unseen = np.zeros(len(X1_unseen)).astype(np.float32)
Y2_unseen = np.zeros(len(X2_unseen)).astype(np.float32)
Y3_unseen = np.zeros(len(X3_unseen)).astype(np.float32)
Y4_unseen = np.zeros(len(X4_unseen)).astype(np.float32)

# Create datasets for extra validation for multichallenge current model v6
# Cut data from seen fake currents 50.1, 50.2, 50.3, 50.4, 50.5 and see if the model recognizes the data as fake
X50_point_1mA_data = X50_point_1mA[-200:]
X50_point_1mA = X50_point_1mA[:-200]
Y50_point_1mA = Y50_point_1mA[:-200]
X50_point_2mA_data = X50_point_2mA[-200:]
X50_point_2mA = X50_point_2mA[:-200]
Y50_point_2mA = Y50_point_2mA[:-200]
X50_point_3mA_data = X50_point_3mA[-200:]
X50_point_3mA = X50_point_3mA[:-200]
Y50_point_3mA = Y50_point_3mA[:-200]
X50_point_4mA_data = X50_point_4mA[-200:]
X50_point_4mA = X50_point_4mA[:-200]
Y50_point_4mA = Y50_point_4mA[:-200]
X50_point_5mA_data = X50_point_5mA[-200:]
X50_point_5mA = X50_point_5mA[:-200]
Y50_point_5mA = Y50_point_5mA[:-200]
fake_seen_dev_cut_data_v6 = np.concatenate((X50_point_1mA_data, X50_point_2mA_data, X50_point_3mA_data, X50_point_4mA_data, X50_point_5mA_data), axis=0).astype(np.float32)
fake_seen_dev_cut_labels_v6 = np.zeros(len(fake_seen_dev_cut_data_v6)).astype(np.float32)

# Cut data from unseen fake currents 51, 52, 53, 54, 55 and see if the model recognizes the data as fake
X51mA_data = X51mA[-200:]
X51mA = X51mA[:-200]
Y51mA = Y51mA[:-200]
X52mA_data = X52mA[-200:]
X52mA = X52mA[:-200]
Y52mA = Y52mA[:-200]
X53mA_data = X53mA[-200:]
X53mA = X53mA[:-200]
Y53mA = Y53mA[:-200]
X54mA_data = X54mA[-200:]
X54mA = X54mA[:-200]
Y54mA = Y54mA[:-200]
X55mA_data = X55mA[-200:]
X55mA = X55mA[:-200]
Y55mA = Y55mA[:-200]
fake_unseen_dev_cut_data_v6 = np.concatenate((X51mA_data, X52mA_data, X53mA_data, X54mA_data, X55mA_data), axis=0).astype(np.float32)
fake_unseen_dev_cut_labels_v6 = np.zeros(len(fake_unseen_dev_cut_data_v6)).astype(np.float32)

# Cut data from real seen device 0 current 50mA and see if the model recognizes the data as real
real_seen_dev_cut_data_v6 = X0[-1000:]
X0 = X0[:-1000]
real_seen_dev_cut_labels_v6 = Y0[-1000:]
Y0 = Y0[:-1000]

# Create datasets for extra validation for single challenge current model v5
# Cut data from seen fake devices 1,2,3,4,8 and see if the model recognizes the data as fake
dev1_cut_data = X1[-200:]
X1 = X1[:-200]
Y1 = Y1[:-200]
dev2_cut_data = X2[-200:]
X2 = X2[:-200]
Y2 = Y2[:-200]
dev3_cut_data = X3[-200:]
X3 = X3[:-200]
Y3 = Y3[:-200]
dev4_cut_data = X4[-200:]
X4 = X4[:-200]
Y4 = Y4[:-200]
dev8_cut_data = X8[-200:]
X8 = X8[:-200]
Y8 = Y8[:-200]
fake_seen_dev_cut_data_v5 = np.concatenate((dev1_cut_data, dev2_cut_data, dev3_cut_data, dev4_cut_data, dev8_cut_data), axis=0).astype(np.float32)
fake_seen_dev_cut_labels_v5 = np.zeros(len(fake_seen_dev_cut_data_v5)).astype(np.float32)

# Take fake unseen device data from device 5, and from first data set, dev 1, 2, 3, 4
unseen_dev5_cut_data = X5[-200:]
X5 = X5[:-200]
Y5 = Y5[:-200]
dev1_unseen_cut_data = X1_unseen[-200:]
X1_unseen = X1_unseen[:-200]
Y1_unseen = Y1_unseen[:-200]
dev2_unseen_cut_data = X2_unseen[-200:]
X2_unseen = X2_unseen[:-200]
Y2_unseen = Y2_unseen[:-200]
dev3_unseen_cut_data = X3_unseen[-200:]
X3_unseen = X3_unseen[:-200]
Y3_unseen = Y3_unseen[:-200]
dev4_unseen_cut_data = X4_unseen[-200:]
X4_unseen = X4_unseen[:-200]
Y4_unseen = Y4_unseen[:-200]
fake_unseen_dev_cut_data_v5 = np.concatenate((unseen_dev5_cut_data, dev1_unseen_cut_data, dev2_unseen_cut_data, dev3_unseen_cut_data, dev4_unseen_cut_data), axis=0).astype(np.float32)
fake_unseen_dev_cut_labels_v5 = np.zeros(len(fake_unseen_dev_cut_data_v5)).astype(np.float32)

# Also reserve some data from device 7 for testing, see if it recognizes the real device
real_seen_dev_cut_data_v5 = X7[-1000:]
X7 = X7[:-1000]
real_seen_dev_cut_labels_v5 = Y7[-1000:]
Y7 = Y7[:-1000]

# Generate confusion matrices for each individual set of validation data for mutliple currents version (v6)
# Generate predictions for each test set
preds_real_v6 = extra_validation_preds(model_v6, real_seen_dev_cut_data_v6)
preds_fake_seen_v6 = extra_validation_preds(model_v6, fake_seen_dev_cut_data_v6)
preds_fake_unseen_v6 = extra_validation_preds(model_v6, fake_unseen_dev_cut_data_v6)

# True labels for the test sets
labels_real_v6 = real_seen_dev_cut_labels_v6  # Real unseen signals (should be 1s)
labels_fake_seen_v6 = fake_seen_dev_cut_labels_v6  # Fake signals from a seen device (should be 0s)
labels_fake_unseen_v6 = fake_unseen_dev_cut_labels_v6  # Fake signals from an unseen device (should be 0s)

# Generate confusion matrix for each set
# Define the mapping from numeric labels to string labels and the numeric labels actually returned by the model
label_mapping = {0: 'Fake', 1: 'Real'}
labels = [0, 1]
labels_str = [label_mapping[label] for label in labels]

cm_real = confusion_matrix(labels_real_v6, preds_real_v6, labels=labels)
cm_fake_seen = confusion_matrix(labels_fake_seen_v6, preds_fake_seen_v6, labels=labels)
cm_fake_unseen = confusion_matrix(labels_fake_unseen_v6, preds_fake_unseen_v6, labels=labels)

# Plot the confusion matrices
def plot_confusion_matrix(cm, title, filename, save_dir):
    plt.figure()
    
    # Define vmax and vmin for the color scale of the confusion matrix
    vmin = 0
    vmax = cm.max()
    
    print(f"Confusion Matrix: {cm}")
    print(f"Confusion Matrix Shape: {cm.shape}")
    
    # Plot the confusion matrix with custom labels
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels_str, yticklabels=labels_str, vmin=vmin, vmax=vmax)
    
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the figure as a PNG file
    file_path = os.path.join(save_dir, filename)
    plt.savefig(file_path)
    plt.close()

# Define the directory where the PNGs will be saved
save_directory = 'figures/confusion_matrices'

# Plotting and saving the confusion matrices
plot_confusion_matrix(cm_real, "Confusion Matrix - Real Signals", "cm_real_current_v6.png", save_directory)
plot_confusion_matrix(cm_fake_seen, "Confusion Matrix - Fake Signals from Seen Currents", "cm_fake_seen_currents_v6.png", save_directory)
plot_confusion_matrix(cm_fake_unseen, "Confusion Matrix - Fake Signals from Unseen Currents", "cm_fake_unseen_currents_v6.png", save_directory)

# Generate confusion matrices for each individual set of validation data for single current version (v5)
# Generate predictions for each test set
preds_real_v5 = extra_validation_preds(model_v5, real_seen_dev_cut_data_v5)
preds_fake_seen_v5 = extra_validation_preds(model_v5, fake_seen_dev_cut_data_v5)
preds_fake_unseen_v5 = extra_validation_preds(model_v5, fake_unseen_dev_cut_data_v5)

# True labels for the test sets
labels_real_v5 = real_seen_dev_cut_labels_v5  # Real unseen signals (should be 1s)
labels_fake_seen_v5 = fake_seen_dev_cut_labels_v5  # Fake signals from a seen device (should be 0s)
labels_fake_unseen_v5 = fake_unseen_dev_cut_labels_v5  # Fake signals from an unseen device (should be 0s)

# Generate confusion matrix for each set
cm_real_v5 = confusion_matrix(labels_real_v5, preds_real_v5)
cm_fake_seen_v5 = confusion_matrix(labels_fake_seen_v5, preds_fake_seen_v5)
cm_fake_unseen_v5 = confusion_matrix(labels_fake_unseen_v5, preds_fake_unseen_v5)

# Plotting and saving the confusion matrices
plot_confusion_matrix(cm_real_v5, "Confusion Matrix - Real Signals", "cm_real_device_v5.png", save_directory)
plot_confusion_matrix(cm_fake_seen_v5, "Confusion Matrix - Fake Signals from Seen Devices", "cm_fake_seen_devices_v5.png", save_directory)
plot_confusion_matrix(cm_fake_unseen_v5, "Confusion Matrix - Fake Signals from Unseen Devices", "cm_fake_unseen_devices_v5.png", save_directory)