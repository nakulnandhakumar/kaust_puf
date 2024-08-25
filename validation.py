import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from puf_classifier_v6 import CNN

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

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load in puf_classifier_v6 PyTorch model
model = torch.load("saved_models/puf_classifier_v6.pth", map_location=device)

sequence_size = 10000

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

d50_point_1mA_accuracy = extra_validation(model, X50_point_1mA, Y50_point_1mA)
d50_point_2mA_accuracy = extra_validation(model, X50_point_2mA, Y50_point_2mA)
d50_point_3mA_accuracy = extra_validation(model, X50_point_3mA, Y50_point_3mA)
d50_point_4mA_accuracy = extra_validation(model, X50_point_4mA, Y50_point_4mA)
d50_point_5mA_accuracy = extra_validation(model, X50_point_5mA, Y50_point_5mA)
d51mA_accuracy = extra_validation(model, X51mA, Y51mA)
d52mA_accuracy = extra_validation(model, X52mA, Y52mA)
d53mA_accuracy = extra_validation(model, X53mA, Y53mA)
d54mA_accuracy = extra_validation(model, X54mA, Y54mA)
d55mA_accuracy = extra_validation(model, X55mA, Y55mA)
d56mA_accuracy = extra_validation(model, X56mA, Y56mA)
d57mA_accuracy = extra_validation(model, X57mA, Y57mA)
d58mA_accuracy = extra_validation(model, X58mA, Y58mA)
d59mA_accuracy = extra_validation(model, X59mA, Y59mA)
d60mA_accuracy = extra_validation(model, X60mA, Y60mA)

print("\nValidation Accuracy on Devices with Different Currents:")
print(f"Validation accuracy on 50.1mA: {d50_point_1mA_accuracy:.2f}%")
print(f"Validation accuracy on 50.2mA: {d50_point_2mA_accuracy:.2f}%")
print(f"Validation accuracy on 50.3mA: {d50_point_3mA_accuracy:.2f}%")
print(f"Validation accuracy on 50.4mA: {d50_point_4mA_accuracy:.2f}%")
print(f"Validation accuracy on 50.5mA: {d50_point_5mA_accuracy:.2f}%")
print(f"Validation accuracy on 51mA: {d51mA_accuracy:.2f}%")
print(f"Validation accuracy on 52mA: {d52mA_accuracy:.2f}%")
print(f"Validation accuracy on 53mA: {d53mA_accuracy:.2f}%")
print(f"Validation accuracy on 54mA: {d54mA_accuracy:.2f}%")
print(f"Validation accuracy on 55mA: {d55mA_accuracy:.2f}%")
print(f"Validation accuracy on 56mA: {d56mA_accuracy:.2f}%")
print(f"Validation accuracy on 57mA: {d57mA_accuracy:.2f}%")
print(f"Validation accuracy on 58mA: {d58mA_accuracy:.2f}%")
print(f"Validation accuracy on 59mA: {d59mA_accuracy:.2f}%")
print(f"Validation accuracy on 60mA: {d60mA_accuracy:.2f}%")