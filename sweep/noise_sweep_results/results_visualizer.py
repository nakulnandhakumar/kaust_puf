import pandas as pd
import matplotlib.pyplot as plt

# List of CSV file paths to read data from
csv_files = [
    "sweep/noise_sweep_results/val_results_broad.csv",
    "sweep/noise_sweep_results/val_results_narrow.csv",
    "sweep/noise_sweep_results/train_val_results.csv"
]

# Read data into individual DataFrames
val_narrow_data = pd.read_csv(csv_files[0])
val_broad_data = pd.read_csv(csv_files[1])
train_val_broad_data = pd.read_csv(csv_files[2])

# Plot model accuracy vs noise percentage for broad sweep of model with noise just added to validation data
plt.figure(figsize=(10, 6))
plt.plot(val_broad_data["Noise Percentage"], val_broad_data["Accuracy on Device 7 Cut"], label="Unseen Data from Real Device")
plt.title("Model Accuracy vs Percent Noise Added to Val Data")
plt.xlabel("Percent Noise Added")
plt.ylabel("Model Accuracy")
plt.legend()
plt.savefig("figures/noise_sweep/val_accuracy_vs_noise_percentage_broad.png")

# Plot model accuracy vs noise percentage for broad sweep of model with noise just added to validation data
plt.figure(figsize=(10, 6))
plt.plot(val_narrow_data["Noise Percentage"], val_narrow_data["Accuracy on Device 7 Cut"], label="Unseen Data from Real Device")
plt.title("Model Accuracy vs Percent Noise Added to Val Data")
plt.xlabel("Percent Noise Added")
plt.ylabel("Model Accuracy")
plt.legend()
plt.savefig("figures/noise_sweep/val_accuracy_vs_noise_percentage_narrow.png")

# Plot model accuracy vs noise percentage for broad sweep of model with noise added to both training and validation data
plt.figure(figsize=(10, 6))
plt.plot(train_val_broad_data["Noise Percentage"], train_val_broad_data["Accuracy on Device 7 Cut"], label="Unseen Data from Real Device")
plt.title("Model Accuracy vs Percent Noise Added to Train & Val Data")
plt.xlabel("Percent Noise Added")
plt.ylabel("Model Accuracy")
plt.legend()
plt.savefig("figures/noise_sweep/train_val_accuracy_vs_noise_percentage.png")