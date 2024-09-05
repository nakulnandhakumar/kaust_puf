import pandas as pd
import matplotlib.pyplot as plt

# List of CSV file paths to read data from
csv_files = [
    "sweep/sequence_size_sweep_results/results1.csv",
    "sweep/sequence_size_sweep_results/results2.csv",
    "sweep/sequence_size_sweep_results/results3.csv",
    "sweep/sequence_size_sweep_results/results4.csv",
    "sweep/sequence_size_sweep_results/results5.csv",
    "sweep/sequence_size_sweep_results/results6.csv",
]

# Read data into a single DataFrame
data_frames = [pd.read_csv(file) for file in csv_files]
data = pd.concat(data_frames)

# With sequence size on the X-axis, plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(data["Sequence Size"], data["Train Loss"], label="Train Loss")
plt.plot(data["Sequence Size"], data["Validation Loss"], label="Validation Loss")
plt.title("Model Loss vs Sequence Size")
plt.xlabel("Sequence Size")
plt.ylabel("Loss")
plt.legend()
plt.savefig("figures/sequence_sweep/loss_vs_sequence_size.png")

# Plot Sequence Size vs training and validation accuracy scores on separate figure
plt.figure(figsize=(10, 6))
plt.plot(data["Sequence Size"], data["Train Accuracy"], label="Train Accuracy")
plt.plot(data["Sequence Size"], data["Validation Accuracy"], label="Validation Accuracy")
plt.plot(data["Sequence Size"], data["Accuracy on Device 10 Probe-Again"], label="Fake Device Probed Again Data")
plt.plot(data["Sequence Size"], data["Accuracy on Device 7 Cut"], label="Unseen Data from Real Device")
plt.title("Model Accuracy vs Sequence Size")
plt.xlabel("Sequence Size")
plt.ylabel("Percent Accuracy")
plt.legend()
plt.savefig("figures/sequence_sweep/accuracy_vs_sequence_size.png")