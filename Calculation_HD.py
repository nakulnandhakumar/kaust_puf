import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes

file_name = "Demo: File_Path.csv"
data = pd.read_csv(file_name)

raw_sequences = data.iloc[1:1000001, 4].values.reshape(1000, 1000)

n_segments = 10
segment_length = 1000 // n_segments

binary_data = []
for sample in raw_sequences:
    discretized = ((sample + 1) * 127.5).astype(np.uint8)  
    binary_representation = np.unpackbits(discretized[:, np.newaxis], axis=1).reshape(-1) 
    binary_data.append(binary_representation)

binary_data = np.array(binary_data)

intra_distances = []
inter_distances = []

for sample in binary_data:
    segments = np.array_split(sample, n_segments)
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            distance = np.sum(segments[i] != segments[j])
            intra_distances.append(distance)

for i in range(len(binary_data)):
    for j in range(i + 1, len(binary_data)):
        distance = np.sum(binary_data[i] != binary_data[j])
        inter_distances.append(distance)

sequence_bit_length = 1000 * 8  
intra_distances = np.array(intra_distances) / sequence_bit_length
inter_distances = np.array(inter_distances) / sequence_bit_length

intra_hd_df = pd.DataFrame({"Intra-HD": intra_distances})
inter_hd_df = pd.DataFrame({"Inter-HD": inter_distances})
output_data = pd.concat([intra_hd_df, inter_hd_df], axis=1)
output_data.to_csv("outputHD_c5.csv", index=False)


plt.rcParams['font.family'] = 'Arial'
fig = plt.figure(figsize=(12,10))
bax = brokenaxes(xlims=((0, 0.1), (0.3, 0.6)), hspace=0.001, despine=False)  

bax.hist(intra_distances, bins=10, alpha=0.7, label='Intra-HD', density=True, color='#A6BEE1',
         histtype='bar', edgecolor='#006BAC', linewidth=1.2)
bax.hist(inter_distances, bins=40, alpha=0.7, label='Inter-HD', density=True, color='#BFA6D9',
         histtype='bar', edgecolor='#490093', linewidth=1.2)

bax.axvline(np.mean(intra_distances), color='#006BAC', linestyle='--', label=f'Intra-HD Mean={np.mean(intra_distances):.4f}', linewidth=2.5)
bax.tick_params(axis='both', labelsize=35)  

bax.axvline(np.mean(inter_distances), color='#490093', linestyle='--', label=f'Inter-HD Mean={np.mean(inter_distances):.4f}', linewidth=2.5)
bax.tick_params(axis='both', labelsize=35)  

bax.set_xlabel('HD (Norm.)', labelpad=40, fontsize=45)

bax.legend(fontsize=38)
bax.set_ylabel('Counts',labelpad=40, fontsize=45)

plt.show()

print(f"Intra-HD Mean: {np.mean(intra_distances):.4f}")
print(f"Inter-HD Mean: {np.mean(inter_distances):.4f}")
