#!/bin/sh

#SBATCH --job-name=sequence_sweep6
#SBATCH --mail-user=nakul.nandhakumar@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=07:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --constraint=v100&gpu_ai
#SBATCH --error=error/sequence_sweep6.err 
#SBATCH --output=output/sequence_sweep6.out

# Run the application:
srun -c 32 python sweep/puf_classifier_sequence_sweep6.py