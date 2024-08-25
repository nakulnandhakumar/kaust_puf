#!/bin/sh

#SBATCH --job-name=noise_sweep
#SBATCH --mail-user=nakul.nandhakumar@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=07:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gpus=8
#SBATCH --gpus-per-node=8
#SBATCH --constraint=v100&gpu_ai
#SBATCH --error=error/noise_sweep.err 
#SBATCH --output=output/noise_sweep.out

# Run the application:
srun -c 32 python sweep/puf_classifier_noise_sweep.py