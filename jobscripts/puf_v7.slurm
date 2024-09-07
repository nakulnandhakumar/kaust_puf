#!/bin/sh

#SBATCH --job-name=puf_v7
#SBATCH --mail-user=nakul.nandhakumar@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=07:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gpus=8
#SBATCH --gpus-per-node=8
#SBATCH --constraint=v100&gpu_ai
#SBATCH --error=error/puf_v7.err 
#SBATCH --output=output/puf_v7.out

# Run the application:
srun -c 32 python puf_classifier_v7.py