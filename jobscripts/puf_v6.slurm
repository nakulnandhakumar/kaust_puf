#!/bin/sh

#SBATCH --job-name=puf_v6
#SBATCH --mail-user=nakul.nandhakumar@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=07:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gpus=8
#SBATCH --gpus-per-node=8
#SBATCH --constraint=v100&gpu_ai
#SBATCH --error=error/puf_v6.err 
#SBATCH --output=output/puf_v6.out

# Run the application:
srun -c 32 python puf_classifier_v6.py