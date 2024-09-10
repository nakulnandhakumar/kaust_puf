#!/bin/sh

#SBATCH --job-name=corr_coeff
#SBATCH --mail-user=nakul.nandhakumar@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=07:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --constraint=v100&gpu_ai
#SBATCH --error=error/corr_coeff.err 
#SBATCH --output=output/corr_coeff.out

# Run the application:
srun -c 32 python corr_coeff.py