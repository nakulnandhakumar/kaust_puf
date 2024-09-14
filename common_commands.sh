# Initial IBEX Terminal Setup
conda activate base
cd /ibex/user/nandhan/kaust_puf


# SCP Commands
scp Documents/KAUST\ Internship/kaust_puf/puf_classifier_v7.py ibex:/ibex/user/nandhan/kaust_puf
scp Documents/KAUST\ Internship/kaust_puf/validation.py ibex:/ibex/user/nandhan/kaust_puf
scp Documents/KAUST\ Internship/kaust_puf/jobscripts/validation.slurm ibex:/ibex/user/nandhan/kaust_puf/jobscripts
scp Documents/KAUST\ Internship/kaust_puf/gan_v1.py ibex:/ibex/user/nandhan/kaust_puf
scp Documents/KAUST\ Internship/kaust_puf/vae_v1.py ibex:/ibex/user/nandhan/kaust_puf
scp Documents/KAUST\ Internship/kaust_puf/corr_coeff.py ibex:/ibex/user/nandhan/kaust_puf
scp Documents/KAUST\ Internship/kaust_puf/sweep/puf_classifier_sequence_sweep.py ibex:/ibex/user/nandhan/kaust_puf/sweep
scp Documents/KAUST\ Internship/kaust_puf/sweep/puf_classifier_noise_sweep.py ibex:/ibex/user/nandhan/kaust_puf/sweep
scp Documents/KAUST\ Internship/kaust_puf/saved_models/vae_v1.pth ibex:/ibex/user/nandhan/kaust_puf/saved_models
scp Documents/KAUST\ Internship/kaust_puf/saved_models/gan_generator_v1.pth ibex:/ibex/user/nandhan/kaust_puf/saved_models
scp -r Documents/KAUST\ Internship/kaust_puf/sweep ibex:/ibex/user/nandhan/kaust_puf
scp -r Documents/KAUST\ Internship/kaust_puf/jobscripts ibex:/ibex/user/nandhan/kaust_puf
scp -r Documents/KAUST\ Internship/kaust_puf/saved_models ibex:/ibex/user/nandhan/kaust_puf
scp -r Documents/KAUST\ Internship/kaust_puf/figures ibex:/ibex/user/nandhan/kaust_puf

scp ibex:/ibex/user/nandhan/kaust_puf/sweep/sequence_size_sweep_results/results3.csv Documents/KAUST\ Internship/kaust_puf/sweep/sequence_size_sweep_results 
scp ibex:/ibex/user/nandhan/kaust_puf/sweep/noise_sweep_results/results.csv Documents/KAUST\ Internship/kaust_puf/sweep/noise_sweep_results 
scp ibex:/ibex/user/nandhan/kaust_puf/saved_models/vae_v1.pth Documents/KAUST\ Internship/kaust_puf/saved_models 
scp ibex:/ibex/user/nandhan/kaust_puf/saved_models/gan_generator_v1.pth Documents/KAUST\ Internship/kaust_puf/saved_models 
scp -r ibex:/ibex/user/nandhan/kaust_puf/sweep/sequence_size_sweep_results Documents/KAUST\ Internship/kaust_puf/sweep 
scp -r ibex:/ibex/user/nandhan/kaust_puf/sweep/noise_sweep_results Documents/KAUST\ Internship/kaust_puf/sweep 
scp -r ibex:/ibex/user/nandhan/kaust_puf/saved_models Documents/KAUST\ Internship/kaust_puf
scp -r ibex:/ibex/user/nandhan/kaust_puf/figures/confusion_matrices Documents/KAUST\ Internship/kaust_puf/figures


# SLURM Commands
sbatch jobscripts/sequence_sweep1.slurm
squeue -u nandhan
scancel -u nandhan
