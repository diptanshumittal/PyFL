#!/bin/bash
#SBATCH --ntasks=1              # number of tasks
##SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12      # number of CPU cores per process
#SBATCH --partition=gpuextralong         # put the job into the gpu partition/queue
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --mem=200G              # how much CPU memory can be allocated for the job (hardware limit: 384 GB per node)
#SBATCH --time=20-20:00:00         # maximum wall time allocated for the job (max 24h for the gpu partition)
#SBATCH --job-name=imagenet_testing        # job name (default is the name of this file)
ml fosscuda/2019a
PYTHON=/home/chattbap/anaconda3/bin/python
mpirun -np 1 $PYTHON model/pytorch_model_trainer.py