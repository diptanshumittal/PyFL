#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12    
#SBATCH --gpu-bind=verbose,single:1
#SBATCH --partition=gpu   
#SBATCH --output=data/logs/job_%j.out   
#SBATCH --error=data/logs/job_%j.err
#SBATCH --mem=100G         
#SBATCH --time=23:59:00      
#SBATCH --job-name=fed_clients        
ml fosscuda/2019a 
PYTHON=/home/chattbap/anaconda3/bin/python
srun $PYTHON start_clients.py --type=batchscript --clients=10

wait