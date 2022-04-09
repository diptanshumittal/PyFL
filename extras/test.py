import torch
import os

print("Total visible devices",torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES",os.environ["CUDA_VISIBLE_DEVICES"])
print("SLURM_JOB_GPUS",os.environ["SLURM_JOB_GPUS"])
if 'OMPI_COMM_WORLD_RANK' in os.environ:
    print("OMPI_COMM_WORLD_RANK")
    print(os.environ['OMPI_COMM_WORLD_RANK'])
    print(os.environ['OMPI_COMM_WORLD_SIZE'])
if 'SLURM_PROCID' in os.environ:  
    print("SLURM_PROCID")      
    print("SLURM_PROCID",os.environ['SLURM_PROCID'])
    print("SLURM_NTASKS",os.environ['SLURM_NTASKS'])