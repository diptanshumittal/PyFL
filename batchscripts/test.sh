#!/bin/bash
#SBATCH --ntasks=8           # number of tasks
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12      # number of CPU cores per process
#SBATCH --gpu-bind=verbose,single:1
#SBATCH --partition=gpufast         # put the job into the gpu partition/queue
#SBATCH --output=data/logs/job_%j.out   
#SBATCH --error=data/logs/job_%j.err
#SBATCH --mem=100G              # how much CPU memory can be allocated for the job (hardware limit: 384 GB per node)
#SBATCH --time=3:00:00         # maximum wall time allocated for the job (max 24h for the gpu partition)
#SBATCH --job-name=fed_clients        # job name (default is the name of this file)
ml fosscuda/2019a 
PYTHON=/home/chattbap/anaconda3/bin/python
srun $PYTHON start_clients.py 
# mpirun -np 1 $PYTHON test.py &
# mpirun -np 1 $PYTHON test.py &
# # mpirun -np 1 $PYTHON test.py &
# # mpirun -np 1 $PYTHON test.py &
# mpirun -np 1 $PYTHON test.py 


wait






# SBATCH --nodes=2               # number of nodes
# SBATCH --ntasks-per-node=4     # processes per node
# SBATCH --cpus-per-task=12       # number of CPU cores per process
# #SBATCH --gres=gpu:4
# SBATCH --gpus-per-node=3



# SBATCH --ntasks=4           # number of tasks
# SBATCH --gpus-per-task=1
# SBATCH --cpus-per-task=12      # number of CPU cores per process
# SBATCH --gpu-bind=verbose,map_gpu:0,1,2,3



# SBATCH --nodes=1               # number of nodes
# SBATCH --ntasks-per-node=72     # processes per node
# #SBATCH --cpus-per-task=72       # number of CPU cores per process
# SBATCH --gres=gpu:4            # GPUs per node 