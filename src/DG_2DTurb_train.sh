#!/bin/bash
#SBATCH --output=/work/hdd/bdiu/dpatel6/Logfiles/2DTurb_%x.out
#SBATCH --account=bdiu-delta-gpu    # <- match to a "Project" returned by the "accounts" command

#SBATCH --partition=gpuA100x4
#SBATCH --mem-per-gpu=60G
#SBATCH --nodes=1
#SBATCH --ntasks=4          # could be 1 for py-torch
#SBATCH --cpus-per-task=8   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --no-requeue
#SBATCH -t 1-02:00:00

#SBATCH --mail-type=begin,end,fail                                      
#SBATCH --mail-user=dpp94@uchicago.edu 


# COMMAND: sbatch -J <job_name> PM_2DTurb_train.sh <run_num> <yaml_config> <config>
# <job_name>: Job name used by slurm
# <run_num>: run_num used to create expDir to store all run details
# <yaml_config>: absolute path for YAML config file
# <config>: config name

set -x

cd /u/dpatel6/SSL-for-2D-Turbulence/src

# Activate conda env
ml anaconda3_gpu
source activate 2DTurbEmulator


# ------ Define all DDP vars ------ #
source export_DDP_vars.sh
export NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
#export WORLD_SIZE=${SLURM_NTASKS}
export OMP_NUM_THREADS=1

# ------ WANDB ------ #

source $HOME/set_wandb_key_dpp94.sh
export WANDB_MODE=online


# ------ Run main script ------ #

torchrun --nproc_per_node=${NUM_TASKS_PER_NODE} --nnodes=${SLURM_NNODES} train.py --yaml_config=${2} --config=${3} --run_num=${1} --fresh_start
