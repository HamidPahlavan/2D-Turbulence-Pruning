#!/bin/bash -l
#PBS -l select=1:ncpus=64:ngpus=8
#PBS -q by-gpu
#PBS -l filesystems=home:eagle
#PBS -l walltime=24:00:00
#PBS -A AI-S2S
#PBS -M dpatel505@gmail.com
#PBS -m abe
#PBS -j oe

# Enable GPU-MPI (if supported by application)
export MPICH_GPU_SUPPORT_ENABLED=1


# COMMAND: qsub -N <job_name> -v run_num=<run_num>,yaml_config=<yaml_config>,config=<config> SOPHIA_2DTurb_train.sh
# <job_name>: Job name used by slurm
# <run_num>: run_num used to create expDir to store all run details
# <yaml_config>: absolute path for YAML config file
# <config>: config name

set -x

cd /eagle/MDClimSim/dpp94/SSL-for-2D-Turbulence/src

# Activate conda env
ml conda
#conda activate base
source /lus/eagle/projects/MDClimSim/dpp94/venvs/2DTurbEmulator/bin/activate
module list

# ------ Define all DDP vars ------ #
# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
#Follwing will be the number of GPUs on each node, so 4 in our case as each node has 4 GPUs
NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
WORLD_SIZE=$((NNODES * NUM_TASKS_PER_NODE))

echo "NUM_OF_NODES= ${NNODES} NUM_TASKS_PER_NODE= ${NUM_TASKS_PER_NODE} WORLD_SIZE= ${WORLD_SIZE}"
which python

# ------ WANDB ------ #

source $HOME/set_wandb_key_dpp94.sh
export WANDB_MODE=online


# ------ Run main script ------ #

config=SMAE_FINETUNE
#torchrun --nproc_per_node=${NUM_TASKS_PER_NODE} --nnodes=${NNODES} train.py --yaml_config=${yaml_config} --config=${config} --run_num=${run_num} --fresh_start
python -m torch.distributed.run --nproc_per_node=${NUM_TASKS_PER_NODE} --nnodes=${NNODES} train.py --yaml_config=${yaml_config} --config=${config} --run_num=${run_num} --fresh_start