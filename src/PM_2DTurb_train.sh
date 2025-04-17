#!/bin/bash -l                                                          
#SBATCH --time=1-02:00:00                                                     
#SBATCH -C 'gpu&hbm40g'
#SBATCH --account=m4416                                         
#SBATCH -q regular                                                      
#SBATCH --nodes=1                                                     
#SBATCH --ntasks-per-node=4                                             
#SBATCH --gpus-per-node=4                                               
#SBATCH --cpus-per-task=8
#SBATCH --module=gpu,nccl-2.18                                          
#SBATCH -o /pscratch/sd/d/dpp94/Logfiles/2DTurb_%x.out
#SBATCH --mail-type=begin,end,fail                                      
#SBATCH --mail-user=dpatel505@gmail.com


# COMMAND: sbatch -J <job_name> PM_2DTurb_train.sh <run_num> <yaml_config> <config>
# <job_name>: Job name used by slurm
# <run_num>: run_num used to create expDir to store all run details
# <yaml_config>: absolute path for YAML config file
# <config>: config name

set -x

cd /global/homes/d/dpp94/SSL-for-2D-Turbulence/src

# Activate conda env
ml conda
conda activate 2DTurbEmulator

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