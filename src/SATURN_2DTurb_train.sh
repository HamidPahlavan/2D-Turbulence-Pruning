# COMMAND: bash SATURN_2DTurb_train.sh <run_num> <yaml_config> <config>
# <job_name>: Job name used by slurm
# <run_num>: run_num used to create expDir to store all run details
# <yaml_config>: absolute path for YAML config file
# <config>: config name

set -x

# ------ Define all DDP vars ------ #
#source export_DDP_vars.sh
export NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
#export WORLD_SIZE=${SLURM_NTASKS}
export OMP_NUM_THREADS=1

# ------ WANDB ------ #

source /home/jovyan/shared/dpp94/ssl-for-2d-turbulence-2/set_wandb_key_dpp94.sh
export WANDB_MODE=online


# ------ Run main script ------ #

torchrun --nproc_per_node=${NUM_TASKS_PER_NODE} train.py --yaml_config=${2} --config=${3} --run_num=${1} --fresh_start