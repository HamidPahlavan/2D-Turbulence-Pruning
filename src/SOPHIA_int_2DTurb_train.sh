# Enable GPU-MPI (if supported by application)
export MPICH_GPU_SUPPORT_ENABLED=1

set -x

cd /home/dpp94/SSL-for-2D-Turbulence/src

# Activate conda env
module load conda
conda activate base
source /lus/eagle/projects/MDClimSim/dpp94/venvs/2DTurbEmulator/bin/activate


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

# ----- Set args ----- #
run_num=SOPHIA_test
yaml_config=/home/dpp94/SSL-for-2D-Turbulence/src/config/vitnet_SOPHIA.yaml
config=BASE

# ------ Run main script ------ #

torchrun --nproc_per_node=${NUM_TASKS_PER_NODE} --nnodes=${NNODES} train.py --yaml_config=${yaml_config} --config=${config} --run_num=${run_num} --fresh_start
