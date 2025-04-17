# ------ Define all DDP vars ------ #
source export_DDP_vars.sh
export NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
#export WORLD_SIZE=$NUM_TASKS_PER_NODE
export OMP_NUM_THREADS=1

# ------ WANDB ------ #

source $HOME/set_wandb_key_dpp94.sh

# ------ Define all input args ------ #

YAML_CONFIG=/global/homes/d/dpp94/SSL-for-2D-Turbulence/src/config/vitnet_PM_MAE_mr0.5_seed_20_pf.yaml
CONFIG=MAE_FINETUNE
RUN_NUM=0000_MAE_mr0.5_seed_20_pf


# ------ Run main script ------ #

torchrun --nproc_per_node=${NUM_TASKS_PER_NODE} --nnodes=${SLURM_NNODES} train.py --yaml_config $YAML_CONFIG --config $CONFIG --run_num $RUN_NUM --fresh_start
#python -u train.py --yaml_config $YAML_CONFIG --config $CONFIG --run_num $RUN_NUM --fresh_start