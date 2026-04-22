#!/bin/bash

# Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate flowmatching

# Set the GPU to use
export CUDA_VISIBLE_DEVICES=0,1

# Navigate to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Set the PYTHONPATH to the project root directory
export PYTHONPATH=$(pwd)

# Define variables
DATA_ROOT_DIR="${DATA_ROOT_DIR:-${PROJECT_ROOT}/../data/waymo/}"
CFG_FILE="${CFG_FILE_OVERRIDE:-$PROJECT_ROOT/configs/mtr/mtr+100_percent_data_waymo_v1_2_validation.yaml}"
RESULT_FOLDER="${RESULT_FOLDER:-$PROJECT_ROOT/output/trajectory_consistency_10p}"
DATA_STAT_FILE_PATH="${DATA_STAT_FILE_PATH:-${PROJECT_ROOT}/../data/waymo/v1_2/training_data_local_coord_statistics_reference_type_last_valid_surrounding_k_5_distance_threshold_10_metric_only_future_data_downsample_0.1.pkl}"
ACCELERATOR_CONFIG="configs/accelerator/accelerator_config_batch_100_gpu_2.yaml"
WANDB_API_KEY="wandb_v1_MtCJ0jheDj5Zq9VBN99Srl2evlu"

accelerate launch \
    --config_file $ACCELERATOR_CONFIG \
    --num_processes 2 \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_port 29508 \
    run/train/consistency/train_consistency.py \
    --cfg_file=$CFG_FILE \
    --use_lr_scheduler=False \
    --data_root_dir=$DATA_ROOT_DIR \
    --result_folder=$RESULT_FOLDER \
    --workers 8 \
    --dataset_downsample_ratio 0.10 \
    --train_batch_size 100 \
    --validation_batch_size 100 \
    --epochs 50 \
    --wandb_mode=offline \
    --wandb_api_key=$WANDB_API_KEY \
    --project_name=multiagent_prediction_10p \
    --unet_type=original_unet \
    --sigma_max=80. \
    --data_x_type=x_y_vx_vy \
    --rollout_within_NN=False \
    --increasing_sampling_step=False \
    --surrounding_k=5 \
    --distance_threshold=10. \
    --unet_model_channels 128 \
    --embed_condition_layers_dims "(64,)" \
    --surrounding_distance_metric=only_future \
    --data_stat_file_path=$DATA_STAT_FILE_PATH \
    --data_stat_type=per_car_timestep_state \
    --all_agent_reference_state_type=last_valid \
    --condition_pos_type=local_coord_pos_heading \
    --main_model_type=consistency \
    --consistency_loss_weight=100.