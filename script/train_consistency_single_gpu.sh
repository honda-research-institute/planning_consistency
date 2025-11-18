#!/bin/bash

# Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate consistency

# Set the GPU to use (single GPU)
export CUDA_VISIBLE_DEVICES=0

# Navigate to the project root directory
cd /n/fs/beesondm/anjian/project/trajectory_diffusion

# Set the PYTHONPATH to the project root directory
export PYTHONPATH=$(pwd)

# Define variables
DATA_ROOT_DIR="PATH_TO_DATA_ROOT_DIR"
RESULT_FOLDER="PATH_TO_RESULT_FOLDER"
DATA_STAT_FILE_PATH="PATH_TO_DATA_STAT_FILE_PATH/training_data_local_coord_statistics_reference_type_last_valid_surrounding_k_5_distance_threshold_10_metric_only_future_data_downsample_1.pkl"
WANDB_API_KEY="YOUR_WANDB_API_KEY"

python run/train/consistency/train_consistency.py \
    --use_lr_scheduler=False \
    --data_root_dir=$DATA_ROOT_DIR \
    --result_folder=$RESULT_FOLDER \
    --workers 8 \
    --dataset_downsample_ratio 1.0 \
    --train_batch_size 1 \
    --validation_batch_size 1 \
    --epochs 50 \
    --wandb_mode=offline \
    --wandb_api_key=$WANDB_API_KEY \
    --project_name=multiagent_prediction \
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