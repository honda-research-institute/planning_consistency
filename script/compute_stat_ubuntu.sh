#!/bin/bash

# Set project folder variable
PROJECT_FOLDER="PATH_TO_PROJECT_FOLDER"

# Set data root directory variable
DATA_ROOT_DIR="PATH_TO_DATA_ROOT_DIR"

# Activate the conda environment
#source ~/.bashrc
#eval "$(conda shell.bash hook)"
conda activate consistency

# Set the GPU to use
export CUDA_VISIBLE_DEVICES=0

# Navigate to the project root directory
cd $PROJECT_FOLDER

# Set the PYTHONPATH to the project root directory
export PYTHONPATH=$(pwd)

python model/consistency/data_processing/compute_data_stat.py --wandb_mode=offline \
  --project_name=compute_stat \
  --dataset_downsample_ratio=1. \
  --surrounding_distance_metric=only_future \
  --surrounding_k=5 \
  --distance_threshold=10.0 \
  --all_agent_reference_state_type=last_valid \
  --data_root_dir=$DATA_ROOT_DIR \
  --result_folder=$PROJECT_FOLDER/output




