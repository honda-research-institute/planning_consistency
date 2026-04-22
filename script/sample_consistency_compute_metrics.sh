#!/bin/bash
# set -euo pipefail

# Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /data/conda_envs/ftariq/envs/flowmatching

# Set the GPU to use
export CUDA_VISIBLE_DEVICES=2,3

# Navigate to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Set the PYTHONPATH to the project root directory
export PYTHONPATH=$(pwd)

# Define path variables
DATA_ROOT_DIR="PATH_TO_DATA_ROOT_DIR"
DATA_ROOT="${DATA_ROOT_DIR}/data/waymo/v1_2_interactive/processed_scenarios_validation_interactive"
OUTPUT_METRICS_DIR="${PROJECT_ROOT}/output/release"
RESULT_FOLDER="${PROJECT_ROOT}/output/test"
CHECKPOINT_FOLDER="${DATA_ROOT_DIR}/output/neuronic/multiagent_prediction/2024-10-27_17-58-20/checkpoint"
DATA_STAT_FILE="${DATA_ROOT_DIR}/data/waymo/v1_2/training_data_local_coord_statistics_reference_type_last_valid_surrounding_k_5_distance_threshold_10_metric_only_future_data_downsample_1.pkl"
TEST_SCRIPT="${PROJECT_ROOT}/run/test/consistency/test_trajectory_metrics.py"

# consistency + guidance
python \
${TEST_SCRIPT} \
--wandb_mode offline \
--cfg_file configs/mtr/mtr+100_percent_data_waymo_v1_2_validation.yaml \
--output_metrics_dir ${OUTPUT_METRICS_DIR} \
--data_root ${DATA_ROOT} \
--data_root_dir ${DATA_ROOT_DIR} \
--result_folder ${RESULT_FOLDER} \
--dataset_downsample_ratio 0.25 \
--test_batch_size 40 \
--checkpoint_results_folder ${CHECKPOINT_FOLDER} \
--checkpoint_name model-best_validation_epoch-23.pt \
--to_sample True \
--sample_num 1 \
--plot_batch_idx "(0,)" \
--sigma_max 80 \
--unet_type original_unet \
--project_name release_test \
--model_name consistency \
--condition_pos_type local_coord_pos_heading \
--data_x_type x_y_vx_vy \
--rollout_within_NN False \
--rollout_type new_integration \
--to_save False \
--to_plot False \
--data_stat_file_path ${DATA_STAT_FILE} \
--data_stat_type per_car_timestep_state \
--distance_threshold 10.0 \
--surrounding_distance_metric only_future \
--surrounding_k 5 \
--unet_model_channels 128 \
--all_agent_reference_state_type last_valid \
--image_type center_agent_traj_image \
--sample_with_mask False \
--gradient_step_num 100 \
--goal_reaching_guidance True \
--acceleration_limit_guidance True \
--angular_speed_limit_guidance True \
--main_model_type consistency \
--training_timesteps 5 \
--sampling_steps 5 \
--to_test_entire_validation_set True \
--compute_planning_constraints True \
--compute_trajectory_quality True