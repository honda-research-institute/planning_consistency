import time

from run.train.consistency.trainer_consistency import Trainer
from model.trajectory_consistency_model import TrajectoryConsistency
from model.mtr.utils import common_utils
from model.mtr.datasets import build_dataloader

from model.mtr.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from accelerate import Accelerator, DataLoaderConfiguration

import torch
from torch.utils.data import TensorDataset
import argparse
from pathlib import Path

import datetime
import os
import shutil
from pathlib import Path

import logging

from ema_pytorch import EMA
import numpy as np
import pickle

from run.test.consistency.previous.test_consistency_utils import *

to_plot = False

def main():
    # Read the config from yaml file #######################################################################################
    args, cfg = parse_config()

    cfg['ROOT_DIR'] = Path(args.data_root_dir).resolve()
    model_params = {}
    for key, val in vars(args).items():
        model_params[key] = val
    model_params["unet_attention_resolutions"] = list(model_params["unet_attention_resolutions"])

    dist_test = False
    total_gpus = 1
    args.without_sync_bn = True

    # Set up random seed for model initialization
    common_utils.set_random_seed(args.random_seed)

    # Configure result_folder with time
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_folder = f"{args.result_folder}/{args.project_name}"
    result_folder_with_time = f"{args.result_folder}/{args.project_name}/log/{current_datetime}"
    os.makedirs(result_folder_with_time, exist_ok=True)

    # Configure wandb mode
    if args.wandb_mode == "offline":
        os.environ["WANDB_MODE"] = "offline"

    # Configure accelerator ##########################################################################################
    dataloader_config = DataLoaderConfiguration(split_batches=True)
    accelerator = Accelerator(
        dataloader_config=dataloader_config,
        mixed_precision='no',
    )

    # log output #########################################################################################################
    output_dir = f"{result_folder_with_time}/output"
    os.makedirs(output_dir, exist_ok=True)

    log_file = f"{output_dir}/log_train_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"

    # Determine logging level based on main process
    log_level = logging.INFO if accelerator.is_main_process else logging.WARN
    logger = common_utils.create_logger(log_file, rank=0 if accelerator.is_main_process else 1, log_level=log_level)

    # log to file
    if accelerator.is_main_process:
        logger.info('**********************Start logging**********************')
        gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
        logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

        for key, val in vars(args).items():
            logger.info('{:16} {}'.format(key, val))

        # Copy config file only in the main process
        # SECURITY FIX: Use shutil.copy2 instead of os.system to prevent command injection
        try:
            cfg_path = Path(args.cfg_file)
            dest_path = Path(output_dir) / cfg_path.name
            shutil.copy2(cfg_path, dest_path)
            logger.info(f"Config file copied to: {dest_path}")
        except Exception as e:
            logger.error(f"Failed to copy config file: {e}")

    # Build dataset ####################################################################################################
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=args.batch_size,
        dist=False, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        add_worker_init_fn=args.add_worker_init_fn,
        downsample_ratio=args.dataset_downsample_ratio,
        args=args,
    )

    # Initialize accumulators for per_state statistics
    sum_per_state = torch.zeros(5, dtype=torch.float32)
    sum_per_state_squared = torch.zeros(5, dtype=torch.float32)

    # Initialize accumulators for dimensional statistics
    sum_per_car_timestep_state = torch.zeros((args.surrounding_k, 81, 5), dtype=torch.float32)
    sum_per_car_timestep_state_squared = torch.zeros((args.surrounding_k, 81, 5), dtype=torch.float32)

    # Initialize accumulators per timestep state statistics
    sum_per_timestep_state = torch.zeros((81, 5), dtype=torch.float32)
    sum_per_timestep_state_squared = torch.zeros((81, 5), dtype=torch.float32)

    # Count the number of valid samples for per_state and dimensional statistics
    num_samples_per_state = torch.zeros(5, dtype=torch.float32)
    num_samples_per_car_timestep_state = torch.zeros((args.surrounding_k, 81, 5), dtype=torch.float32)
    num_samples_per_timestep_state = torch.zeros((81, 5), dtype=torch.float32)

    curr_batch_num = 0
    for batch in train_loader:
        # Extract the data and mask
        data_x = batch['input_dict']['surrounding_obj_trajs_full'][:, :, 10:, :5].float()
        data_x_mask = batch['input_dict']['surrounding_obj_trajs_valid_mask'].unsqueeze(-1).expand(-1, -1, -1, 5)[:, :, 10:, :].float()

        # Apply the mask
        valid_data_x = data_x * data_x_mask

        # Update the sample count
        num_samples_per_state += data_x_mask.sum(dim=(0, 1, 2))
        num_samples_per_car_timestep_state += data_x_mask.sum(dim=0)
        num_samples_per_timestep_state += data_x_mask.sum(dim=(0, 1))

        # Accumulate sums for per_state statistics
        sum_per_state += valid_data_x.sum(dim=(0, 1, 2))
        sum_per_state_squared += (valid_data_x ** 2).sum(dim=(0, 1, 2))

        # Accumulate sums for dimensional statistics
        sum_per_car_timestep_state += valid_data_x.sum(dim=0)
        sum_per_car_timestep_state_squared += (valid_data_x ** 2).sum(dim=0)

        # Accumulate sums per timestep state
        sum_per_timestep_state += valid_data_x.sum(dim=(0, 1))
        sum_per_timestep_state_squared += (valid_data_x ** 2).sum(dim=(0, 1))

        if curr_batch_num % 100 == 0:
            print(f"progress {curr_batch_num}/{len(train_loader)}")

        curr_batch_num += 1

    # Compute per_state mean and std
    mean_per_state = sum_per_state / num_samples_per_state
    std_per_state = torch.sqrt((sum_per_state_squared / num_samples_per_state) - (mean_per_state ** 2))

    # Compute dimensional mean and std
    mean_per_car_timestep_state = sum_per_car_timestep_state / num_samples_per_car_timestep_state
    std_per_car_timestep_state = torch.sqrt((sum_per_car_timestep_state_squared / num_samples_per_car_timestep_state) - (mean_per_car_timestep_state ** 2))

    # Compute mean and std per timestep state
    mean_per_timestep_state = sum_per_timestep_state / num_samples_per_timestep_state
    std_per_timestep_state = torch.sqrt((sum_per_timestep_state_squared / num_samples_per_timestep_state) - (mean_per_timestep_state ** 2))

    # Handle zero standard deviation by adding epsilon
    # TODO: already add std padding in the consistency model code
    epsilon = 1e-3
    std_per_state[std_per_state == 0.] += epsilon
    std_per_car_timestep_state[std_per_car_timestep_state == 0.] += epsilon
    std_per_timestep_state[std_per_timestep_state == 0.] += epsilon

    # Prepare the statistics dictionary
    statistics = {
        'mean_per_state': mean_per_state.numpy(),
        'std_per_state': std_per_state.numpy(),
        'mean_per_car_timestep_state': mean_per_car_timestep_state.numpy(),
        'std_per_car_timestep_state': std_per_car_timestep_state.numpy(),
        'mean_per_timestep_state': mean_per_timestep_state.numpy(),
        'std_per_timestep_state': std_per_timestep_state.numpy(),
    }

    # Save to a pickle file
    data_path = f'{args.data_root_dir}/data/waymo/v1_2/training_data_local_coord_statistics_reference_type_{args.all_agent_reference_state_type}_surrounding_k_{args.surrounding_k}_distance_threshold_{int(args.distance_threshold)}_metric_{args.surrounding_distance_metric}_data_downsample_{args.dataset_downsample_ratio}.pkl'

    with open(data_path, 'wb') as f:
        pickle.dump(statistics, f)

    print("Statistics saved to", data_path)

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # Training setup #################################################################
    parser.add_argument('--cfg_file',
                        type=str,
                        default='configs/mtr/mtr+100_percent_data_waymo_v1_2.yaml',
                        help='specify the config for mtr encoder')
    parser.add_argument('--extra_tag',
                        type=str,
                        default='default',
                        help='extra tag for this experiment')
    parser.add_argument('--workers',
                        type=int,
                        default=8,
                        help='number of workers for dataloader')
    parser.add_argument('--merge_all_iters_to_one_epoch',
                        action='store_true',
                        default=False, help='')
    parser.add_argument('--add_worker_init_fn',
                        action='store_true',
                        default=False, help='')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='batch size for train dataloader')
    parser.add_argument('--epochs',
                        type=int,
                        default=1,
                        help='number of epochs')
    parser.add_argument('--random_seed',
                        type=int,
                        default=0,
                        help='random seed to initialize model')
    parser.add_argument('--result_folder',
                        type=str,
                        help='folder to save results')
    parser.add_argument('--data_root_dir',
                        type=str,
                        help='project root dir')
    parser.add_argument('--dataset_downsample_ratio',
                        type=float,
                        default=1.,
                        help='downsample ratio of dataset to train dataset')
    parser.add_argument('--project_name',
                        type=str,
                        default="compute_stat",
                        help='project name')
    parser.add_argument('--wandb_mode',
                        type=str,
                        default='offline',
                        choices=['offline', 'online'],
                        help='wandb mode')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=1.,
                        help='gradient norm clipping')
    parser.add_argument('--project_folder',
                        type=str,
                        default='/media/anjian/T7/project/honda/trajectory_diffusion',
                        help='project folder')
    parser.add_argument('--selected_scenario_ids',
                        type=parse_list,
                        default=[],
                        help="list of scenario IDs to train (as strings)")

    # Model setup ################################################################
    parser.add_argument('--dense_prediction_loss_weight',
                        type=float,
                        default=0.1,
                        help='loss weight for dense_prediction')
    parser.add_argument('--consistency_loss_weight',
                        type=float,
                        default=100.,
                        help='loss weight for consistency')
    parser.add_argument('--unet_model_channels',
                        type=int,
                        default=2,
                        help='dimensions for unet model')
    parser.add_argument('--unet_channel_mult',
                        type=parse_tuple,
                        default=(1, 2, 4),
                        help='dimensions mult for unet model'
                        )
    parser.add_argument('--unet_attention_resolutions',
                        type=parse_tuple,
                        default=(1, 2, 4),
                        help='dimensions with attention layer in unet')
    parser.add_argument('--sampling_steps',
                        type=int,
                        default=5,
                        help='consistency sampling step')
    parser.add_argument('--embed_condition_layers_dims',
                        type=parse_tuple,
                        default=(2,),
                        help='dimensions for embedding condition layers for each separate condition')
    parser.add_argument('--max_all_agent_num',
                        type=int,
                        default=128,
                        help='maximum number of agents in a scenario')
    parser.add_argument('--embed_x_dim',
                        type=int,
                        default=2,
                        help='Embed x dim before UNet')
    parser.add_argument('--all_condition_layer_dim',
                        type=parse_tuple,
                        default=(2,),
                        help='condition layer dim for all conditions concatenated')
    parser.add_argument('--build_mlp_method',
                        type=str,
                        default='mtr',
                        choices=['mtr', 'unet'],
                        help='sources of building mlp script')
    parser.add_argument('--unet_type',
                        type=str,
                        default="original_unet",
                        choices=["original_unet"],
                        help='type of unet model to use')
    parser.add_argument('--checkpoint_results_folder',
                        type=str,
                        default="/data/01/anjianli/trajectory_diffusion/output/consistency_train_v1/2024-07-12_15-37-18/checkpoint",
                        help='folder for saved checkpoints')
    parser.add_argument('--checkpoint_name',
                        type=str,
                        default="model-best_validation_epoch-29.pt",
                        help='name of checkpoint to use')
    parser.add_argument('--to_compute_loss',
                        type=str,
                        default='False',
                        help='whether to compute test loss')
    parser.add_argument('--to_sample',
                        type=str,
                        default='False',
                        help='whether to sample')
    parser.add_argument('--sample_num',
                        type=int,
                        default=1,
                        help='number of samples for each center agent')
    parser.add_argument('--data_root',
                        type=str,
                        default='/home/anjianli/Desktop/project/trajectory_diffusion/data/waymo/v1_2/processed_scenarios_validation')
    parser.add_argument('--surrounding_k',
                        type=int,
                        default=5,
                        help='choose surrounding k agents as training data')
    parser.add_argument('--distance_threshold',
                        type=float,
                        default=10.,
                        help='choose distance threshold when choosing surrounding agents')
    parser.add_argument('--surrounding_distance_metric',
                        type=str,
                        default='only_future',
                        choices=['all', 'only_future'],
                        help='which distance to look at when choosing surrounding agents')
    parser.add_argument('--all_agent_reference_state_type',
                        type=str,
                        default='last_valid',
                        choices=['last_valid', 'first_valid'],
                        help='How to choose the reference state for computing local coordinate')

    args = parser.parse_args()

    # SECURITY FIX: Validate the config file path to prevent path traversal attacks
    validated_cfg_path = validate_config_path(args.cfg_file)
    
    # Use the validated path for all config operations
    cfg_from_yaml_file(str(validated_cfg_path), cfg)
    cfg.TAG = validated_cfg_path.stem
    cfg.EXP_GROUP_PATH = '/'.join(validated_cfg_path.parts[-3:-1])  # safer path manipulation
    
    # Store the validated path back to args for later use
    args.cfg_file = str(validated_cfg_path)

    return args, cfg


def validate_config_path(cfg_file_input):
    """
    Validate and sanitize the config file path to prevent path traversal attacks.
    
    Args:
        cfg_file_input: User-provided config file path
        
    Returns:
        pathlib.Path: Validated absolute path
        
    Raises:
        ValueError: If the path is invalid or outside allowed directories
    """
    # Define the allowed base directory
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    allowed_config_dir = project_root / "configs"
    
    # Convert input to Path object and resolve to absolute path
    # resolve() normalizes the path and resolves symlinks
    try:
        cfg_path = Path(cfg_file_input).resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid config file path: {cfg_file_input}") from e
    
    # Check if the path exists and is a file
    if not cfg_path.exists():
        raise ValueError(f"Config file does not exist: {cfg_path}")
    
    if not cfg_path.is_file():
        raise ValueError(f"Config path is not a file: {cfg_path}")
    
    # Check if the file is within the allowed directory
    try:
        cfg_path.relative_to(allowed_config_dir)
    except ValueError:
        raise ValueError(
            f"Config file must be within {allowed_config_dir}, "
            f"but got: {cfg_path}"
        )
    
    # Validate file extension (only allow YAML files)
    if cfg_path.suffix not in ['.yaml', '.yml']:
        raise ValueError(f"Config file must be a YAML file (.yaml or .yml), got: {cfg_path.suffix}")
    
    return cfg_path


def parse_tuple(s):
    try:
        return eval(s)
    except:
        raise argparse.ArgumentTypeError("Tuple argument must be a valid Python tuple")


def parse_list(s):
    try:
        return eval(s)
    except:
        raise argparse.ArgumentTypeError("List argument must be a valid Python list")


if __name__ == '__main__':
    main()
