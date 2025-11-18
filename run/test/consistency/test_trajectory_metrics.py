import time

from run.train.consistency.trainer_consistency import Trainer
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

from run.test.consistency.test_consistency_utils import prepare_data_batch, compute_planning_constraint_violation, compute_trajectory_quality


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
    test_set, test_loader, test_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=args.test_batch_size,
        dist=dist_test, workers=args.workers,
        logger=logger,
        training=False,
        downsample_ratio=args.dataset_downsample_ratio,
        args=args,
    )

    # Build model ########################################################################################################
    mtr_model_cfg = cfg.MODEL
    mtr_data_cfg = cfg.DATA_CONFIG

    if model_params['main_model_type'] == 'consistency':
        from model.trajectory_consistency_model import TrajectoryConsistency
        model = TrajectoryConsistency(mtr_model_cfg=mtr_model_cfg, mtr_data_cfg=mtr_data_cfg,
                                    consistency_model_params=model_params)
    else:
        raise ValueError(f"main_model_type {model_params['main_model_type']} is not supported") 

    checkpoint_results_folder = args.checkpoint_results_folder
    checkpoint_name = args.checkpoint_name

    # Prepare model and dataloader
    model, test_loader = accelerator.prepare(model, test_loader)
    device = accelerator.device
    data = torch.load(f"{checkpoint_results_folder}/{checkpoint_name}", map_location=device)

    # Load model
    model = accelerator.unwrap_model(model)
    model.load_state_dict(data['model'])
    # model.eval()

    # Load ema model
    ema = EMA(model).to(device)
    ema.load_state_dict(data['ema'])
    ema_model = ema.model

    ema_model.eval()
        
    # Sample and compute planning constraints ##############################################################################################
    if args.to_sample == "True" and args.compute_planning_constraints == "True":
        total_batch_data_num = 0
        total_goal_reaching_violation = 0.
        total_acc_limit_violation = 0.
        total_omega_limit_violation = 0.

        batch_idx = 0
        for batch in test_loader:
            if args.to_test_entire_validation_set == "True":
                pass
            else:
                if batch_idx in args.plot_batch_idx:
                    pass
                else:
                    break

            # Sample predicted_x_future
            start_time = time.time()

            image_parent_dir = f"{result_folder}/{args.model_name}/{args.image_type}_gradient_step_{args.gradient_step_num}_guidance_goal_{args.goal_reaching_guidance}_acceleration_{args.acceleration_limit_guidance}_angular_{args.angular_speed_limit_guidance}/idx_{batch_idx}"
            os.makedirs(image_parent_dir, exist_ok=True)
            
            # predicted_x_future shape: (center_agent_num, sample_num, agent_num, timestep, x_feature_size)
            predicted_x_future = model.sample(batch=batch, sample_num=args.sample_num, image_parent_dir=image_parent_dir, test_loader=test_loader)

            end_time = time.time()
            print(f"sample time need {end_time - start_time}")

            # Prepare data 
            if model_params['main_model_type'] == 'consistency':
                data_batch = prepare_data_batch(predicted_x_future=predicted_x_future, batch=batch, args=args,
                                to_transform_coordinate=True)
            else:
                raise ValueError(f"main_model_type {model_params['main_model_type']} is not supported")

            # compute planning constraint violation
            curr_batch_data_num, curr_batch_goal_reaching_violation, curr_batch_acc_limit_violation, curr_batch_omega_limit_violation = compute_planning_constraint_violation(data_batch=data_batch, args=args)

            total_batch_data_num += curr_batch_data_num
            total_goal_reaching_violation += curr_batch_goal_reaching_violation * curr_batch_data_num
            total_acc_limit_violation += curr_batch_acc_limit_violation * curr_batch_data_num
            total_omega_limit_violation += curr_batch_omega_limit_violation * curr_batch_data_num


            # print the progress for every 5 batches
            if batch_idx % 5 == 0:
                print(f"current progress: {batch_idx / len(test_loader) * 100:.2f}%")

            batch_idx += 1

        print("sample is done!")
        print(f"total_batch_data_num {total_batch_data_num}")
    
        goal_reaching_violation = total_goal_reaching_violation / total_batch_data_num
        acc_limit_violation = total_acc_limit_violation / total_batch_data_num
        omega_limit_violation = total_omega_limit_violation / total_batch_data_num

        # Save the data to a file
        if not os.path.exists(f"{args.output_metrics_dir}/planning_constraints"):
            os.makedirs(f"{args.output_metrics_dir}/planning_constraints", exist_ok=True)
        data_path = f"{args.output_metrics_dir}/planning_constraints/{args.model_name}_guidance_goal_reaching_{args.goal_reaching_guidance}_acceleration_{args.acceleration_limit_guidance}_angular_{args.angular_speed_limit_guidance}_planning_constraints.txt"  # Specify your desired file path

        # Open the file in write mode ('w')
        with open(data_path, 'w') as f:
            # Write the formatted string to the file
            f.write(f"model {args.model_name}, guidance goal_reaching {args.goal_reaching_guidance}, acceleration {args.acceleration_limit_guidance}, angular {args.angular_speed_limit_guidance}, total_batch_data_num {total_batch_data_num}, goal_reaching_violation {goal_reaching_violation}, acc_limit_violation {acc_limit_violation}, omega_limit_violation {omega_limit_violation}\n")
        
        print(f"{data_path} is saved!")

    # Sample and compute trajectory quality ##############################################################################################
    if args.to_sample == "True" and args.compute_trajectory_quality == "True":
        total_batch_data_num = 0
        total_angle_change = 0.
        total_path_length = 0.
        total_curvature = 0.

        batch_idx = 0
        for batch in test_loader:
            if args.to_test_entire_validation_set == "True":
                pass
            else:
                if batch_idx in args.plot_batch_idx:
                    pass
                else:
                    break

            # Sample predicted_x_future
            start_time = time.time()

            image_parent_dir = f"{result_folder}/{args.model_name}/{args.image_type}_gradient_step_{args.gradient_step_num}_guidance_goal_{args.goal_reaching_guidance}_acceleration_{args.acceleration_limit_guidance}_angular_{args.angular_speed_limit_guidance}/idx_{batch_idx}"
            os.makedirs(image_parent_dir, exist_ok=True)
            
            # predicted_x_future shape: (center_agent_num, sample_num, agent_num, timestep, x_feature_size)
            predicted_x_future = model.sample(batch=batch, sample_num=args.sample_num, image_parent_dir=image_parent_dir, test_loader=test_loader)

            end_time = time.time()
            print(f"sample time need {end_time - start_time}")

            # Prepare data 
            if model_params['main_model_type'] == 'consistency':
                data_batch = prepare_data_batch(predicted_x_future=predicted_x_future, batch=batch, args=args,
                                to_transform_coordinate=True)
            else:
                raise ValueError(f"main_model_type {model_params['main_model_type']} is not supported")

            # compute trajectory quality
            curr_batch_data_num, curr_batch_angle_change, curr_batch_path_length, curr_batch_curvature = compute_trajectory_quality(data_batch=data_batch, args=args)

            total_batch_data_num += curr_batch_data_num
            total_angle_change += curr_batch_angle_change * curr_batch_data_num
            total_path_length += curr_batch_path_length * curr_batch_data_num
            total_curvature += curr_batch_curvature * curr_batch_data_num

            # print the progress for every 5 batches
            if batch_idx % 5 == 0:
                print(f"current progress: {batch_idx / len(test_loader) * 100:.2f}%")

            batch_idx += 1

        print("sample is done!")
    
        angle_change = total_angle_change / total_batch_data_num
        path_length = total_path_length / total_batch_data_num
        curvature = total_curvature / total_batch_data_num

        # Save the data to a file
        if not os.path.exists(f"{args.output_metrics_dir}/trajectory_quality"):
            os.makedirs(f"{args.output_metrics_dir}/trajectory_quality", exist_ok=True)
        data_path = f"{args.output_metrics_dir}/trajectory_quality/{args.model_name}_guidance_goal_reaching_{args.goal_reaching_guidance}_acceleration_{args.acceleration_limit_guidance}_angular_{args.angular_speed_limit_guidance}_trajectory_quality.txt"  # Specify your desired file path

        # Open the file in write mode ('w')
        with open(data_path, 'w') as f:
            # Write the formatted string to the file
            f.write(f"model {args.model_name}, guidance goal_reaching {args.goal_reaching_guidance}, acceleration {args.acceleration_limit_guidance}, angular {args.angular_speed_limit_guidance}, total_batch_data_num {total_batch_data_num}, angle change {angle_change}, path length {path_length}, curvature {curvature}\n")
        
        print(f"{data_path} is saved!")


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # Training setup #################################################################
    parser.add_argument('--output_metrics_dir',
                        type=str,
                        help='directory to save metrics')
    parser.add_argument('--cfg_file',
                        type=str,
                        default='configs/mtr/test.yaml',
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
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=64,
                        help='batch size for train dataloader')
    parser.add_argument('--epochs',
                        type=int,
                        default=5,
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
                        default="consistency_test_v2",
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
                        default=32,
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
    parser.add_argument('--embed_condition_layers_dims',
                        type=parse_tuple,
                        default=(64,),
                        help='dimensions for embedding condition layers for each separate condition')
    parser.add_argument('--max_all_agent_num',
                        type=int,
                        default=128,
                        help='maximum number of agents in a scenario')
    parser.add_argument('--embed_x_dim',
                        type=int,
                        default=8,
                        help='Embed x dim before UNet')
    parser.add_argument('--all_condition_layer_dim',
                        type=parse_tuple,
                        default=(256,),
                        help='condition layer dim for all conditions concatenated')
    parser.add_argument('--build_mlp_method',
                        type=str,
                        default='mtr',
                        choices=['mtr', 'unet'],
                        help='sources of building mlp script')
    parser.add_argument('--unet_type',
                        type=str,
                        default="unet_ldm",
                        choices=["unet_ldm", "original_unet"],
                        help='type of unet model to use')
    parser.add_argument('--checkpoint_results_folder',
                        type=str,
                        help='folder for saved checkpoints')
    parser.add_argument('--checkpoint_name',
                        type=str,
                        default="model-best_validation_epoch-29.pt",
                        help='name of checkpoint to use')
    parser.add_argument('--to_sample',
                        type=str,
                        default='True',
                        help='whether to sample')
    parser.add_argument('--sample_num',
                        type=int,
                        default=4,
                        help='number of samples for each center agent')
    parser.add_argument('--data_root',
                        type=str,
                        help='path to the data')
    parser.add_argument('--plot_batch_num',
                        type=int,
                        default=1,)
    parser.add_argument('--plot_batch_idx',
                        type=parse_tuple,
                        default=(0,),)
    parser.add_argument('--data_stat_file_path',
                        type=str,
                        default='data/waymo/v1_2/training_data_statistics.pkl')
    parser.add_argument('--data_stat_type',
                        type=str,
                        default='per_timestep_state',
                        choices=['per_timestep_state','per_car_timestep_state'])
    parser.add_argument('--sigma_max',
                        type=float,
                        default=80.,
                        help='maximum noise level during forward diffusion sample')
    parser.add_argument('--model_name',
                        type=str,
                        default='20_data_sigma_max_80_sampling_step_5_original_unet',
                        help='name of model to test')
    parser.add_argument('--data_x_type',
                        type=str,
                        default='x_y_heading_vx_vy',
                        choices=['x_y_vx_vy', 'x_y_heading_vx_vy', 'x_y'],
                        help='which data x to use, choose x_y_vx_vy or x_y_heading_vx_vy')
    parser.add_argument('--rollout_within_NN',
                        type=str,
                        default="True",
                        choices=["True", "False"],
                        help="whether to roll out trajectory in NN. If true, output actions. If False, output trajectory")
    parser.add_argument('--rollout_type',
                        type=str,
                        default="old_fixed",
                        choices=["old_fixed", "new_integration"],
                        help="which rollout method to use")
    parser.add_argument('--to_save',
                        type=str,
                        default="False",
                        choices=["True", "False"],
                        help="whether to save images")
    parser.add_argument('--image_type',
                        type=str,
                        default="center_agent_traj_image",
                        choices=["all_agent_traj_image", "center_agent_traj_image", "surrounding_agent_traj_image", "all_agent_traj_image_only_gt"])
    parser.add_argument('--increasing_sampling_step',
                        type=str,
                        default="False",
                        choices=["True", "False"],
                        help="whether to increasing sampling step")
    parser.add_argument('--to_plot',
                        type=str,
                        default="False",
                        choices=["True", "False"],)
    parser.add_argument('--surrounding_k',
                        type=int,
                        default=10,
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
    parser.add_argument('--encoder_type',
                        type=str,
                        default='mtr',
                        choices=['mtr', 'mtr_pp'],
                        help='type of encoder')
    parser.add_argument('--all_agent_reference_state_type',
                        type=str,
                        default='first_valid',
                        choices=['last_valid', 'first_valid'],
                        help='How to choose the reference state for computing local coordinate')
    parser.add_argument('--condition_pos_type',
                        type=str,
                        default="local_coord_pos_heading",
                        choices=['original_pos', 'local_coord_pos_heading'],)
    parser.add_argument('--std_padding',
                        type=float,
                        default=0.0,
                        help='add a small value to std, avoid nan loss')
    parser.add_argument('--sample_with_mask',
                        type=str,
                        default="True",
                        choices=["True", "False"],
                        help="whether to sample with mask")
    parser.add_argument('--goal_reaching_guidance',
                        type=str,
                        default="False",
                        choices=["True", "False"],
                        help="whether to use goal reaching guidance")
    parser.add_argument('--acceleration_limit_guidance',
                        type=str,
                        default="False",
                        choices=["True", "False"],
                        help="whether to use acceleration limit guidance")
    parser.add_argument('--angular_speed_limit_guidance',
                        type=str,
                        default="False",
                        choices=["True", "False"],
                        help="whether to use angular speed limit guidance")
    parser.add_argument('--gradient_step_num',
                        type=int,
                        default=1,
                        help="number of gradient steps to apply classifier guidance")
    parser.add_argument('--selected_scenario_ids',
                        type=parse_list,
                        default=[],
                        help="list of scenario IDs to test (as strings)")
    parser.add_argument('--main_model_type',
                        type=str,
                        default='consistency',
                        choices=['consistency'],
                        help='type of main model to generate trajectory')
    parser.add_argument('--mtr_decoder_model_size',
                        type=int,
                        default=512,
                        help='size of mtr decoder model')
    parser.add_argument('--to_test_entire_validation_set',
                        type=str,
                        default='False',
                        choices=['True', 'False'],
                        help='whether to test entire validation set')
    parser.add_argument('--compute_planning_constraints',
                        type=str,
                        default='False',
                        choices=['True', 'False'],
                        help='whether to compute planning constraints')
    parser.add_argument('--sampling_steps',
                    type=int,
                    default=5,
                    help='consistency or diffusion sampling step. Consistency default is 5, diffusion default is 4')
    parser.add_argument('--training_timesteps',
                    type=int,
                    default=4,
                    help='especially for diffusion model training timesteps, maybe larger than sampling steps')
    parser.add_argument('--compute_trajectory_quality',
                        type=str,
                        default='False',
                        choices=['True', 'False'],
                        help='whether to compute trajectory quality')
    parser.add_argument('--to_plot_planning_constraints',
                        type=str,
                        default='False',
                        choices=['False'],
                        help='whether to plot planning constraints')
    parser.add_argument('--optimization_type',
                        type=str,
                        default='admm',
                        choices=['admm', 'gd', 'projection'],
                        help='whether to optimize gradient descent or ascent')
    parser.add_argument('--to_plot_violation',
                        type=lambda x: x.lower() == 'true',
                        default=False,
                        help='whether to plot violation')
    parser.add_argument('--goal_reaching_grad_scale',
                        type=float,
                        default=None,
                        help='gradient scale for goal reaching guidance')
    parser.add_argument('--acceleration_limit_grad_scale',
                        type=float,
                        default=None,
                        help='gradient scale for acceleration limit guidance')
    parser.add_argument('--angular_speed_limit_grad_scale',
                        type=float,
                        default=None,
                        help='gradient scale for angular speed limit guidance')
    parser.add_argument('--to_compute_trajectory_diversity',
                        type=lambda x: x.lower() == 'true',
                        default=False,
                        help='whether to compute trajectory diversity')
    parser.add_argument('--compute_collision_rate',
                        type=lambda x: x.lower() == 'true',
                        default=False,
                        help='whether to compute collision rate')

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
    project_root = Path(__file__).resolve().parent.parent.parent
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
        # Evaluate the string representation of a list
        parsed = eval(s)
        # Ensure all elements are strings
        return [str(x) for x in parsed]
    except:
        raise argparse.ArgumentTypeError("List argument must be a valid Python list")


if __name__ == '__main__':
    main()
