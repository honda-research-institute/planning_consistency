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



def main():
    # Read the config from yaml file #######################################################################################
    args, cfg = parse_config()

    cfg['ROOT_DIR'] = Path(args.data_root_dir).resolve()
    model_params = {}
    for key, val in vars(args).items():
        model_params[key] = val

    dist_train = False
    args.without_sync_bn = True

    # Set up random seed for model initialization
    common_utils.set_random_seed(args.random_seed)

    # Configure result_folder with time
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_folder_with_time = f"{args.result_folder}/{args.project_name}/{current_datetime}"
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
        batch_size=args.train_batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        add_worker_init_fn=args.add_worker_init_fn,
        downsample_ratio=args.dataset_downsample_ratio,
        args=args,
    )

    validation_set, validation_loader, validation_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=args.validation_batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=False,
        downsample_ratio=args.val_dataset_downsample_ratio,
        args=args,
    )

    # Build model ########################################################################################################
    mtr_model_cfg = cfg.MODEL
    mtr_data_cfg = cfg.DATA_CONFIG

    # Modify mtr model size if needed
    if args.mtr_decoder_model_size == 256:
        mtr_model_cfg.MOTION_DECODER.D_MODEL = 256
        mtr_model_cfg.MOTION_DECODER.MAP_D_MODEL = 128
    elif args.mtr_decoder_model_size == 512:
        pass
    else:
        raise ValueError(f"unet_model_channels {args.unet_model_channels} is not supported!")

    if args.main_model_type == 'consistency':
        from model.trajectory_consistency_model import TrajectoryConsistency
        model = TrajectoryConsistency(mtr_model_cfg=mtr_model_cfg, mtr_data_cfg=mtr_data_cfg, consistency_model_params=model_params)
    else:
        raise ValueError(f"main_model_type {args.main_model_type} is not supported!")

    total_data_len = len(train_loader)
    train_num_steps = total_data_len * args.epochs

    # Build trainer ####################################################################################################
    trainer = Trainer(
        model=model,
        train_data_loader=train_loader,
        validation_data_loader=validation_loader,
        train_lr=args.train_lr,
        train_num_steps=train_num_steps,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        results_folder=result_folder_with_time,
        project_name=args.project_name,
        model_params=model_params,
        max_grad_norm=args.max_grad_norm,
        curr_datetime=current_datetime,
        accelerator=accelerator,
        checkpoint_name=args.checkpoint_name,
        use_lr_scheduler=args.use_lr_scheduler,
        cfg_mtr_optimization=cfg.OPTIMIZATION
    )

    trainer.train()

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # Training setup #################################################################
    parser.add_argument('--cfg_file',
                        type=str,
                        default='configs/mtr/mtr+100_percent_data_waymo_v1_2_validation.yaml',
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
    parser.add_argument('--train_batch_size',
                        type=int,
                        default=2,
                        help='batch size for train dataloader')
    parser.add_argument('--validation_batch_size',
                        type=int,
                        default=2,
                        help='batch size for validation dataloader')
    parser.add_argument('--epochs',
                        type=int,
                        default=2,
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
                        default=0.001,
                        help='downsample ratio of dataset to train dataset')
    parser.add_argument('--val_dataset_downsample_ratio',
                        type=float,
                        default=1.,
                        help='downsample ratio of dataset to train dataset')
    parser.add_argument('--project_name',
                        type=str,
                        default="consistency_train_v1",
                        help='project name')
    parser.add_argument('--wandb_mode',
                        type=str,
                        default='offline',
                        choices=['offline', 'online'],
                        help='wandb mode')
    parser.add_argument('--wandb_api_key',
                        type=str,
                        help='wandb api key to login wandb')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=1.,
                        help='gradient norm clipping')
    parser.add_argument('--use_lr_scheduler',
                        type=str,
                        default="True",
                        choices=["True", "False"],
                        help="whether to use learning rate scheduler")
    parser.add_argument('--train_lr',
                        type=float,
                        default=8e-5,
                        help='learning rate for training')
    parser.add_argument('--to_plot_violation',
                        type=lambda x: x.lower() == 'true',
                        default=False,
                        help='whether to plot violation')

    # Model setup ################################################################
    parser.add_argument('--dense_prediction_loss_weight',
                        type=float,
                        default=0.01,
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
                        default="original_unet",
                        choices=["original_unet"],
                        help='type of unet model to use')
    parser.add_argument('--data_stat_file_path',
                        type=str,
                        help='path to the data stat file')
    parser.add_argument('--data_stat_type',
                        type=str,
                        default='per_timestep_state',
                        choices=['per_timestep_state','per_car_timestep_state'])
    parser.add_argument('--sigma_max',
                        type=float,
                        default=80.,
                        help='maximum noise level during forward diffusion sample')
    parser.add_argument('--data_x_type',
                        type=str,
                        default='x_y_vx_vy',
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
                        help="type of rollout")
    parser.add_argument('--checkpoint_name',
                        default=None,
                        type=str,
                        help='checkpoint path')
    parser.add_argument('--increasing_sampling_step',
                        type=str,
                        default="False",
                        choices=["True", "False"],
                        help="whether to increasing sampling step")
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
                        default='last_valid',
                        choices=['last_valid', 'first_valid'],
                        help='How to choose the reference state for computing local coordinate')
    parser.add_argument('--condition_pos_type',
                        type=str,
                        default="original_pos",
                        choices=['original_pos', 'local_coord_pos_heading'],)
    parser.add_argument('--std_padding',
                        type=float,
                        default=0.0,
                        help='add a small value to std, avoid nan loss')
    parser.add_argument('--main_model_type',
                        type=str,
                        default='consistency',
                        choices=['consistency'],
                        help='type of main model to generate trajectory')
    parser.add_argument('--selected_scenario_ids',
                        type=parse_list,
                        default=[],
                        help="list of scenario IDs to train (as strings)")
    parser.add_argument('--mtr_decoder_model_size',
                        type=int,
                        default=512,
                        choices=[256, 512],
                        help='size of mtr decoder model')
    parser.add_argument('--sampling_steps',
                    type=int,
                    default=5,
                    help='consistency or diffusion sampling step. Consistency default is 5, diffusion default is 4')
    parser.add_argument('--training_timesteps',
                    type=int,
                    default=4,
                    help='especially for diffusion model training timesteps, maybe larger than sampling steps')

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
