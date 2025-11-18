
# Predictive Planner for Autonomous Driving with Consistency Models

This repository is the official implementation of the paper [**Predictive Planner for Autonomous Driving with Consistency Models**](https://arxiv.org/abs/2502.08033), accepted at the IEEE International Conference on Intelligent Transportation Systems (ITSC) 2025.

## Overview

We leverage the consistency model to build a predictive planner for autonomous vehicles that samples from a joint distribution of ego and surrounding agents, conditioned on the ego vehicle's navigational goal. Our method generates high-quality trajectories with fewer sampling steps than standard diffusion models, and achieve better satisfication of multiple planning constraints through an ADMM-inspired guided sampling method.

## Acknowledgments

This codebase builds upon and references:
- [**Motion Transformer (MTR)**](https://github.com/sshaoshuai/MTR) - For the motion encoder backbone and waymo dataset processing.
- [**denoising-diffusion-pytorch**](https://github.com/lucidrains/denoising-diffusion-pytorch) - For diffusion model implementations.

## Install

First create a conda environment and install packages. Our default setup uses CUDA 12.4.

```bash
$ conda create -n consistency python=3.8.19
$ conda activate consistency
$ pip install -r requirements.txt
```

Then build MTR model. Make sure to load cudatoolkit/12.4. Let `PROJECT_FOLDER` be the project root path.

```bash
$ cd PROJECT_FOLDER/model
$ python setup.py develop
```

**Note**: If you have a different CUDA version on your machine, make sure to install the corresponding PyTorch CUDA version and also load the corresponding cudatoolkit version when building MTR.

## Dataset preparation

We use [Waymo Open Motion Dataset v1.2](https://waymo.com/open/download/) with the scenario proto format.

To process the dataset, we follow the similar procedure as the [MTR dataset preparation guide](https://github.com/sshaoshuai/MTR/blob/master/docs/DATASET_PREPARATION.md).

First we install the waymo open dataset API
```bash
$ pip install waymo-open-dataset-tf-2-11-0
```

Then we download the dataset under `training/` and `validation_interactive/` to a local folder `data/waymo/scenario/`, and use them for training and testing, respectively.

Next we process the dataset
```bash
$ cd model/mtr/datasets/waymo
$ python data_preprocess.py ../../../../data/waymo/scenario  ../../../../data/waymo/v1_2
```

The processed dataset is saved in `data/waymo/v1_2`, and contains the following folders

```
processed_scenarios_training 
processed_scenarios_validation_interactive 
```
and the following files
```
processed_scenarios_training_infos.pkl
processed_scenarios_validation_interactive_infos.pkl
```

## Usage

First we enter the project folder `PROJECT_FOLDER`

```bash
cd PROJECT_FOLDER
```

### Configuration

The config file for MTR is in 

```bash
configs/mtr/mtr+100_percent_data_waymo_v1_2_interactive.yaml
```

For multi-GPU training with accelerator and deepspeed, the config files for accelerator and deepspeed are in

```
configs/accelerator/neuronic/accelerator_config_batch_100_gpu_4.yaml
configs/deepspeed/neuronic/deepspeed_config_server_batch_100_gpu_4.yaml
```

Note that we need to set `deepspeed_config_file` in `configs/accelerator/neuronic/accelerator_config_batch_100_gpu_4.yaml` to the absolute path of `configs/deepspeed/neuronic/deepspeed_config_server_batch_100_gpu_4.yaml`.


### Compute dataset statistics

To normalize the input data, we will compute the mean and std of the dataset. In the following script:
```bash
script/compute_stat.sh
```
Set the `PROJECT_FOLDER` to be the path to the project and set `DATA_ROOT_DIR` to be the path to the root of the training data.

Then run:

```bash
. script/compute_stat.sh
```

The stat file will be saved in `DATA_ROOT_DIR/data/waymo/v1_2/`

### Training


For multi-GPU training with accelerator and deepspeed, for the following script
```bash
script/train_consistency.sh
```
we set `WANDB_API_KEY` to be the wandb api key for you wandb account, `DATA_ROOT_DIR` to be the path to the root of the training data, `RESULT_FOLDER` to be the path to save the training results, and `DATA_STAT_FILE_PATH` to be the path of the computed data stat file.

Then we run the following script
```bash
. script/train_consistency.sh
```

This will start training with batch size of 100 across 4 GPUs.

For single-GPU training, use the following script
```bash
script/train_consistency_single_gpu.sh
```
Similar to the multi-GPU training script, we need to set `WANDB_API_KEY` to be the wandb api key for you wandb account, `DATA_ROOT_DIR` to be the path to the root of the training data, `RESULT_FOLDER` to be the path to save the training results, and `DATA_STAT_FILE_PATH` to be the path of the computed data stat file.

Then run:
```bash
. script/train_consistency_single_gpu.sh
```

**Note**: In this script, we use training batch size and validation batch size of 1 as a starting point to avoid memory overflow on a single GPU.

### Testing

To sample from the consistency model and compute the metrics, first set up the corresponding path variables in
```bash
script/sample_consistency_compute_metrics.sh
```
Then run the following script

```bash
. script/sample_consistency_compute_metrics.sh
```

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{li2025predictive,
  title={Predictive Planner for Autonomous Driving with Consistency Models},
  author={Li, Anjian and Bae, Sangjae and Isele, David and Beeson, Ryne and Tariq, Faizan M},
  journal={arXiv preprint arXiv:2502.08033},
  year={2025}
}
```
