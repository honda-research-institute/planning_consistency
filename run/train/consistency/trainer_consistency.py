from pathlib import Path

import torch
from torch.optim import Adam

from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import torch.optim.lr_scheduler as lr_sched

from accelerate import Accelerator, DataLoaderConfiguration

# from denoising_diffusion_pytorch.version import __version__
# from model.consistency.utils.tools import *

import os
import wandb


# trainer class

class Trainer(object):
    def __init__(
            self,
            model,
            train_data_loader,
            validation_data_loader,
            *,
            train_batch_size=16,
            gradient_accumulate_every=1,
            train_lr=1e-4,
            train_num_steps=100000,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            weight_decay=0.01,
            results_folder='./results',
            project_name='test',
            amp=False,
            mixed_precision_type='fp16',
            split_batches=True,
            max_grad_norm=1.,
            model_params=None,
            curr_datetime,
            accelerator,
            checkpoint_name=None,
            use_lr_scheduler,
            cfg_mtr_optimization,
    ):
        super().__init__()

        # accelerator
        self.accelerator = accelerator

        # model
        self.model = model

        # sampling and training hyperparameters
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        # dataset and dataloader
        train_dl = self.accelerator.prepare(train_data_loader)
        self.train_dl = self.cycle(train_dl)
        self.step_per_epoch = len(train_dl)

        val_dl = self.accelerator.prepare(validation_data_loader)
        self.val_dl = val_dl

        # optimizer and learning rate scheduler
        self.use_lr_scheduler = use_lr_scheduler
        if model_params['main_model_type'] == 'transformer':
            self.use_lr_scheduler = "True"
            self.opt = self.build_optimizer(self.model, cfg_mtr_optimization)
            self.scheduler = self.build_scheduler(self.opt, cfg_mtr_optimization, model_params['epochs'], len(train_dl), -1)
        else:
            self.opt = Adam(model.parameters(), lr=train_lr, betas=adam_betas)
            if self.use_lr_scheduler == "True":
                self.scheduler = ReduceLROnPlateau(self.opt, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6)
            else:
                self.scheduler = None

        if self.accelerator.is_main_process:
            self.ema = EMA(model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        # Configure result folder
        self.checkpoint_results_folder = Path(f"{results_folder}/checkpoint")
        self.checkpoint_results_folder.mkdir(exist_ok=True, parents=True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # Keep track of the best checkpoints
        self.best_checkpoints = []
        # Define a variable to track the best validation loss
        self.best_val_loss = torch.tensor(float("inf"))

        # Initialize Wandb
        self.wandb_results_folder = Path(f"{results_folder}")
        self.wandb_results_folder.mkdir(parents=True, exist_ok=True)
        if self.accelerator.is_main_process:
            os.environ['WANDB_DIR'] = str(self.wandb_results_folder)
            if model_params['wandb_api_key'] is not None:
                wandb.login(key=model_params['wandb_api_key'])
            else:
                raise ValueError("Wandb api key is not provided")

            run_name = (
                f"{curr_datetime}_"
                f"main_model_type_{model_params['main_model_type']}_"
                f"mtr_decoder_model_size_{model_params['mtr_decoder_model_size']}_"
                f"{model_params['data_x_type']}_"
                f"{model_params['dataset_downsample_ratio'] * 100}%_data_"
                f"increase_sample_step_{model_params['increasing_sampling_step']}_step_{model_params['sampling_steps']}_"
                f"unet_model_dim_{model_params['unet_model_channels']}_"
                f"batch_size_{model_params['train_batch_size']}_"
            )
            wandb.init(
                project=project_name,
                name=run_name,
                config=model_params,
                group='multi_gpu_training',  # Add group for multi-GPU training
                job_type='train'  # Specify job type as train
            )

        # Load checkpoint if path is provided
        if checkpoint_name is not None:
            self.load(checkpoint_name)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, model_name):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if self.exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.checkpoint_results_folder / f'{model_name}.pt'))

    def load(self, model_name):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.checkpoint_results_folder / f'{model_name}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if self.exists(self.accelerator.scaler) and self.exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.
                total_predictor_loss = 0.
                total_consistency_loss = 0.
                total_pred_x0_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    batch = next(self.train_dl)

                    with self.accelerator.autocast():
                        self.curr_epoch_num = self.step // self.step_per_epoch

                        loss, predictor_loss, consistency_loss, pred_x0_loss = self.model(batch=batch, curr_epoch_num=self.curr_epoch_num)

                        # Ensure the losses are reduced to scalar values
                        loss = loss.mean() / self.gradient_accumulate_every
                        predictor_loss = predictor_loss.mean() / self.gradient_accumulate_every
                        consistency_loss = consistency_loss.mean() / self.gradient_accumulate_every
                        pred_x0_loss = pred_x0_loss.mean() / self.gradient_accumulate_every

                        # print(f"loss {loss}")
                        total_loss += loss.item()

                        # Update each loss
                        total_predictor_loss += predictor_loss.item()
                        total_consistency_loss += consistency_loss.item()
                        total_pred_x0_loss += pred_x0_loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                if self.accelerator.is_main_process:
                    wandb.log({
                        'train total_loss': total_loss,
                        'train predictor_loss': total_predictor_loss,
                        'train consistency_loss': total_consistency_loss,
                        'train pred_x0_loss': total_pred_x0_loss,
                        'step': self.step
                    }, commit=True)

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    # Save the checkpoint with the best validation value
                    if self.step % self.step_per_epoch == 0 and self.step != 0:
                        milestone = self.step // self.step_per_epoch  # this gives us the epoch number
                        print(f"Epoch {milestone}")

                        val_loss, val_predictor_loss, val_consistency_loss, val_predict_x0_loss = self.compute_validation_loss()

                        # Log validation loss
                        wandb.log({
                            'val total_loss': val_loss,
                            'val predictor_loss': val_predictor_loss,
                            'val consistency_loss': val_consistency_loss,
                            'val_predict_x0_loss': val_predict_x0_loss,
                            'epoch': milestone
                        }, commit=True)

                        # If validation loss is decreasing, save checkpoint Update the best validation loss and checkpoints
                        if val_consistency_loss < self.best_val_loss:
                            self.save(f"model-best_validation_epoch-{milestone}")
                            self.best_val_loss = val_consistency_loss
                            self.update_best_checkpoints(val_consistency_loss, f"model-best_validation_epoch-{milestone}")

                        # Use scheduler if it's enabled
                        if self.use_lr_scheduler == "True" and self.scheduler is not None:
                            self.scheduler.step(val_consistency_loss)

                pbar.update(1)

        accelerator.print('training complete')

    def update_best_checkpoints(self, val_loss, model_name):
        # Adding the new checkpoint and sorting
        self.best_checkpoints.append((val_loss, str(self.checkpoint_results_folder / f'{model_name}.pt')))
        self.best_checkpoints.sort(key=lambda x: x[0])

        # Keeping only top 1 checkpoints
        if len(self.best_checkpoints) > 1:
            _, checkpoint_to_remove = self.best_checkpoints.pop(1)
            if os.path.exists(checkpoint_to_remove):
                os.remove(checkpoint_to_remove)  # Delete the checkpoint file

    def compute_validation_loss(self):
        self.model.eval()  # Set model to evaluation mode
        self.ema.ema_model.eval()

        total_val_loss = 0.
        total_val_predictor_loss = 0.
        total_val_consistency_loss = 0.
        total_val_predict_x0_loss = 0.

        with torch.no_grad():
            for batch in self.val_dl:
                val_loss, predictor_loss, consistency_loss, predict_x0_loss = self.ema.ema_model(batch=batch, curr_epoch_num=self.curr_epoch_num)  # Use EMA model here
                total_val_loss += val_loss.mean().item()
                total_val_predictor_loss += predictor_loss.mean().item()
                total_val_consistency_loss += consistency_loss.mean().item()
                total_val_predict_x0_loss += predict_x0_loss.mean().item()

        average_val_loss = total_val_loss / len(self.val_dl)
        average_val_predictor_loss = total_val_predictor_loss / len(self.val_dl)
        average_val_consistency_loss = total_val_consistency_loss / len(self.val_dl)
        average_val_predict_x0_loss = total_val_predict_x0_loss / len(self.val_dl)
        return average_val_loss, average_val_predictor_loss, average_val_consistency_loss, average_val_predict_x0_loss

    def clip_lr(self):

        lr_clip = 1e-6
        for param_group in self.opt.param_groups:
            if param_group['lr'] < lr_clip:
                param_group['lr'] = lr_clip
    
    def build_optimizer(self, model, opt_cfg):
        if opt_cfg.OPTIMIZER == 'Adam':
            optimizer = torch.optim.Adam(
                [each[1] for each in model.named_parameters()],
                lr=opt_cfg.LR, weight_decay=opt_cfg.get('WEIGHT_DECAY', 0)
            )
        elif opt_cfg.OPTIMIZER == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg.LR, weight_decay=opt_cfg.get('WEIGHT_DECAY', 0))
        else:
            assert False

        return optimizer

    def build_scheduler(self, optimizer, opt_cfg, total_epochs, total_iters_each_epoch, last_epoch):
        decay_steps = [x * total_iters_each_epoch for x in opt_cfg.get('DECAY_STEP_LIST', [5, 10, 15, 20])]
        def lr_lbmd(cur_epoch):
            cur_decay = 1
            for decay_step in decay_steps:
                if cur_epoch >= decay_step:
                    cur_decay = cur_decay * opt_cfg.LR_DECAY
            return max(cur_decay, opt_cfg.LR_CLIP / opt_cfg.LR)

        if opt_cfg.get('SCHEDULER', None) == 'lambdaLR':
            scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)
        elif opt_cfg.get('SCHEDULER', None) == 'linearLR':
            total_iters = total_iters_each_epoch * total_epochs
            scheduler = lr_sched.LinearLR(optimizer, start_factor=1.0, end_factor=opt_cfg.LR_CLIP / opt_cfg.LR, total_iters=total_iters, last_epoch=last_epoch)
        else:
            scheduler = None

        return scheduler
    
    @staticmethod
    def cycle(dl):
        while True:
            for data in dl:
                yield data
    
    @staticmethod
    def exists(x):
        return x is not None