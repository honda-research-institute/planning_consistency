import os
import torch
import torch.nn as nn
from .mtr.models.context_encoder import build_context_encoder
from .mtr.models.context_encoder import build_context_encoder_predictor
from .mtr.models.motion_decoder import build_motion_decoder
from .consistency import build_consistency


class TrajectoryConsistency(nn.Module):

    def __init__(self, mtr_model_cfg, mtr_data_cfg, consistency_model_params):
        super(TrajectoryConsistency, self).__init__()

        self.mtr_model_cfg = mtr_model_cfg
        self.mtr_data_cfg = mtr_data_cfg
        self.consistency_model_params = consistency_model_params

        self.to_plot_violation = consistency_model_params["to_plot_violation"]

        self.context_encoder = build_context_encoder(self.mtr_model_cfg.CONTEXT_ENCODER)
        self.context_encoder_predictor = build_context_encoder_predictor(
            in_channels=self.context_encoder.num_out_channels,
            config=self.mtr_model_cfg.MOTION_DECODER
        )

        self.consistency_model = build_consistency(consistency_model_params=self.consistency_model_params,
                                                   mtr_data_cfg=self.mtr_data_cfg)

        self.dense_prediction_loss_weight = self.consistency_model_params['dense_prediction_loss_weight']
        self.consistency_loss_weight = self.consistency_model_params['consistency_loss_weight']

    def forward(self, batch, curr_epoch_num=None):

        # Encoder
        encoder_output = self.context_encoder(batch)

        # Dense prediction for encoder
        _, pred_dense_future_trajs = self.context_encoder_predictor(encoder_output)
        context_encoder_predictor_loss = self.context_encoder_predictor.compute_loss(
            pred_dense_trajs=pred_dense_future_trajs,
            obj_trajs_future_state=batch['input_dict']['obj_trajs_future_state'],
            obj_trajs_future_mask=batch['input_dict']['obj_trajs_future_mask'])


        # Consistency model
        consistency_loss, pred_x0_loss = self.consistency_model(encoder_output, curr_epoch_num)

        # loss function
        loss = self.consistency_loss_weight * consistency_loss + self.dense_prediction_loss_weight * context_encoder_predictor_loss

        return loss.mean(), context_encoder_predictor_loss.mean(), consistency_loss.mean(), pred_x0_loss.mean()

    def get_loss(self):
        loss, tb_dict, disp_dict = self.motion_decoder.get_loss()

        return loss, tb_dict, disp_dict

    # Function to print the size of each layer
    def print_model_parameters(self, model):
        params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params.append((name, param.size(), param.numel()))

        # Sort layers by number of parameters in descending order
        params.sort(key=lambda x: x[2], reverse=True)

        for name, size, num_params in params:
            print(f"Layer: {name} | Size: {size} | Number of parameters: {num_params}")

    @torch.no_grad()
    def sample(self, batch, sample_num, image_parent_dir=None, test_loader=None, to_change_agent_goal="False", optimization_type="admm"):

        # Encoder
        encoder_output = self.context_encoder(batch)

        # TODO: hack to change goal for agent 979 in scenario bd5
        if to_change_agent_goal == "True":
            encoder_output['center_obj_trajs_goal_state'][3, :] = torch.tensor([60., -4., 13., 0.5], device=batch['center_obj_trajs_goal_state'].device)

        # Consistency model
        # predicted_x_future shape: (batch_size, sample_num, agent_num, timestep, x_feature_size)
        if self.to_plot_violation == True:
            predicted_x_future, goal_reaching_violation_list, acc_limit_violation_list, omega_limit_violation_list = self.consistency_model.p_sample(batch=encoder_output, sample_num=sample_num, image_parent_dir=image_parent_dir, optimization_type=optimization_type)
            return predicted_x_future, goal_reaching_violation_list, acc_limit_violation_list, omega_limit_violation_list
        else:
            predicted_x_future = self.consistency_model.p_sample(batch=encoder_output, sample_num=sample_num, image_parent_dir=image_parent_dir, optimization_type=optimization_type)
            return predicted_x_future
