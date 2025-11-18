import torch
from torch.nn import Module
import torch.nn.functional as F
from torch import nn
import copy

from model.mtr.models.utils import common_layers
from model.consistency.utils.consistency_utils import *
from model.consistency.utils.classifier_guidance import classifier_guidance_admm

from tqdm import tqdm
import pickle

class Consistency(Module):
    def __init__(
            self,
            unet_model,
            sampling_steps,
            consistency_model_params,
            mtr_data_cfg,
    ):
        super(Consistency, self).__init__()

        self.consistency_model_params = consistency_model_params
        self.mtr_data_cfg = mtr_data_cfg
        self.max_map_num = self.mtr_data_cfg['NUM_OF_SRC_POLYLINES']
        self.max_all_agent_num = self.consistency_model_params['max_all_agent_num']

        self.unet_model = unet_model
        self.build_encoder_process_layer()
        if self.consistency_model_params["increasing_sampling_step"] == "False":
            self.sampling_steps = sampling_steps
        elif self.consistency_model_params["increasing_sampling_step"] == "True":
            self.sampling_steps = None
        else:
            print("wrong increasing_sampling_step")
            exit()
        
        # For increasing sampling steps
        self.s0 = 10
        self.s1 = 1280

        self.sigma_min = 0.002
        self.sigma_max = self.consistency_model_params['sigma_max']

        self.pseudo_huber_loss_c = 0.002

        self.curr_max_agent_num = 0

        # Load data statistics
        stat_file = self.consistency_model_params['data_stat_file_path']
        with open(stat_file, "rb") as f:
            stat = pickle.load(f)

        if self.consistency_model_params["data_stat_type"] == "per_car_timestep_state":
            data_mean = torch.from_numpy(stat['mean_per_car_timestep_state']).to(dtype=torch.float32)
            data_std = torch.from_numpy(stat['std_per_car_timestep_state']).to(dtype=torch.float32)

            self.x_future_mean, self.x_future_std = data_mean[:, 1:, :], data_std[:, 1:, :]
            self.x_current_mean, self.x_current_std = data_mean[:, 0, :].unsqueeze(1), data_std[:, 0, :].unsqueeze(1)
        elif self.consistency_model_params["data_stat_type"] == "per_timestep_state":
            data_mean = torch.from_numpy(stat['mean_per_timestep_state']).to(dtype=torch.float32)
            data_std = torch.from_numpy(stat['std_per_timestep_state']).to(dtype=torch.float32)

            self.x_future_mean, self.x_future_std = data_mean[1:, :], data_std[1:, :]
            self.x_current_mean, self.x_current_std = data_mean[0, :].unsqueeze(1), data_std[0, :].unsqueeze(1)

        self.std_padding = self.consistency_model_params['std_padding']

        self.curr_epoch_num = None

    def forward(self, batch, curr_epoch_num=None):

        # center_objects_world, center obj at the current timestep ##########################################################
        # (num_center_objects, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        center_objects_world = batch['input_dict']['center_objects_world']

        # Prepare UNet data ##############################################################################################
        batch_size = center_objects_world.shape[0]
        device = center_objects_world.device

        # Update the sampling step
        if self.consistency_model_params["increasing_sampling_step"] == "True":
            assert curr_epoch_num is not None, "no curr_epoch_num!"
            self.sampling_steps = self.get_current_sampling_steps(curr_epoch_num, device)

        if self.curr_epoch_num != curr_epoch_num:
            print(f"current epoch {curr_epoch_num}, sampling step is {self.sampling_steps}")
            self.curr_epoch_num = curr_epoch_num

        # Prepare condition c
        c, c_mask, _ = self.prepare_condition_c(batch=batch)

        # prepare data x
        (x_future, x_current, x_current_future_mask,
         normalized_x_future, normalized_x_current) = self.prepare_data_x(batch=batch)

        # Sample the noisy data ######################################################################################
        # Sample the noise level
        # sigma_level = [sigma_min = sigma_1, sigma_2, ..., sigma_N = sigma_max]
        sigma_level = self.sample_noise_level(N=self.sampling_steps, device=device)

        # Prepare time given sampling steps
        # x_sigma_min is an approximation of the original x (normalized_x_future)
        x_sigma_t, x_sigma_t_plus_1, x_sigma_min, sigma_t, sigma_t_plus_1 = self.q_sample(x=normalized_x_future,
                                                                                          sigma_level=sigma_level,
                                                                                          batch_size=batch_size,
                                                                                          )

        # Consistency function ##################################################################
        # Compute f_theta for both x_sigma_t and x_sigma_t_plus_1
        f_theta_x_t = self.f_theta_function(x_sigma_t=x_sigma_t, sigma_t=sigma_t, c=c,
                                            x_current=x_current)
        f_theta_x_t_plus_1 = self.f_theta_function(x_sigma_t=x_sigma_t_plus_1, sigma_t=sigma_t_plus_1, c=c,
                                                   x_current=x_current)

        # Loss function ##################################################################################
        consistency_loss = self.compute_loss(f_theta_x_t, f_theta_x_t_plus_1, sigma_t, sigma_t_plus_1, x_current_future_mask)

        # Compute predict x0 loss, to measure how the function output close to the groundtruth
        with torch.no_grad():
            x_future_mask = x_current_future_mask[:, :, 1:]
            x_future_mask = x_future_mask.unsqueeze(-1).expand(-1, -1, -1, f_theta_x_t.shape[-1])
            pred_x0_loss_1 = torch.sqrt((f_theta_x_t - normalized_x_future) ** 2) * x_future_mask
            pred_x0_loss_2 = torch.sqrt((f_theta_x_t_plus_1 - normalized_x_future) ** 2) * x_future_mask
            pred_x0_loss = pred_x0_loss_1 + pred_x0_loss_2
            pred_x0_loss = torch.mean(pred_x0_loss)

        return consistency_loss, pred_x0_loss

    def get_current_sampling_steps(self, curr_epoch_num, device):

        if not torch.is_tensor(self.s0):
            self.s0 = torch.tensor(self.s0, device=device)
            self.s1 = torch.tensor(self.s1, device=device)

        # K is the total training epochs
        K = self.consistency_model_params['epochs']
        assert K >= (torch.log2(torch.floor(self.s1 / self.s0)) + 1), f"total epoch is too small, needs to be bigger than {(torch.log2(torch.floor(self.s1 / self.s0)) + 1)}"
        K_prime = torch.floor(K / (torch.log2(torch.floor(self.s1 / self.s0)) + 1))
        k = curr_epoch_num

        sampling_steps = torch.min(self.s0 * (2. ** (torch.floor(k / K_prime))), self.s1) + 1

        return sampling_steps


    def f_theta_function(self, x_sigma_t, sigma_t, c, x_current):

        predict_x_0_from_t = self.F_theta_function(x_sigma_t=x_sigma_t, sigma_t=sigma_t, c=c,
                                                   x_current=x_current)

        c_skip_sigma_t, c_out_sigma_t = self.get_c_skip_c_out_function(sigma_curr=sigma_t)

        f_theta_x_t = c_skip_sigma_t * x_sigma_t + c_out_sigma_t * predict_x_0_from_t

        return f_theta_x_t

    @torch.no_grad()
    def p_sample(self, batch, sample_num, image_parent_dir=None, optimization_type="admm"):

        # Prepare UNet data ##############################################################################################
        # Prepare condition c
        c, c_mask, _ = self.prepare_condition_c(batch=batch)

        # # prepare data x
        (x_future, x_current, x_current_future_mask,
         normalized_x_future, normalized_x_current) = self.prepare_data_x(batch=batch)

        # Configure data dimension
        batch_size, agent_num, timestep, x_future_feature_size = x_future.shape
        x_current_feature_size = x_current.shape[-1]
        device = x_future.device

        # Expand c and x_current, x_current by sample_num
        c_feature_size = c.shape[-1]
        c_expanded = c.unsqueeze(1).expand(batch_size, sample_num, c_feature_size).reshape(-1, c_feature_size)
        x_current_expanded = x_current.unsqueeze(1).expand(batch_size, sample_num, agent_num, 1, x_current_feature_size).reshape(-1, agent_num, 1, x_current_feature_size)

        # expand x_future_mask by sample_num
        x_future_mask = x_current_future_mask[:, :, 1:]
        x_future_mask = x_future_mask.unsqueeze(-1).unsqueeze(1).expand(-1, sample_num, -1, -1, x_future_feature_size)

        # Expand groundtruth, normalized_x_future, normalized_x_future by sample_num
        normalized_x_future_expanded = normalized_x_future.unsqueeze(1).expand(batch_size, sample_num, agent_num, timestep, x_future_feature_size)
        x_future_expanded = x_future.unsqueeze(1).expand(batch_size, sample_num, agent_num, timestep, x_future_feature_size)

        # Sample loop #################################################################################################
        # sigma_level = [sigma_min = sigma_1, sigma_2, ..., sigma_N = sigma_max]
        sigma_level = self.sample_noise_level(N=self.sampling_steps, device=device)
        sigma_min = sigma_level[0]
        sigma_max = sigma_level[-1]

        # At the beginning, curr_x is sampled from N(0, sigma_max^2 * I)
        # total dim is batch_size * sample_num
        curr_x_sigma_t = torch.randn(batch_size * sample_num, agent_num, timestep, x_future_feature_size, device=device) * sigma_max
        curr_sigma_t = sigma_max.expand(batch_size * sample_num, )

        # Collect violation
        goal_reaching_violation_list = []
        acc_limit_violation_list = []
        omega_limit_violation_list = []

        for i in tqdm(reversed(range(0, self.sampling_steps - 1)), desc='sampling step', total=self.sampling_steps - 1):
            # curr_x_sigma_t = f_theta(curr_x_sigma_t+1,curr_sigma_t+1) + sqrt(curr_sigma_t^2 - sigma_min^2) * z_k, z_k ~ N(0, I)
            # Compute f_theta(curr_x_sigma_t+1,curr_sigma_t+1)
            f_theta_x_t = self.f_theta_function(x_sigma_t=curr_x_sigma_t, sigma_t=curr_sigma_t, c=c_expanded,
                                                x_current=x_current_expanded)
            
            # f_theta_x_t is the predicted x0, where we apply classifier guidance through gradient step
            # if self.consistency_model_params["goal_reaching_guidance"] == "True" or self.consistency_model_params["acceleration_limit_guidance"] == "True" or self.consistency_model_params["angular_speed_limit_guidance"] == "True":
            center_obj_goal = batch['center_obj_trajs_goal_state']
            center_obj_goal = center_obj_goal.unsqueeze(1).expand(-1, sample_num, -1)
            center_obj_goal = center_obj_goal.reshape(batch_size * sample_num, -1)

            # apply classifier guidance
            # if self.consistency_model_params["goal_reaching_guidance"] == "True" or self.consistency_model_params["acceleration_limit_guidance"] == "True" or self.consistency_model_params["angular_speed_limit_guidance"] == "True":
                
            if optimization_type == "admm":
                f_theta_x_t, goal_reaching_violation, acc_limit_violation, omega_limit_violation = classifier_guidance_admm(goal_reaching_guidance=self.consistency_model_params["goal_reaching_guidance"],
                                                                                                                            acceleration_limit_guidance=self.consistency_model_params["acceleration_limit_guidance"],
                                                                                                                            angular_speed_limit_guidance=self.consistency_model_params["angular_speed_limit_guidance"],
                                                                                                                            gradient_step_num=self.consistency_model_params["gradient_step_num"],
                                                                                                                            f_theta_x_t=f_theta_x_t, center_obj_goal=center_obj_goal,
                                                                                                                            x_future_mean=self.x_future_mean, x_future_std=self.x_future_std, std_padding=self.std_padding,
                                                                                                                            preset_goal_reaching_grad_scale=self.consistency_model_params["goal_reaching_grad_scale"],
                                                                                                                            preset_acceleration_limit_grad_scale=self.consistency_model_params["acceleration_limit_grad_scale"],
                                                                                                                            preset_angular_speed_limit_grad_scale=self.consistency_model_params["angular_speed_limit_grad_scale"])
            else:
                print(f"wrong optimization type {self.consistency_model_params['optimization_type']}")
                exit()


            # Collect violation
            if i == 0:
                goal_reaching_violation_list.append(goal_reaching_violation)
                acc_limit_violation_list.append(acc_limit_violation)
                omega_limit_violation_list.append(omega_limit_violation)
                
            # Compute second term: sqrt(curr_sigma_t^2 - sigma_min^2) * z_k, z_k ~ N(0, I)
            curr_sigma_t = sigma_level[i]
            curr_z = torch.randn(batch_size * sample_num, agent_num, timestep, x_future_feature_size, device=device)
            second_term = torch.sqrt(curr_sigma_t.pow(2) - sigma_min.pow(2)) * curr_z

            # Compute the new curr_x_sigma_t
            curr_x_sigma_t = f_theta_x_t + second_term
            curr_sigma_t = curr_sigma_t.repeat(batch_size * sample_num, )

        # Reshape f_theta_x_t to (batch_size, sample_num, agent_num, timestep, x_feature_size)
        f_theta_x_t = f_theta_x_t.reshape(batch_size, sample_num, agent_num, timestep, x_future_feature_size)

        # Use mean and sigma to unnormalize
        if self.x_future_mean is not None and self.x_future_std is not None:
            if self.consistency_model_params['data_x_type'] == "x_y_heading_vx_vy":
                original_f_theta_x_t = f_theta_x_t * self.x_future_std + self.x_future_mean
            elif self.consistency_model_params['data_x_type'] == "x_y_vx_vy":
                original_f_theta_x_t = f_theta_x_t * self.x_future_std[..., [0, 1, 3, 4]] + self.x_future_mean[..., [0, 1, 3, 4]]
            elif self.consistency_model_params['data_x_type'] == "x_y":
                original_f_theta_x_t = f_theta_x_t * self.x_future_std[..., [0, 1]] + self.x_future_mean[..., [0, 1]]
        else:
            print(f"data x statistics is not loaded!")
            exit()

        # Compute diff####################################################################################################
        if self.consistency_model_params["to_plot_violation"] == True:
            return original_f_theta_x_t, goal_reaching_violation_list, acc_limit_violation_list, omega_limit_violation_list
        else:
            return original_f_theta_x_t

    def compute_loss(self, f_theta_x_t, f_theta_x_t_plus_1, sigma_t, sigma_t_plus_1, x_current_future_mask):

        #########################################################################################################
        weight = 1. / (sigma_t_plus_1 - sigma_t)
        weight = weight / weight.sum()

        x_future_mask = x_current_future_mask[:, :, 1:]
        x_future_mask = x_future_mask.unsqueeze(-1).expand(-1, -1, -1, f_theta_x_t.shape[-1])

        # Pseudo-huber loss
        diff = torch.sqrt(
            (f_theta_x_t - f_theta_x_t_plus_1) ** 2 + self.pseudo_huber_loss_c ** 2) - self.pseudo_huber_loss_c
        diff = diff * x_future_mask

        loss = diff.mean(dim=(1, 2, 3))
        loss = torch.sum(loss * weight)
        return loss

    def get_c_skip_c_out_function(self, sigma_curr):

        batch_size = sigma_curr.shape[0]

        sigma_data = 0.5
        c_skip_sigma_curr = sigma_data ** 2 / ((sigma_curr - self.sigma_min) ** 2 + sigma_data ** 2)
        c_out_sigma_curr = sigma_data * (sigma_curr - self.sigma_min) / torch.sqrt(sigma_data ** 2 + sigma_curr ** 2)

        c_skip_sigma_curr = c_skip_sigma_curr.view(batch_size, 1, 1, 1)
        c_out_sigma_curr = c_out_sigma_curr.view(batch_size, 1, 1, 1)

        return c_skip_sigma_curr, c_out_sigma_curr

    def F_theta_function(self, x_sigma_t, sigma_t, c, x_current):

        # before UNet, embed the data x to get x_emb
        x_emb = self.embed_x(x_sigma_t)

        # Here UNet output will have the same dimension as x_emb，here the timesteps will be replaced by the noise sigma
        if self.consistency_model_params['unet_type'] == "original_unet":
            output = self.unet_model(x=x_emb, time=sigma_t, c=c)
        elif self.consistency_model_params['unet_type'] == "unet_ldm":
            # Since x_emb has shape (batch_size, surrounding_agent_num, 80, embed_x_dim), to have cross attention,
            #  context c should also reshape to have shape (batch_size, 1, all_condition_layer_dim)
            c = c.unsqueeze(1)
            output = self.unet_model(x=x_emb, timesteps=sigma_t, context=c)

        # Use FC layers to get only 2-dim action output
        if self.consistency_model_params["rollout_within_NN"] == "True":
            output_action = self.unet_output_layer(output)

            if self.consistency_model_params['data_x_type'] == "x_y_heading_vx_vy":
                # Integrate the actions to get x ########################################################################
                predict_x_future = roll_out_x_y_heading_vx_vy(output_action=output_action, x_current=x_current)
                predict_normalized_x_future = (predict_x_future - self.x_future_mean) / (self.x_future_std + self.std_padding)
            elif self.consistency_model_params['data_x_type'] == "x_y_vx_vy":
                predict_x_future = roll_out_x_y_vx_vy(output_action=output_action, x_current=x_current)
                predict_normalized_x_future = (predict_x_future - self.x_future_mean[:, :, [0, 1, 3, 4]]) / (self.x_future_std[:, :, [0, 1, 3, 4]] + self.std_padding)
            elif self.consistency_model_params['data_x_type'] == "x_y":
                if self.consistency_model_params['rollout_type'] == "old_fixed":
                    predict_x_future = roll_out_x_y_old_fixed(output_action=output_action, x_current=x_current)
                elif self.consistency_model_params['rollout_type'] == "new_integration":
                    predict_x_future = roll_out_x_y_new_integration(output_action=output_action, x_current=x_current)
                predict_normalized_x_future = (predict_x_future - self.x_future_mean[:, :, [0, 1]]) / (self.x_future_std[:, :, [0, 1]] + self.std_padding)
            else:
                print("wrong data x type")
                exit()


        elif self.consistency_model_params["rollout_within_NN"] == "False":
            predict_normalized_x_future = self.unet_output_layer(output)

        return predict_normalized_x_future

    def sample_noise_level(self, N, device):

        assert N >= 2, "N should be an integer at least 2"

        ############################################################################################################
        # Compute noise level sigma_level = [sigma_min = sigma_1, sigma_2, ..., sigma_N = sigma_max]
        level = torch.arange(1, N + 1, device=device)
        rou = 7.
        one_over_rou = 1. / rou

        sigma_min_rou = self.sigma_min ** one_over_rou
        sigma_max_rou = self.sigma_max ** one_over_rou

        sigma_level = sigma_min_rou + ((level - 1) / (N - 1)) * (sigma_max_rou - sigma_min_rou)
        sigma_level = sigma_level ** rou

        return sigma_level

    def q_sample(self, x, sigma_level, batch_size):

        ############################################################################################################
        # Sample noise sigma_t, sigma_{t+1} for batch with discretized lognormal distribution
        P_mean = -1.1
        P_std = 2.0

        # Suppose sampling step = N, Here probabilities has shape N-1,
        #  actually we need f(x_sigma_t) and f(x_sigma_t-1), so that is fine because each has shape N-1
        log_sigma_t = torch.log(sigma_level)
        erf_upper = torch.erf((log_sigma_t[1:] - P_mean) / (P_std * torch.sqrt(torch.tensor(2.0))))
        erf_lower = torch.erf((log_sigma_t[:-1] - P_mean) / (P_std * torch.sqrt(torch.tensor(2.0))))
        probabilities = erf_upper - erf_lower

        # Normalize probabilities to sum to 1
        # probabilities = probabilities / probabilities.sum()
        # Sample indices based on the calculated probabilities, the input probabilities doesn't need to sum to 1 according to https://pytorch.org/docs/stable/generated/torch.multinomial.html
        sigma_t_indices = torch.multinomial(probabilities, batch_size, replacement=True)
        sigma_t_plus_1_indices = sigma_t_indices + 1

        # Get the corresponding sigma values
        sigma_t = sigma_level[sigma_t_indices]
        sigma_t_plus_1 = sigma_level[sigma_t_plus_1_indices]

        sigma_t = sigma_t.view(-1, 1, 1, 1)
        sigma_t_plus_1 = sigma_t_plus_1.view(-1, 1, 1, 1)

        # Compute noisy_x as x + sigma * z
        z = torch.randn_like(x)
        x_sigma_t = x + sigma_t * z
        x_sigma_t_plus_1 = x + sigma_t_plus_1 * z
        x_sigma_min = x + self.sigma_min * z

        return x_sigma_t, x_sigma_t_plus_1, x_sigma_min, sigma_t.view(-1), sigma_t_plus_1.view(-1)

    def prepare_condition_c(self, batch):

        # Extract condition y information ###############################################################################
        if self.consistency_model_params["condition_pos_type"] == "original_pos":
            obj_feature, obj_mask, obj_pos = batch['obj_feature'], batch['obj_mask'], batch['obj_pos'][..., 0:2]

        elif self.consistency_model_params["condition_pos_type"] == "local_coord_pos_heading":
            # TODO: here we use obj_pos to represent [x, y, angle]
            obj_feature, obj_mask, obj_pos = batch['obj_feature'], batch['obj_mask'], batch['all_agent_reference_xy_heading_normalized']
        else:
            print("Condition pos type is not wrong!")
        map_feature, map_mask, map_pos = batch['map_feature'], batch['map_mask'], batch['map_pos'][..., 0:2]
        center_objects_feature = batch['center_objects_feature']
        center_obj_trajs_goal_state = batch['center_obj_trajs_goal_state']

        batch_size, device = obj_feature.shape[0], obj_feature.device

        # Set maximum all agent num
        max_map_num = self.max_map_num

        # Get obj_condition, map_condition ###############################################################################
        # Here before encoding obj_pos, map_pos, apply mask
        obj_feature_valid = obj_feature * obj_mask.unsqueeze(-1)
        obj_pos_valid = obj_pos * obj_mask.unsqueeze(-1)
        map_feature_valid = map_feature * map_mask.unsqueeze(-1)
        map_pos_valid = map_pos * map_mask.unsqueeze(-1)
        # Encode obj_pos, map_pos
        if self.consistency_model_params["condition_pos_type"] == "original_pos":
            obj_pos_feature_valid = self.obj_pos_encoding_layer(obj_pos_valid)
        elif self.consistency_model_params["condition_pos_type"] == "local_coord_pos_heading":
            obj_pos = self.obj_pos_encoding_layer(obj_pos_valid[:, :, :2])
            obj_heading = self.heading_encoding_layer(obj_pos_valid[:, :, 2].unsqueeze(-1))
            obj_pos_feature_valid = torch.cat([obj_pos, obj_heading], dim=-1)
        map_pos_feature_valid = self.map_pos_encoding_layer(map_pos_valid)

        # Calculate the current feature sizes
        all_agent_num = obj_feature_valid.shape[1]
        all_map_num = map_feature_valid.shape[1]

        # TODO: test that actually all_agent_num is indeed > 128
        if all_agent_num > self.curr_max_agent_num:
            self.curr_max_agent_num = all_agent_num

        # print(f"valid agent num is at most {last_true_index}")
        if all_agent_num > self.max_all_agent_num:
            print(f"wow, valid obj num is {all_agent_num}")
            print(f"curr max agent num is {self.curr_max_agent_num}")
            # exit()

        # Pad the flattened features/mask to the target size,
        # First [0, 0] means no padding for the last dim
        # Next [0, pad_num] means right padding for the second last dim by pad_num
        padded_obj_feature = F.pad(obj_feature_valid, (0, 0, 0, self.max_all_agent_num - all_agent_num),
                                   'constant', 0.0)
        padded_obj_pos_feature = F.pad(obj_pos_feature_valid, (0, 0, 0, self.max_all_agent_num - all_agent_num),
                                       'constant', 0.0)
        padded_obj_mask = F.pad(obj_mask, (0, self.max_all_agent_num - all_agent_num), 'constant', False)

        padded_map_feature = F.pad(map_feature_valid, (0, 0, 0, max_map_num - all_map_num),
                                   'constant', 0.0)
        padded_map_pos_feature = F.pad(map_pos_feature_valid, (0, 0, 0, max_map_num - all_map_num), 'constant', 0.0)
        padded_map_mask = F.pad(map_mask, (0, max_map_num - all_map_num), 'constant', False)

        # Encode center agent goal state ###############################################################################
        center_obj_goal_condition = self.agent_goal_encoding_layer(center_obj_trajs_goal_state)

        # Combined the flattened features and position #############################################################
        obj_condition = torch.cat([padded_obj_feature, padded_obj_pos_feature], dim=-1)
        map_condition = torch.cat([padded_map_feature, padded_map_pos_feature], dim=-1)

        # Encode each condition and concatenate
        # Apply the mask again
        obj_condition = obj_condition * padded_obj_mask.unsqueeze(-1)
        map_condition = map_condition * padded_map_mask.unsqueeze(-1)

        # embed condition
        c_obj = self.obj_condition_mlp(obj_condition)
        c_map = self.map_condition_mlp(map_condition)
        c_goal = self.goal_condition_mlp(center_obj_goal_condition)

        # Apply the mask again
        c_obj = c_obj * padded_obj_mask.unsqueeze(-1)
        c_map = c_map * padded_map_mask.unsqueeze(-1)

        flatten_obj_mask = padded_obj_mask.unsqueeze(-1).expand(-1, -1, c_obj.shape[2]).reshape(batch_size, -1)
        flatten_map_mask = padded_map_mask.unsqueeze(-1).expand(-1, -1, c_map.shape[2]).reshape(batch_size, -1)
        flatten_goal_mask = torch.ones_like(c_goal, dtype=torch.bool, device=device)

        c_obj = c_obj.view(batch_size, -1)
        c_map = c_map.view(batch_size, -1)
        c_goal = c_goal.view(batch_size, -1)

        c = torch.cat([c_obj, c_map, c_goal], dim=-1)
        c_mask = torch.cat([flatten_obj_mask, flatten_map_mask, flatten_goal_mask], dim=-1)

        try:
            assert torch.sum(c[~c_mask]) == 0.  # assert that all the invalid feature are masked as 0
        except AssertionError:
            print("may have nan error")
            exit()

        # After flatten the original mask structure disappears and becomes [center_agent_num * all_agent_num/map_num], thus after mlp there is no need to mask again
        c = self.all_condition_mlp(c)

        return c, c_mask, center_objects_feature

    def prepare_data_x(self, batch):

        # Extract data x information ####################################################################################
        surrounding_obj_trajs_full, surrounding_obj_trajs_valid_mask, surrounding_obj_index = (
            batch['input_dict']['surrounding_obj_trajs_full'],
            batch['input_dict']['surrounding_obj_trajs_valid_mask'],
            batch['input_dict']['surrounding_obj_index'])

        # Only get the future trajectory for data x, which should be the first 5 entry (x, y, heading, vx, vy)
        surrounding_obj_trajs_future = surrounding_obj_trajs_full[:, :, 11:, :5]
        surrounding_obj_trajs_future_valid_mask = surrounding_obj_trajs_valid_mask[:, :, 11:].unsqueeze(-1).expand(-1,
                                                                                                                   -1,
                                                                                                                   -1,
                                                                                                                   surrounding_obj_trajs_future.shape[
                                                                                                                       3])
        x_future = surrounding_obj_trajs_future * surrounding_obj_trajs_future_valid_mask

        # Get the current trajectory for data x, will be used in integration part
        surrounding_obj_trajs_current = surrounding_obj_trajs_full[:, :, 10, :5].unsqueeze(-2)
        surrounding_obj_trajs_current_valid_mask = surrounding_obj_trajs_valid_mask[:, :, 10].unsqueeze(-1).unsqueeze(
            -1).expand(-1, -1, -1, surrounding_obj_trajs_current.shape[3])
        x_current = surrounding_obj_trajs_current * surrounding_obj_trajs_current_valid_mask

        # Normalize data using mean and sigma ####################################################################
        if self.x_future_mean.device != x_future.device:
            self.x_future_mean, self.x_future_std = self.x_future_mean.to(x_future.device), self.x_future_std.to(x_future.device)
            self.x_current_mean, self.x_current_std = self.x_current_mean.to(x_future.device), self.x_current_std.to(x_future.device)

        normalized_x_future = (x_future - self.x_future_mean) / (self.x_future_std + self.std_padding)
        normalized_x_current = (x_current - self.x_current_mean) / (self.x_current_std + self.std_padding)

        ################################################################################################################
        x_current_future_mask = surrounding_obj_trajs_valid_mask[:, :, 10:]

        if self.consistency_model_params['data_x_type'] == "x_y_vx_vy":
            x_future, x_current, normalized_x_future, normalized_x_current = x_future[:, :, :, [0, 1, 3, 4]], x_current[:, :, :, [0, 1, 3, 4]], normalized_x_future[:, :, :, [0, 1, 3, 4]], normalized_x_current[:, :, :, [0, 1, 3, 4]]
        elif self.consistency_model_params['data_x_type'] == "x_y_heading_vx_vy":
            pass
        elif self.consistency_model_params['data_x_type'] == "x_y":
            x_future, x_current, normalized_x_future, normalized_x_current = x_future[:, :, :, [0, 1]], x_current[:, :, :, [0, 1, 3, 4]], normalized_x_future[:, :, :, [0, 1]], normalized_x_current[:, :, :, [0, 1, 3, 4]]
        else:
            print("wrong data x type")
            exit()
        return x_future, x_current, x_current_future_mask, normalized_x_future, normalized_x_current

    def build_encoder_process_layer(self):

        consistency_model_params = self.consistency_model_params

        obj_condition_dim, map_condition_dim, goal_condition_dim = 512, 512, 256
        embed_condition_layers_dims = consistency_model_params['embed_condition_layers_dims']
        embed_condition_layers_dims = list(embed_condition_layers_dims)
        all_condition_layer_dim = consistency_model_params['all_condition_layer_dim']
        all_condition_layer_dim = list(all_condition_layer_dim)

        embedded_condition_dim = embed_condition_layers_dims[-1]
        all_embedded_condition_dim = (embedded_condition_dim * self.max_all_agent_num) + (
                embedded_condition_dim * self.max_map_num) + embedded_condition_dim

        embed_x_dim = consistency_model_params['embed_x_dim']

        if self.consistency_model_params["condition_pos_type"] == "original_pos":
            self.obj_pos_encoding_layer = common_layers.build_mlps(
                c_in=2, mlp_channels=[256, 256, 256], ret_before_act=True, without_norm=True
            )
        elif self.consistency_model_params["condition_pos_type"] == "local_coord_pos_heading":
            self.heading_encoding_layer = common_layers.build_mlps(
                c_in=1, mlp_channels=[256, 256, 128], ret_before_act=True, without_norm=True
            )
            self.obj_pos_encoding_layer = common_layers.build_mlps(
                c_in=2, mlp_channels=[256, 256, 128], ret_before_act=True, without_norm=True
            )
        else:
            print("wrong condition pos type")
            exit()

        self.map_pos_encoding_layer = common_layers.build_mlps(
            c_in=2, mlp_channels=[256, 256, 256], ret_before_act=True, without_norm=True
        )
        self.agent_goal_encoding_layer = common_layers.build_mlps(
            c_in=4, mlp_channels=[256, 256, 256], ret_before_act=True, without_norm=True
        )

        # Use build_mlp
        if consistency_model_params['build_mlp_method'] == 'mtr':
            self.obj_condition_mlp = common_layers.build_mlps(
                c_in=obj_condition_dim, mlp_channels=embed_condition_layers_dims, ret_before_act=True, without_norm=True
            )
            self.map_condition_mlp = common_layers.build_mlps(
                c_in=map_condition_dim, mlp_channels=embed_condition_layers_dims, ret_before_act=True, without_norm=True
            )
            self.goal_condition_mlp = common_layers.build_mlps(
                c_in=goal_condition_dim, mlp_channels=embed_condition_layers_dims, ret_before_act=True,
                without_norm=True
            )

            self.all_condition_mlp = common_layers.build_mlps(
                c_in=all_embedded_condition_dim, mlp_channels=all_condition_layer_dim, ret_before_act=True,
                without_norm=True
            )

            # Build embed x layer
            if self.consistency_model_params['data_x_type'] == "x_y_vx_vy":
                self.embed_x = common_layers.build_mlps(
                    c_in=4, mlp_channels=[embed_x_dim, embed_x_dim, embed_x_dim], ret_before_act=True, without_norm=True
                )
            elif self.consistency_model_params['data_x_type'] == "x_y_heading_vx_vy":
                self.embed_x = common_layers.build_mlps(
                    c_in=5, mlp_channels=[embed_x_dim, embed_x_dim, embed_x_dim], ret_before_act=True, without_norm=True
                )
            elif self.consistency_model_params['data_x_type'] == "x_y":
                self.embed_x = common_layers.build_mlps(
                    c_in=2, mlp_channels=[embed_x_dim, embed_x_dim, embed_x_dim], ret_before_act=True, without_norm=True
                )
            else:
                print("wrong data x type")
                exit()

            # Build unet output layer
            if self.consistency_model_params['rollout_within_NN'] == "True":
                self.unet_output_layer = common_layers.build_mlps(
                    c_in=embed_x_dim, mlp_channels=[embed_x_dim, embed_x_dim, 2], ret_before_act=True, without_norm=True
                )
            elif self.consistency_model_params['rollout_within_NN'] == "False":
                if self.consistency_model_params['data_x_type'] == "x_y_heading_vx_vy":
                    self.unet_output_layer = common_layers.build_mlps(
                        c_in=embed_x_dim, mlp_channels=[embed_x_dim, embed_x_dim, 5], ret_before_act=True,
                        without_norm=True
                    )
                elif self.consistency_model_params['data_x_type'] == "x_y_vx_vy":
                    self.unet_output_layer = common_layers.build_mlps(
                        c_in=embed_x_dim, mlp_channels=[embed_x_dim, embed_x_dim, 4], ret_before_act=True,
                        without_norm=True
                    )
                elif self.consistency_model_params['data_x_type'] == "x_y":
                    self.unet_output_layer = common_layers.build_mlps(
                        c_in=embed_x_dim, mlp_channels=[embed_x_dim, embed_x_dim, 2], ret_before_act=True,
                        without_norm=True
                    )
                else:
                    print("wrong data x type")
                    exit()
        
        elif consistency_model_params['build_mlp_method'] == 'unet':
            # TODO: Use build_condition_mlp
            self.obj_condition_mlp = self.build_condition_mlp(obj_condition_dim, embed_condition_layers_dims)
            self.map_condition_mlp = self.build_condition_mlp(map_condition_dim, embed_condition_layers_dims)
            self.goal_condition_mlp = self.build_condition_mlp(goal_condition_dim, embed_condition_layers_dims)

            # self.all_condition_mlp = self.build_condition_mlp(all_embedded_condition_dim, [all_condition_layer_dim])
            self.all_condition_mlp = self.build_condition_mlp(all_embedded_condition_dim, all_condition_layer_dim)

            if self.consistency_model_params['data_x_type'] == "x_y_vx_vy":
                self.embed_x = self.build_condition_mlp(4, [embed_x_dim, embed_x_dim, embed_x_dim])
            elif self.consistency_model_params['data_x_type'] == "x_y_heading_vx_vy":
                self.embed_x = self.build_condition_mlp(5, [embed_x_dim, embed_x_dim, embed_x_dim])
            elif self.consistency_model_params['data_x_type'] == "x_y":
                self.embed_x = self.build_condition_mlp(2, [embed_x_dim, embed_x_dim, embed_x_dim])
            else:
                print("wrong data x type")
                exit()

            # Build unet output layer
            if self.consistency_model_params['rollout_within_NN'] == "True":
                self.unet_output_layer = self.build_condition_mlp(embed_x_dim, [embed_x_dim, embed_x_dim, 2])
            elif self.consistency_model_params['rollout_within_NN'] == "False":
                if self.consistency_model_params['data_x_type'] == "x_y_heading_vx_vy":
                    self.unet_output_layer = self.build_condition_mlp(embed_x_dim, [embed_x_dim, embed_x_dim, 5])
                elif self.consistency_model_params['data_x_type'] == "x_y_vx_vy":
                    self.unet_output_layer = self.build_condition_mlp(embed_x_dim, [embed_x_dim, embed_x_dim, 4])
                elif self.consistency_model_params['data_x_type'] == "x_y":
                    self.unet_output_layer = self.build_condition_mlp(embed_x_dim, [embed_x_dim, embed_x_dim, 2])
                else:
                    print("wrong data x type")
                    exit()

    def build_condition_mlp(self, condition_dim, embed_condition_layers_dims):

        layers = []
        input_dim = copy.copy(condition_dim)
        for output_dim in embed_condition_layers_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.GELU())
            input_dim = output_dim

        layers.pop()

        return nn.Sequential(*layers)
