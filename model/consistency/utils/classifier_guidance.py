import torch


def classifier_guidance_admm(goal_reaching_guidance, acceleration_limit_guidance, angular_speed_limit_guidance, 
                        gradient_step_num, f_theta_x_t, center_obj_goal,
                        x_future_mean, x_future_std, std_padding, guidance_order='goal,acc,omega',
                        preset_goal_reaching_grad_scale=None, preset_acceleration_limit_grad_scale=None, preset_angular_speed_limit_grad_scale=None):
    
    # Store original predictions
    original_pred = f_theta_x_t.clone().detach()
    
    goal_reaching_violation_list = []
    acc_limit_violation_list = []
    omega_limit_violation_list = []

    # Parse the guidance order
    guidance_order = 'goal,acc,omega'

    guidance_steps = guidance_order.split(',')
    
    for i in range(gradient_step_num):
        # Set gradient scales
        if goal_reaching_guidance == "True":
            if preset_goal_reaching_grad_scale is not None:
                goal_reaching_grad_scale = preset_goal_reaching_grad_scale
            else:
                goal_reaching_grad_scale = 2 * 1e-5
        else:
            goal_reaching_grad_scale = 0
        
        if acceleration_limit_guidance == "True":
            if preset_acceleration_limit_grad_scale is not None:
                acc_limit_grad_scale = preset_acceleration_limit_grad_scale
            else:
                acc_limit_grad_scale = 3 * 1e-6
        else:
            acc_limit_grad_scale = 0
        
        if angular_speed_limit_guidance == "True":
            if preset_angular_speed_limit_grad_scale is not None:
                omega_limit_grad_scale = preset_angular_speed_limit_grad_scale
            else:
                omega_limit_grad_scale = 5 * 1e-7
        else:
            omega_limit_grad_scale = 0
        
        # Apply guidance in the specified order
        for step in guidance_steps:
            step = step.strip().lower()
            
            if step == 'goal' and goal_reaching_guidance == "True":
                # Goal reaching guidance
                goal_reaching_grad, goal_reaching_violation = guidance_goal_reaching(
                    f_theta_x_t, center_obj_goal, x_future_mean, x_future_std, std_padding
                )
                
                goal_reaching_grad = torch.nan_to_num(goal_reaching_grad, 0.0)
                
                max_norm = 2.0
                with torch.no_grad():
                    f_theta_x_t = f_theta_x_t - (goal_reaching_grad_scale * goal_reaching_grad)
                    
                    # Gradient clipping
                    grad_norms = torch.norm((f_theta_x_t - original_pred).view(f_theta_x_t.shape[0], -1), dim=1)
                    scale = torch.clamp(max_norm / (grad_norms + 1e-6), max=1.0)
                    f_theta_x_t = original_pred + (f_theta_x_t - original_pred) * scale.view(-1, 1, 1, 1)
                    original_pred = f_theta_x_t.clone().detach()
                
                # Record violation for this step
                goal_reaching_violation_list.append(goal_reaching_violation)
            
            elif step == 'acc' and acceleration_limit_guidance == "True":
                # Acceleration limit guidance
                acc_limit_grad, acc_limit_violation = guidance_acc_limit(
                    f_theta_x_t, x_future_mean, x_future_std, std_padding
                )
                
                acc_limit_grad = torch.nan_to_num(acc_limit_grad, 0.0)
                
                max_norm = 2.0
                with torch.no_grad():
                    f_theta_x_t = f_theta_x_t - (acc_limit_grad_scale * acc_limit_grad)
                    
                    # Gradient clipping
                    grad_norms = torch.norm((f_theta_x_t - original_pred).view(f_theta_x_t.shape[0], -1), dim=1)
                    scale = torch.clamp(max_norm / (grad_norms + 1e-6), max=1.0)
                    f_theta_x_t = original_pred + (f_theta_x_t - original_pred) * scale.view(-1, 1, 1, 1)
                    original_pred = f_theta_x_t.clone().detach()
                
                # Record violation for this step
                acc_limit_violation_list.append(acc_limit_violation)
            
            elif step == 'omega' and angular_speed_limit_guidance == "True":
                # Angular speed limit guidance
                omega_limit_grad, omega_limit_violation = guidance_omega_limit(
                    f_theta_x_t, x_future_mean, x_future_std, std_padding
                )
                
                omega_limit_grad = torch.nan_to_num(omega_limit_grad, 0.0)
                
                max_norm = 2.0
                with torch.no_grad():
                    f_theta_x_t = f_theta_x_t - (omega_limit_grad_scale * omega_limit_grad)
                    
                    # Gradient clipping
                    grad_norms = torch.norm((f_theta_x_t - original_pred).view(f_theta_x_t.shape[0], -1), dim=1)
                    scale = torch.clamp(max_norm / (grad_norms + 1e-6), max=1.0)
                    f_theta_x_t = original_pred + (f_theta_x_t - original_pred) * scale.view(-1, 1, 1, 1)
                    original_pred = f_theta_x_t.clone().detach()
                
                # Record violation for this step
                omega_limit_violation_list.append(omega_limit_violation)
        
        # Check for NaNs
        if (goal_reaching_guidance == "True" and torch.isnan(goal_reaching_violation).any()) or \
           (acceleration_limit_guidance == "True" and torch.isnan(acc_limit_violation).any()) or \
           (angular_speed_limit_guidance == "True" and torch.isnan(omega_limit_violation).any()):
            print(f"There is nan in the gradient step {i}")
            exit()

    return f_theta_x_t, goal_reaching_violation_list, acc_limit_violation_list, omega_limit_violation_list


def guidance_goal_reaching(f_theta_x_t, center_obj_goal, x_future_mean, x_future_std, std_padding):
    with torch.enable_grad():

        x_in = f_theta_x_t.clone().detach().requires_grad_(True)
        
        # Unnormalize predictions
        pred_all_obj_future_traj_pos = x_in[..., :2]
        pred_all_obj_future_traj_pos = pred_all_obj_future_traj_pos * (x_future_std[..., :2] + std_padding) + x_future_mean[..., :2]
        
        # Get center agent trajectory and goal position
        pred_center_obj_future_traj_pos = pred_all_obj_future_traj_pos[:, 0]
        pred_center_obj_goal_pos = pred_center_obj_future_traj_pos[:, -1]
        center_obj_goal_pos = center_obj_goal[:, :2]
        
        # Compute per-sample distances
        distance_to_goal_per_sample = torch.sqrt(((pred_center_obj_goal_pos - center_obj_goal_pos) ** 2).sum(dim=-1))
        
        # Create grad_outputs for per-sample gradients
        grad_outputs = torch.ones_like(distance_to_goal_per_sample)
        
        # Compute per-sample gradients
        grad = torch.autograd.grad(
            distance_to_goal_per_sample, 
            x_in, 
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False
        )[0]

            
    return grad, distance_to_goal_per_sample.mean()

def guidance_acc_limit(f_theta_x_t, x_future_mean, x_future_std, std_padding):
    with torch.enable_grad():
        x_in = f_theta_x_t.clone().detach().requires_grad_(True)
        dt = 0.1
        
        # Unnormalize predictions
        pred_all_obj_future_traj_pos = x_in[..., :2]
        pred_all_obj_future_traj_pos = pred_all_obj_future_traj_pos * (x_future_std[..., :2] + std_padding) + x_future_mean[..., :2]
        
        # Get center agent trajectory
        pred_center_obj_future_traj_pos = pred_all_obj_future_traj_pos[:, 0]
        
        pos_x = pred_center_obj_future_traj_pos[..., 0]
        pos_y = pred_center_obj_future_traj_pos[..., 1]
        
        # Compute velocities
        x_dot = (pos_x[:, 1:] - pos_x[:, :-1]) / dt
        y_dot = (pos_y[:, 1:] - pos_y[:, :-1]) / dt
        
        # Truncate velocities for acceleration computation
        x_dot_trunc = x_dot[:, :-1]
        y_dot_trunc = y_dot[:, :-1]
        
        # Compute accelerations
        x_dot_dot = (x_dot[:, 1:] - x_dot[:, :-1]) / dt
        y_dot_dot = (y_dot[:, 1:] - y_dot[:, :-1]) / dt
        
        eps = 1e-6
        # Compute tangential acceleration and angular velocity
        acc = (x_dot_dot * x_dot_trunc + y_dot_dot * y_dot_trunc) / (torch.sqrt(x_dot_trunc ** 2 + y_dot_trunc ** 2) + eps)

        # Limits
        acc_limit = 6.0
        
        # Compute per-sample violations
        acc_violation_per_sample = torch.relu(torch.abs(acc) - acc_limit).mean(dim=-1)
        
        # Create grad_outputs for per-sample gradients
        grad_outputs_acc = torch.ones_like(acc_violation_per_sample)
        
        # Compute per-sample gradients
        acc_grad = torch.autograd.grad(
            acc_violation_per_sample, 
            x_in, 
            grad_outputs=grad_outputs_acc,
            create_graph=False,
            retain_graph=False,
        )[0]
        
            
    return acc_grad, acc_violation_per_sample.mean()

def guidance_omega_limit(f_theta_x_t, x_future_mean, x_future_std, std_padding):
    with torch.enable_grad():
        x_in = f_theta_x_t.clone().detach().requires_grad_(True)
        dt = 0.1
        
        # Unnormalize predictions
        pred_all_obj_future_traj_pos = x_in[..., :2]
        pred_all_obj_future_traj_pos = pred_all_obj_future_traj_pos * (x_future_std[..., :2] + std_padding) + x_future_mean[..., :2]
        
        # Get center agent trajectory
        pred_center_obj_future_traj_pos = pred_all_obj_future_traj_pos[:, 0]
        
        pos_x = pred_center_obj_future_traj_pos[..., 0]
        pos_y = pred_center_obj_future_traj_pos[..., 1]
        
        # Compute velocities
        x_dot = (pos_x[:, 1:] - pos_x[:, :-1]) / dt
        y_dot = (pos_y[:, 1:] - pos_y[:, :-1]) / dt
        
        # Truncate velocities for acceleration computation
        x_dot_trunc = x_dot[:, :-1]
        y_dot_trunc = y_dot[:, :-1]
        
        # Compute accelerations
        x_dot_dot = (x_dot[:, 1:] - x_dot[:, :-1]) / dt
        y_dot_dot = (y_dot[:, 1:] - y_dot[:, :-1]) / dt
        
        eps = 1e-6

        # Compute tangential acceleration and angular velocity
        omega = (x_dot_trunc * y_dot_dot - y_dot_trunc * x_dot_dot) / (x_dot_trunc ** 2 + y_dot_trunc ** 2 + eps)
        
        # Limits
        omega_limit = 0.3
        
        # Compute per-sample violations
        omega_violation_per_sample = torch.relu(torch.abs(omega) - omega_limit).mean(dim=-1)
        
        # Create grad_outputs for per-sample gradients
        grad_outputs_omega = torch.ones_like(omega_violation_per_sample)
        
        # Compute per-sample gradients
        omega_grad = torch.autograd.grad(
            omega_violation_per_sample, 
            x_in, 
            grad_outputs=grad_outputs_omega,
            create_graph=False,
            retain_graph=False,
        )[0]
        
        # print(f"curr_sample_step {curr_sample_step}, gradient step {gradient_step_num}, omega_violation is {omega_violation_per_sample.mean()}")

            
    return omega_grad, omega_violation_per_sample.mean()