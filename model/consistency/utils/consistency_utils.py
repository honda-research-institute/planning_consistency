import torch

def roll_out_x_y_heading_vx_vy(output_action, x_current):
    # Extract initial values
    curr_x = x_current[:, :, :, 0].squeeze(-1)  # Shape: (batch_size, 10)
    curr_y = x_current[:, :, :, 1].squeeze(-1)  # Shape: (batch_size, 10)
    curr_psi = x_current[:, :, :, 2].squeeze(-1)  # Shape: (batch_size, 10)
    curr_v_x = x_current[:, :, :, 3].squeeze(-1)  # Shape: (batch_size, 10)
    curr_v_y = x_current[:, :, :, 4].squeeze(-1)  # Shape: (batch_size, 10)

    action_v_dot = output_action[:, :, :, 0]  # Shape: (batch_size, 10, 80)
    action_psi_dot = output_action[:, :, :, 1]  # Shape: (batch_size, 10, 80)

    delta_t = 0.1
    steps = 80

    # Compute cumulative sums for actions
    cumsum_v_dot = torch.cumsum(action_v_dot, dim=2) * delta_t  # Shape: (batch_size, 10, 80)
    cumsum_psi_dot = torch.cumsum(action_psi_dot, dim=2) * delta_t  # Shape: (batch_size, 10, 80)

    # Compute new angles
    curr_psi_expanded = curr_psi.unsqueeze(2).expand(-1, -1, steps)  # Shape: (batch_size, 10, 80)
    new_psi = curr_psi_expanded + cumsum_psi_dot  # Shape: (batch_size, 10, 80)

    # Compute initial velocity magnitude
    curr_v = torch.sqrt(curr_v_x ** 2 + curr_v_y ** 2).unsqueeze(2)  # Shape: (batch_size, 10, 1)

    # Compute new velocities
    curr_v_expanded = curr_v.expand(-1, -1, steps)  # Shape: (batch_size, 10, 80)
    new_v = curr_v_expanded + cumsum_v_dot  # Shape: (batch_size, 10, 80)

    # Compute new velocity components
    new_v_x = new_v * torch.cos(new_psi)  # Shape: (batch_size, 10, 80)
    new_v_y = new_v * torch.sin(new_psi)  # Shape: (batch_size, 10, 80)

    # Compute new positions using cumulative sum
    delta_v_x = new_v_x * delta_t  # Shape: (batch_size, 10, 80)
    delta_v_y = new_v_y * delta_t  # Shape: (batch_size, 10, 80)

    cumsum_delta_v_x = torch.cumsum(delta_v_x, dim=2)  # Shape: (batch_size, 10, 80)
    cumsum_delta_v_y = torch.cumsum(delta_v_y, dim=2)  # Shape: (batch_size, 10, 80)

    new_x = curr_x.unsqueeze(2) + cumsum_delta_v_x  # Shape: (batch_size, 10, 80)
    new_y = curr_y.unsqueeze(2) + cumsum_delta_v_y  # Shape: (batch_size, 10, 80)

    # Prepare output tensor
    predict_normalized_x_future = torch.zeros((x_current.shape[0], x_current.shape[1], steps, x_current.shape[3]), device=x_current.device)  # Shape: (batch_size, 10, 80, 5)

    # Assign computed values to the output tensor
    predict_normalized_x_future[:, :, :, 0] = new_x
    predict_normalized_x_future[:, :, :, 1] = new_y
    predict_normalized_x_future[:, :, :, 2] = new_psi
    predict_normalized_x_future[:, :, :, 3] = new_v_x
    predict_normalized_x_future[:, :, :, 4] = new_v_y

    return predict_normalized_x_future


def roll_out_x_y_vx_vy(output_action, x_current):
    # Extract initial values
    curr_x = x_current[:, :, :, 0].squeeze(-1)  # Shape: (batch_size, 10)
    curr_y = x_current[:, :, :, 1].squeeze(-1)  # Shape: (batch_size, 10)
    curr_v_x = x_current[:, :, :, 2].squeeze(-1)  # Shape: (batch_size, 10)
    curr_v_y = x_current[:, :, :, 3].squeeze(-1)  # Shape: (batch_size, 10)

    action_a_x_dot = output_action[:, :, :, 0]  # Shape: (batch_size, 10, 80)
    action_a_y_dot = output_action[:, :, :, 1]  # Shape: (batch_size, 10, 80)

    delta_t = 0.1
    steps = 80

    # Compute cumulative sums for actions, cumsum: Returns the cumulative sum of elements of input in the dimension dim.
    cumsum_a_x_dot = torch.cumsum(action_a_x_dot, dim=2) * delta_t  # Shape: (batch_size, 10, 80)
    cumsum_a_y_dot = torch.cumsum(action_a_y_dot, dim=2) * delta_t  # Shape: (batch_size, 10, 80)

    # Compute new velocities
    curr_v_x_expanded = curr_v_x.unsqueeze(2).expand(-1, -1, steps)  # Shape: (batch_size, 10, 80)
    curr_v_y_expanded = curr_v_y.unsqueeze(2).expand(-1, -1, steps)  # Shape: (batch_size, 10, 80)
    new_v_x = curr_v_x_expanded + cumsum_a_x_dot  # Shape: (batch_size, 10, 80)
    new_v_y = curr_v_y_expanded + cumsum_a_y_dot  # Shape: (batch_size, 10, 80)

    # Compute new positions using cumulative sum
    cumsum_v_x = torch.cumsum(new_v_x, dim=2) * delta_t  # Shape: (batch_size, 10, 80)
    cumsum_v_y = torch.cumsum(new_v_y, dim=2) * delta_t  # Shape: (batch_size, 10, 80)

    # Compute new positions
    curr_x_expanded = curr_x.unsqueeze(2).expand(-1, -1, steps)  # Shape: (batch_size, 10, 80)
    curr_y_expanded = curr_y.unsqueeze(2).expand(-1, -1, steps)
    new_x = curr_x_expanded + cumsum_v_x  # Shape: (batch_size, 10, 80)
    new_y = curr_y_expanded + cumsum_v_y  # Shape: (batch_size, 10, 80)

    # Prepare output tensor
    predict_normalized_x_future = torch.zeros((x_current.shape[0], x_current.shape[1], steps, 4),
                                              device=x_current.device)  # Shape: (batch_size, 10, 80, 4)

    # Assign computed values to the output tensor
    predict_normalized_x_future[:, :, :, 0] = new_x
    predict_normalized_x_future[:, :, :, 1] = new_y
    predict_normalized_x_future[:, :, :, 2] = new_v_x
    predict_normalized_x_future[:, :, :, 3] = new_v_y

    return predict_normalized_x_future

def roll_out_x_y_old_fixed(output_action, x_current):
    # TODO: roll out with previous cumulative setting

    # Extract initial positions and velocities
    curr_x = x_current[:, :, :, 0].squeeze(-1)  # (agent_num, surrounding_agent_num)
    curr_y = x_current[:, :, :, 1].squeeze(-1)  # (agent_num, surrounding_agent_num)
    curr_v_x = x_current[:, :, :, 2].squeeze(-1)  # (agent_num, surrounding_agent_num)
    curr_v_y = x_current[:, :, :, 3].squeeze(-1)  # (agent_num, surrounding_agent_num)

    action_a_x_dot = output_action[:, :, :, 0]  # (agent_num, surrounding_agent_num, 80)
    action_a_y_dot = output_action[:, :, :, 1]  # (agent_num, surrounding_agent_num, 80)

    delta_t = 0.1
    steps = 80

    # Compute cumulative sum of accelerations over timesteps
    cumulative_a_x = torch.cumsum(action_a_x_dot, dim=2) * delta_t  # Integrate acceleration
    cumulative_a_y = torch.cumsum(action_a_y_dot, dim=2) * delta_t  # Integrate acceleration

    # Compute velocities at each timestep by adding initial velocity
    v_x = curr_v_x.unsqueeze(2) + cumulative_a_x  # (agent_num, surrounding_agent_num, 80)
    v_y = curr_v_y.unsqueeze(2) + cumulative_a_y  # (agent_num, surrounding_agent_num, 80)

    # Compute cumulative sum of velocities to get positions at each timestep
    cumulative_v_x = torch.cumsum(v_x, dim=2) * delta_t  # Integrate velocity
    cumulative_v_y = torch.cumsum(v_y, dim=2) * delta_t  # Integrate velocity

    # Compute final positions by adding the initial positions
    x = curr_x.unsqueeze(2) + cumulative_v_x  # (agent_num, surrounding_agent_num, 80)
    y = curr_y.unsqueeze(2) + cumulative_v_y  # (agent_num, surrounding_agent_num, 80)

    # Prepare output tensor for predicted future positions
    predict_x_future = torch.zeros((x_current.shape[0], x_current.shape[1], steps, 2), device=x_current.device)
    predict_x_future[:, :, :, 0] = x
    predict_x_future[:, :, :, 1] = y

    return predict_x_future

def roll_out_x_y_new_integration(output_action, x_current, delta_t=0.1):
    """
    Rolls out dynamics with new accumulation of initial velocity and acceleration effects.
    """
    # Extract initial states
    curr_x = x_current[..., 0].squeeze(-1)
    curr_y = x_current[..., 1].squeeze(-1)
    curr_v_x = x_current[..., 2].squeeze(-1)
    curr_v_y = x_current[..., 3].squeeze(-1)
    
    steps = output_action.shape[2]  # 80 timesteps
    
    # 1. Initial velocity contribution
    # Initial velocity affects position linearly with time
    # At t=1: v0*dt
    # At t=2: 2*v0*dt
    # At t=3: 3*v0*dt, etc.
    time_steps = torch.arange(1, steps + 1, device=x_current.device) * delta_t
    init_v_x_contribution = curr_v_x.unsqueeze(-1) * time_steps
    init_v_y_contribution = curr_v_y.unsqueeze(-1) * time_steps
    
    # 2. Acceleration contribution
    # For t=1: x(1) = x0 + v0*dt + a0*dt²/2
    # For t=2: x(2) = x0 + 2v0*dt + (2a0*dt² + a1*dt²)/2
    # For t=3: x(3) = x0 + 3v0*dt + (3a0*dt² + 2a1*dt² + a2*dt²)/2
    # First compute velocities at each timestep
    a_x = output_action[..., 0]
    a_y = output_action[..., 1]
    v_changes_x = torch.cumsum(a_x * delta_t, dim=-1)
    v_changes_y = torch.cumsum(a_y * delta_t, dim=-1)
    
    # These velocity changes affect future positions
    # We need to integrate these velocity changes
    pos_from_v_changes_x = torch.cumsum(v_changes_x, dim=-1) * delta_t
    pos_from_v_changes_y = torch.cumsum(v_changes_y, dim=-1) * delta_t
    
    # 3. Combine all effects
    x = curr_x.unsqueeze(-1) + init_v_x_contribution + pos_from_v_changes_x
    y = curr_y.unsqueeze(-1) + init_v_y_contribution + pos_from_v_changes_y
    
    # Stack for output
    predict_x_future = torch.stack([x, y], dim=-1)
    
    return predict_x_future
    

def roll_out_for_loop(output_action, x_current):

    curr_x = x_current[:, :, :, 0].squeeze()
    curr_y = x_current[:, :, :, 1].squeeze()
    curr_psi = x_current[:, :, :, 2].squeeze()
    curr_v_x = x_current[:, :, :, 3].squeeze()
    curr_v_y = x_current[:, :, :, 4].squeeze()

    action_v_dot = output_action[:, :, :, 0]
    action_psi_dot = output_action[:, :, :, 1]

    delta_t = 0.1

    predict_normalized_x_future = torch.zeros((x_current.shape[0], x_current.shape[1],
                                               80, x_current.shape[3]), device=x_current.device)

    for i in range(80):
        new_curr_x = curr_x + curr_v_x * delta_t
        new_curr_y = curr_y + curr_v_y * delta_t
        new_curr_psi = curr_psi + action_psi_dot[:, :, i] * delta_t
        curr_v = torch.sqrt(curr_v_x ** 2 + curr_v_y ** 2) + action_v_dot[:, :, i] * delta_t
        new_curr_v_x = curr_v * torch.cos(new_curr_psi)
        new_curr_v_y = curr_v * torch.sin(new_curr_psi)

        predict_normalized_x_future[:, :, i, 0] = new_curr_x
        predict_normalized_x_future[:, :, i, 1] = new_curr_y
        predict_normalized_x_future[:, :, i, 2] = new_curr_psi
        predict_normalized_x_future[:, :, i, 3] = new_curr_v_x
        predict_normalized_x_future[:, :, i, 4] = new_curr_v_y

        curr_x = new_curr_x
        curr_y = new_curr_y
        curr_psi = new_curr_psi
        curr_v_x = new_curr_v_x
        curr_v_y = new_curr_v_y

    return predict_normalized_x_future

if __name__ == '__main__':
    # Example usage
    # output_action = torch.randn(11, 10, 80, 2)
    # x_current = torch.randn(11, 10, 1, 5)
    # predicted_future = roll_out_x_y_heading_vx_vy(output_action, x_current)
    # print(predicted_future.shape)  # Expected shape: (11, 10, 80, 5)

    # output_action = torch.randn(11, 10, 80, 2)
    # x_current = torch.randn(11, 10, 1, 4)
    # predicted_future = roll_out_x_y_vx_vy(output_action, x_current)
    # print(predicted_future.shape)  # Expected shape: (11, 10, 80, 5)

    output_action = torch.randn(11, 5, 80, 2)
    x_current = torch.randn(11, 5, 1, 4)
    predicted_future = roll_out_x_y(output_action, x_current)
    print(predicted_future.shape)  # Expected shape: (11, 5, 80, 2)
