from model.mtr.utils import common_utils
import torch


def transform_trajs_to_agent_original_coords(obj_trajs_raw_data, agent_reference_xy, agent_reference_heading, heading_index=None):


    center_agent_num, surrounding_agent_num, num_timestamps, num_attrs = obj_trajs_raw_data.shape

    assert agent_reference_xy.shape[0] == agent_reference_heading.shape[0]
    assert agent_reference_xy.shape[-1] in [3, 2]
    assert center_agent_num == agent_reference_xy.shape[0], "center num is not equal to all agent num!"

    # only takes x, y, v_x, v_y from the obj_trajs
    obj_trajs = obj_trajs_raw_data.clone()

    obj_trajs[:, :, :, :2] = common_utils.rotate_points_along_z(
        points=obj_trajs[:, :, :, :2].reshape(center_agent_num * surrounding_agent_num, num_timestamps, 2),
        angle=agent_reference_heading.reshape(center_agent_num * surrounding_agent_num),
    ).reshape(center_agent_num, surrounding_agent_num, num_timestamps, 2)

    obj_trajs[:, :, :, :2] += agent_reference_xy.unsqueeze(2).expand(-1, -1, num_timestamps, -1)

    if heading_index is not None:
        obj_trajs[:, :, :, heading_index] += agent_reference_xy.unsqueeze(2).expand(-1, -1, num_timestamps)

    return obj_trajs

def transform_trajs_to_global_coords(relative_obj_trajs, center_xyz, center_heading, heading_index=None,
                                     rot_vel_index=None):
    """
    Args:
        relative_obj_trajs (num_center_objects, num_objects, num_timestamps, num_attrs):
            first three values of num_attrs are [x, y, z] or [x, y]
        center_xyz (num_center_objects, 3 or 2): [x, y, z] or [x, y]
        center_heading (num_center_objects):
        heading_index: the index of heading angle in the num_attr-axis of relative_obj_trajs
    """
    num_center_objects, num_objects, num_timestamps, num_attrs = relative_obj_trajs.shape
    assert center_xyz.shape[0] == center_heading.shape[0]
    assert center_xyz.shape[1] in [3, 2]

    # Rotate coordinates back to the global frame
    relative_obj_trajs[:, :, :, 0:2] = common_utils.rotate_points_along_z(
        points=relative_obj_trajs[:, :, :, 0:2].reshape(num_center_objects, -1, 2),
        angle=center_heading
    ).view(num_center_objects, num_objects, num_timestamps, 2)

    # Translate coordinates back to the global frame
    relative_obj_trajs[:, :, :, 0:center_xyz.shape[1]] += center_xyz[:, None, None, :]

    # Adjust heading angles back to the global frame
    if heading_index is not None:
        relative_obj_trajs[:, :, :, heading_index] += center_heading[:, None, None]

    # Rotate velocity components back to the global frame (if applicable)
    if rot_vel_index is not None:
        assert len(rot_vel_index) == 2
        relative_obj_trajs[:, :, :, rot_vel_index] = common_utils.rotate_points_along_z(
            points=relative_obj_trajs[:, :, :, rot_vel_index].reshape(num_center_objects, -1, 2),
            angle=center_heading
        ).view(num_center_objects, num_objects, num_timestamps, 2)

    return relative_obj_trajs


def prepare_data_batch(predicted_x_future, batch, args, to_transform_coordinate=True):

    # Unpack arguments #################################################################################################
    data_root = args.data_root
    result_folder = f"{args.result_folder}/{args.project_name}"
    model_name = args.model_name
    batch_size = args.test_batch_size

    # Unpack data #######################################################################################################
    # predicted_x_future: 5 or 4 dim, (x, y, heading, vx, vy) or (x, y, vx, vy)
    center_agent_num, sample_num, agent_num, timestep, x_feature_size = predicted_x_future.shape

    surrounding_obj_trajs_full, surrounding_obj_trajs_valid_mask, surrounding_obj_index, surrounding_obj_id = (
        batch['input_dict']['surrounding_obj_trajs_full'],
        batch['input_dict']['surrounding_obj_trajs_valid_mask'],
        batch['input_dict']['surrounding_obj_index'],
        batch['input_dict']['surrounding_obj_id'],)
    track_index_to_predict = batch['input_dict']['track_index_to_predict']

    center_obj_goal = batch['center_obj_trajs_goal_state']

    # find the index of center agent index, over all agent num,
    #  suppose center_agent_index = [8, 5, 2, 4, 6], center_agent_num = 5, agent_num = 10,
    #  then it means for the 1st center agent, the 8th agent of all 10 surrounding agents is the center agent
    center_agent_index = []
    for i in range(len(surrounding_obj_index)):
        # Find the index of track_index_to_predict[i] in surrounding_obj_index[i]
        try:
            index = surrounding_obj_index[i].index(track_index_to_predict[i].item())
        except:
            print("wrong surrounding obj index")
            exit()
        # Append the index to the center_agent_index list
        center_agent_index.append(index)
    # print(f"center agent index {center_agent_index}")

    # Only get the future trajectory for data x, which should be the first 5 entry (x, y, heading, vx, vy)
    surrounding_obj_trajs_future = surrounding_obj_trajs_full[:, :, 11:, :5]
    surrounding_obj_trajs_future_valid_mask = surrounding_obj_trajs_valid_mask[:, :, 11:]

    # Prepare data transformation ######################################################################################################
    center_objects = batch['input_dict']['center_objects_world']
    # use normalized and center to reconstruct data
    all_agent_reference_xy_heading_normalized = batch['input_dict']['all_agent_reference_xy_heading_normalized']
    all_agent_reference_xy_heading_center = batch['input_dict']['all_agent_reference_xy_heading_center']
    all_agent_reference_xy_heading = all_agent_reference_xy_heading_normalized.clone()
    all_agent_reference_xy_heading[:, :, :2] = all_agent_reference_xy_heading_normalized[:, :, :2] + all_agent_reference_xy_heading_center

    # get reference xy_heading for surrounding agent
    surrounding_agent_reference_xy_heading = torch.zeros((center_agent_num, agent_num, 3), device=all_agent_reference_xy_heading.device)
    for i in range(len(surrounding_obj_index)):
        current_surrounding_obj_index_list = surrounding_obj_index[i]
        for j in range(len(current_surrounding_obj_index_list)):
            curr_surrounding_index = current_surrounding_obj_index_list[j]
            if curr_surrounding_index != -1:
                surrounding_agent_reference_xy_heading[i, j, :] = all_agent_reference_xy_heading[i, curr_surrounding_index, :]

    world_coord_surrounding_obj_trajs_future = transform_trajs_to_agent_original_coords(
        obj_trajs_raw_data=surrounding_obj_trajs_future,
        agent_reference_xy=surrounding_agent_reference_xy_heading[:, :, :2],
        agent_reference_heading=surrounding_agent_reference_xy_heading[:, :, 2],
        heading_index=None,
    )


    if to_transform_coordinate:
        # Since curr_pred_x_future has shape (center_agent_num, sample_num, agent_num, timestep, x_feature_size)
        predicted_x_future = predicted_x_future.reshape(-1, agent_num, timestep, x_feature_size)

        augmented_surrounding_agent_reference_xy_heading = surrounding_agent_reference_xy_heading.unsqueeze(1).expand(-1, sample_num, -1, -1)
        augmented_surrounding_agent_reference_xy_heading = augmented_surrounding_agent_reference_xy_heading.reshape(-1, agent_num, 3)

        world_coord_pred_x_future = transform_trajs_to_agent_original_coords(
            obj_trajs_raw_data=predicted_x_future,
            agent_reference_xy=augmented_surrounding_agent_reference_xy_heading[:, :, :2],
            agent_reference_heading=augmented_surrounding_agent_reference_xy_heading[:, :, 2],
            heading_index=None
        )
        world_coord_pred_x_future = world_coord_pred_x_future.reshape(center_agent_num, sample_num, agent_num, timestep,
                                                                    x_feature_size)
    else:
        world_coord_pred_x_future = predicted_x_future

    world_coord_center_obj_goal = transform_trajs_to_global_coords(
        relative_obj_trajs=center_obj_goal.unsqueeze(1).unsqueeze(1),
        center_xyz=center_objects[:, 0:2],
        center_heading=center_objects[:, 6],
        heading_index=None, rot_vel_index=None
    )
    world_coord_center_obj_goal_pos = world_coord_center_obj_goal[:, 0, 0, :2]

    data_batch = {}
    data_batch['world_coord_pred_x_future'] = world_coord_pred_x_future
    data_batch['world_coord_surrounding_obj_trajs_future'] = world_coord_surrounding_obj_trajs_future
    data_batch['surrounding_obj_trajs_future_valid_mask'] = surrounding_obj_trajs_future_valid_mask
    data_batch['world_coord_center_obj_goal_pos'] = world_coord_center_obj_goal_pos
    data_batch['center_agent_num'] = center_agent_num
    data_batch['sample_num'] = sample_num
    data_batch['all_scenario_id'] = batch['input_dict']['scenario_id']
    data_batch['all_surrounding_obj_id'] = surrounding_obj_id

    return data_batch


def compute_planning_constraint_violation(data_batch, args):

    world_coord_pred_x_future = data_batch['world_coord_pred_x_future']
    world_coord_surrounding_obj_trajs_future = data_batch['world_coord_surrounding_obj_trajs_future']
    surrounding_obj_trajs_future_valid_mask = data_batch['surrounding_obj_trajs_future_valid_mask']
    world_coord_center_obj_goal_pos = data_batch['world_coord_center_obj_goal_pos']
    center_agent_num = data_batch['center_agent_num']
    sample_num = data_batch['sample_num']

    assert world_coord_pred_x_future.shape[0] == center_agent_num, "center agent num is not equal to the shape of world_coord_pred_x_future [0]!"
    assert world_coord_pred_x_future.shape[1] == sample_num, "sample num is not equal to the shape of world_coord_pred_x_future [1]!"
    assert sample_num == 1, "sample num is not equal to 1!"

    # Compute constraint violation
    # 1. goal reaching violation #
    pred_all_obj_future_traj_pos = world_coord_pred_x_future[..., :2]
    pred_all_obj_future_traj_pos = pred_all_obj_future_traj_pos.squeeze(1)
    pred_center_obj_future_traj_pos = pred_all_obj_future_traj_pos[:, 0]
    pred_center_obj_goal_pos = pred_center_obj_future_traj_pos[:, -1]
    center_obj_goal_pos = world_coord_center_obj_goal_pos[:, :2]

    # Compute per-sample distances
    distance_to_goal_per_sample = torch.sqrt(((pred_center_obj_goal_pos - center_obj_goal_pos) ** 2).sum(dim=-1))
    goal_reaching_violation = distance_to_goal_per_sample.mean()

    # 2. acceleration limit violation #####################################################################################
    pos_x = pred_center_obj_future_traj_pos[..., 0]
    pos_y = pred_center_obj_future_traj_pos[..., 1]

    dt = 0.1
    
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
    acc_violation = acc_violation_per_sample.mean()
    # 3. omega limit violation ###########################################################################################
    eps = 1e-6

    # Compute tangential acceleration and angular velocity
    omega = (x_dot_trunc * y_dot_dot - y_dot_trunc * x_dot_dot) / (x_dot_trunc ** 2 + y_dot_trunc ** 2 + eps)
    
    # Limits
    omega_limit = 0.3
    
    # Compute per-sample violations
    omega_violation_per_sample = torch.relu(torch.abs(omega) - omega_limit).mean(dim=-1)
    omega_violation = omega_violation_per_sample.mean()

    return center_agent_num, goal_reaching_violation, acc_violation, omega_violation

def compute_trajectory_quality(data_batch, args):

    # Get center trajectory prediction data #####################################################################################
    world_coord_pred_x_future = data_batch['world_coord_pred_x_future']
    center_agent_num = data_batch['center_agent_num']
    sample_num = data_batch['sample_num']

    assert world_coord_pred_x_future.shape[0] == center_agent_num, "center agent num is not equal to the shape of world_coord_pred_x_future [0]!"
    assert world_coord_pred_x_future.shape[1] == sample_num, "sample num is not equal to the shape of world_coord_pred_x_future [1]!"
    assert sample_num == 1, "sample num is not equal to 1!"


    pred_all_obj_future_traj_pos = world_coord_pred_x_future[..., :2]
    pred_all_obj_future_traj_pos = pred_all_obj_future_traj_pos.squeeze(1)
    pred_center_obj_future_traj_pos = pred_all_obj_future_traj_pos[:, 0]

     # Compute trajectory metrics
    angle_change = compute_delta_yaw(pred_center_obj_future_traj_pos)
    path_length = compute_path_length(pred_center_obj_future_traj_pos)
    curvature = compute_curvature(pred_center_obj_future_traj_pos)

    avg_angle_change = torch.mean(angle_change)
    avg_path_length = torch.mean(path_length)
    avg_curvature = torch.mean(curvature)

    return center_agent_num, avg_angle_change, avg_path_length, avg_curvature

def compute_delta_yaw(paths):
    """
    Compute cumulative angle changes for a batch of trajectories
    Args:
        paths: torch.Tensor of shape (batch_size, timesteps, 2)
    Returns:
        cumulative angle changes: torch.Tensor of shape (batch_size,)
    """
    # Calculate vectors between consecutive points
    vecs = paths[:, 1:] - paths[:, :-1]  # (batch_size, timesteps-1, 2)
    
    # Calculate angles of vectors
    angles = torch.atan2(vecs[..., 1], vecs[..., 0] + 1e-6)  # (batch_size, timesteps-1)
    
    # Calculate angle differences
    delAng = angles[:, 1:] - angles[:, :-1]  # (batch_size, timesteps-2)
    absDelAng = torch.abs(delAng)
    avgDelAng = torch.mean(absDelAng, dim=1)  # (batch_size,)
    
    return avgDelAng

def compute_path_length(paths):
    """
    Compute total path length for a batch of trajectories
    Args:
        paths: torch.Tensor of shape (batch_size, timesteps, 2)
    Returns:
        path lengths: torch.Tensor of shape (batch_size,)
    """
    # Calculate distances between consecutive points
    diffs = paths[:, 1:] - paths[:, :-1]  # (batch_size, timesteps-1, 2)
    lengths = torch.sqrt(torch.sum(diffs**2, dim=2))  # (batch_size, timesteps-1)
    total_lengths = torch.sum(lengths, dim=1)  # (batch_size,)
    
    return total_lengths

def compute_curvature(paths):
    """
    Compute mean curvature for a batch of trajectories
    Args:
        paths: torch.Tensor of shape (batch_size, timesteps, 2)
    Returns:
        mean curvatures: torch.Tensor of shape (batch_size,)
    """
    # Calculate first derivatives
    ds = paths[:, 1:] - paths[:, :-1]  # (batch_size, timesteps-1, 2)
    x_t = ds[..., 0]  # (batch_size, timesteps-1)
    y_t = ds[..., 1]  # (batch_size, timesteps-1)
    
    # Calculate second derivatives using gradient
    # Need to convert to list for handling batch gradient
    xx_t = torch.stack([torch.gradient(x_t[i])[0] for i in range(x_t.shape[0])])  # (batch_size, timesteps-1)
    yy_t = torch.stack([torch.gradient(y_t[i])[0] for i in range(y_t.shape[0])])  # (batch_size, timesteps-1)
    
    eps = 0.00001
    numerator = torch.abs(xx_t * y_t - x_t * yy_t)
    denominator = (x_t * x_t + y_t * y_t)**1.5 + eps
    curvature_val = numerator / denominator  # (batch_size, timesteps-1)
    
    return torch.mean(curvature_val, dim=1)  # (batch_size,)
