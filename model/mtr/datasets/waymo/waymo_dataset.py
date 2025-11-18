# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import os
import numpy as np
from pathlib import Path
import pickle
import torch
import random

from model.mtr.datasets.dataset import DatasetTemplate
from model.mtr.utils import common_utils
from model.mtr.config import cfg


class WaymoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, training=True, logger=None, downsample_ratio=1.0, args=None):
        super().__init__(dataset_cfg=dataset_cfg, training=training, logger=logger)
        self.data_root = cfg.ROOT_DIR / self.dataset_cfg.DATA_ROOT
        self.data_path = self.data_root / self.dataset_cfg.SPLIT_DIR[self.mode]
        info_path = self.data_root / self.dataset_cfg.INFO_FILE[self.mode]
        self.infos = self.get_all_infos(info_path)

        # Filter out those data that has > 128 past valid agents in a scenario
        if self.dataset_cfg.FILTER_INFO_BY_AGENT_NUM:
            self.filter_info_by_agent_num()

        # Only choose certain scenarios if specified
        if args.selected_scenario_ids != []:
            self.infos = [info for info in self.infos if info['scenario_id'] in args.selected_scenario_ids]
        
        if self.infos == []:
            print("selected scenarios are not in the data path!")
            exit()

        # Downsample the dataset
        if downsample_ratio < 1.0:
            downsample_size = int(len(self.infos) * downsample_ratio)
            self.infos = random.sample(self.infos, downsample_size)
        self.logger.info(f'Total scenes after filter object, agent_num and downsample: {len(self.infos)}')

        self.surrounding_k = args.surrounding_k
        self.distance_threshold = args.distance_threshold
        self.surrounding_distance_metric = args.surrounding_distance_metric

        self.all_agent_reference_state_type = args.all_agent_reference_state_type

    def filter_info_by_agent_num(self):

        filtered_info = []
        max_valid_past_agent_num = - np.inf
        for info in self.infos:
            curr_valid_past_agent_num = info['valid_past_agent_num']
            if curr_valid_past_agent_num <= 128:
                filtered_info.append(info)
            max_valid_past_agent_num = max(max_valid_past_agent_num, curr_valid_past_agent_num)

        self.infos = filtered_info

        self.logger.info(f'max valid past agent num: {max_valid_past_agent_num}')
        self.logger.info(f'Total scenes after filter object, agent_num: {len(self.infos)}')

    def get_all_infos(self, info_path):
        self.logger.info(f'Start to load infos from {info_path}')
        with open(info_path, 'rb') as f:
            src_infos = pickle.load(f)

        infos = src_infos[::self.dataset_cfg.SAMPLE_INTERVAL[self.mode]]
        self.logger.info(f'Total scenes before filters and downsample: {len(infos)}')

        # Filter out those object_type that is not desired. Set valid_mask to be False
        for func_name, val in self.dataset_cfg.INFO_FILTER_DICT.items():
            infos = getattr(self, func_name)(infos, val)

        return infos

    def filter_info_by_object_type(self, infos, valid_object_types=None):
        ret_infos = []
        for cur_info in infos:
            num_interested_agents = cur_info['tracks_to_predict']['track_index'].__len__()
            if num_interested_agents == 0:
                continue

            valid_mask = []
            for idx, cur_track_index in enumerate(cur_info['tracks_to_predict']['track_index']):
                valid_mask.append(cur_info['tracks_to_predict']['object_type'][idx] in valid_object_types)

            valid_mask = np.array(valid_mask) > 0
            if valid_mask.sum() == 0:
                continue

            assert len(cur_info['tracks_to_predict'].keys()) == 3, f"{cur_info['tracks_to_predict'].keys()}"
            cur_info['tracks_to_predict']['track_index'] = list(np.array(cur_info['tracks_to_predict']['track_index'])[valid_mask])
            cur_info['tracks_to_predict']['object_type'] = list(np.array(cur_info['tracks_to_predict']['object_type'])[valid_mask])
            cur_info['tracks_to_predict']['difficulty'] = list(np.array(cur_info['tracks_to_predict']['difficulty'])[valid_mask])

            ret_infos.append(cur_info)
        self.logger.info(f'Total scenes after filter_info_by_object_type: {len(ret_infos)}')
        return ret_infos

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        # print(f"Called __getitem__ for index {index}")
        ret_infos = self.create_scene_level_data(index)

        return ret_infos

    def create_scene_level_data(self, index):
        """
        Args:
            index (index):

        Returns:

        """
        # Here although the filtered info doesn't include other agent types,
        #  it is not used when loading the information.
        #  Thus we will save the filtered info and use it to update the info data to load
        filtered_info = self.infos[index]
        scene_id = filtered_info['scenario_id']
        with open(self.data_path / f'sample_{scene_id}.pkl', 'rb') as f:
            info = pickle.load(f)

        # Update the newly loaded info, especially the tracks to predict
        info['tracks_to_predict'] = filtered_info['tracks_to_predict']

        sdc_track_index = info['sdc_track_index']
        current_time_index = info['current_time_index']
        timestamps = np.array(info['timestamps_seconds'][:current_time_index + 1], dtype=np.float32)

        track_infos = info['track_infos']

        track_index_to_predict = np.array(info['tracks_to_predict']['track_index'])
        obj_types = np.array(track_infos['object_type'])
        obj_ids = np.array(track_infos['object_id'])
        obj_trajs_full = track_infos['trajs']  # (num_objects, num_timestamp, 10)
        obj_trajs_past = obj_trajs_full[:, :current_time_index + 1]
        obj_trajs_future = obj_trajs_full[:, current_time_index + 1:]

        # center_objects, this is the center objects' current-time traj, used for transforming coordinate system
        center_objects, track_index_to_predict = self.get_interested_agents(
            track_index_to_predict=track_index_to_predict,
            obj_trajs_full=obj_trajs_full,
            current_time_index=current_time_index,
            obj_types=obj_types, scene_id=scene_id
        )

        # Here obj is all relative position, for each center obj.
        #  The invalid surrounding objects will be filtered out
        #  Thus the format will be: obj_trajs_future_state, (num_center_objects, num_objects, num_timestamps_future, 4):  [x, y, vx, vy]
        (obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos, obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs,
            center_gt_trajs_mask, center_gt_final_valid_idx,
            track_index_to_predict_new, sdc_track_index_new, obj_types, obj_ids,
         surrounding_obj_trajs_full, surrounding_obj_index, surrounding_obj_id, surrounding_obj_trajs_valid_mask,
         all_agent_reference_xy_heading) = (
            self.create_agent_data_for_center_objects(
            center_objects=center_objects,
            obj_trajs_past=obj_trajs_past, obj_trajs_future=obj_trajs_future,
            track_index_to_predict=track_index_to_predict, sdc_track_index=sdc_track_index,
            timestamps=timestamps, obj_types=obj_types, obj_ids=obj_ids
        ))

        # We can normalize all_agent_reference_xy_heading by subtracting the position of the first agent
        all_agent_reference_xy_center = all_agent_reference_xy_heading[0, track_index_to_predict_new, :2].unsqueeze(1)
        all_agent_reference_xy_heading_normalized = all_agent_reference_xy_heading.clone()
        all_agent_reference_xy_heading_normalized[:, :, :2] -= all_agent_reference_xy_center

        ret_dict = {
            'scenario_id': np.array([scene_id] * len(track_index_to_predict)),
            'obj_trajs': obj_trajs_data,
            'obj_trajs_mask': obj_trajs_mask,
            'track_index_to_predict': track_index_to_predict_new,  # used to select center-features
            'obj_trajs_pos': obj_trajs_pos,
            'obj_trajs_last_pos': obj_trajs_last_pos,
            'obj_types': obj_types,
            'obj_ids': obj_ids,

            'center_objects_world': center_objects,
            'center_objects_id': np.array(track_infos['object_id'])[track_index_to_predict],
            'center_objects_type': np.array(track_infos['object_type'])[track_index_to_predict],

            'obj_trajs_future_state': obj_trajs_future_state[:, :, :, :4],  # only the first 4 (x, y, vx, vy) are the original states
            'obj_trajs_future_mask': obj_trajs_future_mask,
            'center_gt_trajs': center_gt_trajs[:, :, :4],  # only the first 4 (x, y, vx, vy) are the original states,
            'center_gt_trajs_mask': center_gt_trajs_mask,
            'center_gt_final_valid_idx': center_gt_final_valid_idx,
            'center_gt_trajs_src': obj_trajs_full[track_index_to_predict],

            'surrounding_obj_trajs_full': surrounding_obj_trajs_full,  # surrounding agents full traj (x, y, heading, vx, vy, acc_x, acc_y, angular_speed)
            'surrounding_obj_index': surrounding_obj_index, # -1 means invalid
            'surrounding_obj_id': surrounding_obj_id,
            'surrounding_obj_trajs_valid_mask': surrounding_obj_trajs_valid_mask,
            'all_agent_reference_xy_heading_center': all_agent_reference_xy_center.numpy(), # center of the reference x,y
            'all_agent_reference_xy_heading_normalized': all_agent_reference_xy_heading_normalized.numpy() # normalized x,y (minus the center) and heading
        }

        if not self.dataset_cfg.get('WITHOUT_HDMAP', False):
            if info['map_infos']['all_polylines'].__len__() == 0:
                info['map_infos']['all_polylines'] = np.zeros((2, 7), dtype=np.float32)
                print(f'Warning: empty HDMap {scene_id}')

            map_polylines_data, map_polylines_mask, map_polylines_center = self.create_map_data_for_center_objects(
                center_objects=center_objects, map_infos=info['map_infos'],
                center_offset=self.dataset_cfg.get('CENTER_OFFSET_OF_MAP', (30.0, 0)),
            )   # (num_center_objects, num_topk_polylines, num_points_each_polyline, 9), (num_center_objects, num_topk_polylines, num_points_each_polyline)

            ret_dict['map_polylines'] = map_polylines_data
            ret_dict['map_polylines_mask'] = (map_polylines_mask > 0)
            ret_dict['map_polylines_center'] = map_polylines_center

        return ret_dict

    def create_agent_data_for_center_objects(
            self, center_objects, obj_trajs_past, obj_trajs_future, track_index_to_predict, sdc_track_index, timestamps,
            obj_types, obj_ids
        ):

        (obj_trajs_data, obj_trajs_mask,
         obj_trajs_future_state, obj_trajs_future_mask,
         obj_trajs_full, obj_trajs_full_valid_mask) = self.generate_centered_trajs_for_agents(
            center_objects=center_objects, obj_trajs_past=obj_trajs_past,
            obj_types=obj_types, center_indices=track_index_to_predict,
            sdc_index=sdc_track_index, timestamps=timestamps, obj_trajs_future=obj_trajs_future,
        )

        # Filter invalid past trajs pretty early.
        assert obj_trajs_past.__len__() == obj_trajs_data.shape[1]
        # Since the last dimension of obj_trajs_past is valid flag
        valid_past_mask = np.logical_not(obj_trajs_past[:, :, -1].sum(axis=-1) == 0)  # (num_objects (original))

        obj_trajs_data = obj_trajs_data[:, valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps, C)
        obj_trajs_mask = obj_trajs_mask[:, valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps)
        obj_trajs_future_state = obj_trajs_future_state[:, valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps_future, 4):  [x, y, vx, vy]
        obj_trajs_future_mask = obj_trajs_future_mask[:, valid_past_mask]  # (num_center_objects, num_objects, num_timestamps_future):

        obj_trajs_past = obj_trajs_past[valid_past_mask]
        obj_trajs_future = obj_trajs_future[valid_past_mask]

        # Now we do not use obj_trajs_full for computing surrounding obj
        obj_trajs_full = obj_trajs_full[:, valid_past_mask]
        obj_trajs_full_valid_mask = obj_trajs_full_valid_mask[:, valid_past_mask]

        obj_types = obj_types[valid_past_mask]
        obj_ids = obj_ids[valid_past_mask]

        valid_index_cnt = valid_past_mask.cumsum(axis=0)
        track_index_to_predict_new = valid_index_cnt[track_index_to_predict] - 1
        sdc_track_index_new = valid_index_cnt[sdc_track_index] - 1  # CHECK THIS

        assert obj_trajs_future_state.shape[1] == obj_trajs_data.shape[1]
        assert len(obj_types) == obj_trajs_future_mask.shape[1]
        assert len(obj_ids) == obj_trajs_future_mask.shape[1]

        # Get local data
        # compute trajectory in each car's local coordinates, used for consistency model, thus only (x, y)
        #  obj_trajs_full_raw_data: (agent_num, 91, 10) [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid].
        # We choose the first timestep that is valid. For center car, there must be a valid timestep before 11 (current timestep)
        obj_trajs_full_raw_data = torch.cat((torch.from_numpy(obj_trajs_past).float(), torch.from_numpy(obj_trajs_future).float()), dim=1)
        all_agent_reference_states, valid_timestep = self.get_all_agent_reference_states(
            data=obj_trajs_full_raw_data, type=self.all_agent_reference_state_type)
        all_agent_reference_xy = all_agent_reference_states[:, :2]
        all_agent_reference_heading = all_agent_reference_states[:, 6]
        all_agent_reference_xy_heading = all_agent_reference_states[:, [0, 1, 6]]
        all_agent_local_obj_trajs = self.transform_trajs_to_agent_local_coords(
            obj_trajs_raw_data=obj_trajs_full_raw_data,
            all_agent_xy=all_agent_reference_xy,
            all_agent_heading=all_agent_reference_heading,
            heading_index=6, rot_vel_index=[7, 8],
            center_agent_num=center_objects.shape[0],
        )
        # Only get (x, y, heading, v_x, v_y)
        all_agent_local_valid_mask = all_agent_local_obj_trajs[:, :, :, -1]
        all_agent_local_obj_trajs = all_agent_local_obj_trajs[:, :, :, [0, 1, 6, 7, 8]]
        all_agent_local_obj_trajs = all_agent_local_obj_trajs * all_agent_local_valid_mask.unsqueeze(-1)
        all_agent_local_obj_trajs, all_agent_local_valid_mask = all_agent_local_obj_trajs.numpy(), all_agent_local_valid_mask.numpy()

        # Assert valid mask
        assert np.all(obj_trajs_data[obj_trajs_mask == 0.] == 0.), "obj_trajs_data invalid mask is wrong"
        assert np.all(obj_trajs_future_state[obj_trajs_future_mask == 0.] == 0.), "obj_trajs_future_state invalid mask is wrong"
        assert np.all(obj_trajs_full[obj_trajs_full_valid_mask == 0.] == 0.), "obj_trajs_full valid mask is wrong"
        assert np.all(all_agent_local_obj_trajs[all_agent_local_valid_mask == 0.] == 0.), "all_agent_local_obj_trajs invalid index is not zero"

        # Generate the labels of track_objects for training
        # Use track_index_to_predict_new here since we already update it earlier
        center_obj_idxs = np.arange(len(track_index_to_predict_new))
        center_gt_trajs = obj_trajs_future_state[center_obj_idxs, track_index_to_predict_new]
        center_gt_trajs_mask = obj_trajs_future_mask[center_obj_idxs, track_index_to_predict_new]  # (num_center_objects, num_future_timestamps)
        center_gt_trajs[center_gt_trajs_mask == 0] = 0

        batch_size = all_agent_local_obj_trajs.shape[0]
        all_agent_reference_xy_heading = all_agent_reference_xy_heading.unsqueeze(0).expand(batch_size, -1, -1)
        assert torch.max(valid_timestep) <= 10, "reference states for all agent is invalid in first 11 timestep"


        # Construct surrounding objects index and trajectories
        # obj_relative_trajs_full, (num_center_objects, num_objects, 91, 8)， (x, y, heading, vx, vy, acc_x, acc_y, angular_speed)
        # obj_trajs_past （num_objects, 11, 10), [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        surrounding_obj_trajs_full, surrounding_obj_index, surrounding_obj_trajs_valid_mask = (
            self.find_interacting_agents(obj_relative_trajs_full=all_agent_local_obj_trajs,
                                         obj_relative_trajs_full_mask=all_agent_local_valid_mask,
                                         obj_trajs_past=obj_trajs_past,
                                         obj_trajs_future=obj_trajs_future,
                                         track_index_to_predict=track_index_to_predict_new,
                                         k=self.surrounding_k,
                                         distance_threshold=self.distance_threshold))
        # Initialize the resulting list
        surrounding_obj_id = []
        # Iterate through each list in surrounding_obj_index
        for indices in surrounding_obj_index:
            # Create a new list for the current set of indices, ignoring -1
            current_ids = [obj_ids[idx] for idx in indices if idx != -1]
            # Append the current list of ids to the result
            surrounding_obj_id.append(current_ids)

        # Generate the final valid position of each object
        obj_trajs_pos = obj_trajs_data[:, :, :, 0:3]
        num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
        obj_trajs_last_pos = np.zeros((num_center_objects, num_objects, 3), dtype=np.float32)
        for k in range(num_timestamps):
            cur_valid_mask = obj_trajs_mask[:, :, k] > 0  # (num_center_objects, num_objects)
            obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, :, k, :][cur_valid_mask]

        center_gt_final_valid_idx = np.zeros((num_center_objects), dtype=np.float32)
        for k in range(center_gt_trajs_mask.shape[1]):
            cur_valid_mask = center_gt_trajs_mask[:, k] > 0  # (num_center_objects)
            center_gt_final_valid_idx[cur_valid_mask] = k

        return (obj_trajs_data, obj_trajs_mask > 0, obj_trajs_pos, obj_trajs_last_pos,
            obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs, center_gt_trajs_mask, center_gt_final_valid_idx,
            track_index_to_predict_new, sdc_track_index_new, obj_types, obj_ids,
                surrounding_obj_trajs_full, surrounding_obj_index, surrounding_obj_id, surrounding_obj_trajs_valid_mask > 0,
                all_agent_reference_xy_heading)

    def find_interacting_agents(self, obj_relative_trajs_full, obj_relative_trajs_full_mask, obj_trajs_past, obj_trajs_future, track_index_to_predict, k, distance_threshold):
        """
            Identify potential interacting agents and construct the surrounding_trajs_full array based on the minimum distance
            between any points along the entire trajectory.
            The input trajectory are already filtered. So there's no invalid trajectory: at least one timestep is valid in each trajectory.

            Args:
            - obj_relative_trajs_full (numpy array): Shape (center_agent_num, all_agent_num, timestep, 2), relative x and y coordinates
            - obj_relative_trajs_full_mask (numpy array): Shape (center_agent_num, all_agent_num, timestep), 1 indicates valid, 0 indicates invalid
            - obj_traj_past (numpy array): Shape (all_agent_num, past_timestep, 10)
            - obj_traj_future (numpy array): Shape (all_agent_num, future_timestep, 10)
            - track_index_to_predict (list): List of indices of center agents
            - k (int): Number of top closest agents to select
            - distance_threshold (float): Distance threshold to filter agents


            Returns:
            - surrounding_obj_trajs_full (numpy array): Shape (center_agent_num, surrounding_agent_num, timestep, obj_relative_trajs_full.shape[-1])
            - surrounding_obj_indices (list): List of indices of the chosen surrounding agents
            - surrounding_obj_trajs_valid_mask (numpy array): Shape (center_agent_num, surrounding_agent_num, timestep)
            """
        # Convert numpy arrays to torch tensors
        obj_relative_trajs_full = torch.from_numpy(obj_relative_trajs_full).float()
        obj_relative_trajs_full_mask = torch.from_numpy(obj_relative_trajs_full_mask).float()

        # Construct full obj trajectory (not relative!)
        obj_trajs_past = torch.from_numpy(obj_trajs_past).float()
        obj_trajs_future = torch.from_numpy(obj_trajs_future).float()
        obj_traj = torch.cat((obj_trajs_past, obj_trajs_future), dim=1)
        obj_traj_mask = obj_traj[:, :, -1]

        all_agent_num, all_timestep, _ = obj_traj.shape
        _, future_timestep, _ = obj_trajs_future.shape

        # May only use future trajectory to find surrounding agents
        if self.surrounding_distance_metric == "only_future":
            traj_to_use = obj_trajs_future
            traj_to_use_mask = obj_traj_mask[:, 11:]
            timestep_to_use = future_timestep
        elif self.surrounding_distance_metric == "all":
            traj_to_use = obj_traj
            traj_to_use_mask = obj_traj_mask
            timestep_to_use = all_timestep

        assert len(track_index_to_predict) == obj_relative_trajs_full.shape[0], "center agent num is incorrect!"
        center_agent_num = len(track_index_to_predict)

        # Extract x and y coordinates
        x_y_coordinates = traj_to_use[:, :, :2]  # Shape: (all_agent_num, timestep, 2)
        # Select the center agent coordinates
        center_x_y_coordinates = x_y_coordinates[track_index_to_predict]  # Shape: (center_agent_num, timestep, 2)

        # Expand all agent, x y coordinates
        x_y_coordinates_expanded = x_y_coordinates.unsqueeze(0).expand(center_agent_num, -1, -1, -1) # Shape: (center_agent_num, all_agent_num, timestep, timestep, 2)
        # Expand center x y coordinates
        center_x_y_coordinates_expanded = center_x_y_coordinates.unsqueeze(1).expand(-1, all_agent_num, -1, -1)  # Shape: (center_agent_num, all_agent_num, timestep, timestep, 2)

        # Compute pairwise distance
        pairwise_distances = torch.cdist(center_x_y_coordinates_expanded, x_y_coordinates_expanded, p=2)  # Shape: (center_agent_num, all_agent_num, timestep, timestep)

        # Create the valid mask for the center agent and all other agents
        center_valid_mask = traj_to_use_mask[track_index_to_predict]  # Shape: (center_agent_num, timestep)
        center_valid_mask = center_valid_mask.unsqueeze(1).unsqueeze(-1).expand(-1, all_agent_num, -1,
                                                                                timestep_to_use)  # Shape: (center_agent_num, all_agent_num, timestep, timestep)
        other_valid_mask = traj_to_use_mask.unsqueeze(0).unsqueeze(-2).expand(center_agent_num, -1, timestep_to_use,
                                                                           -1)  # Shape: (center_agent_num, all_agent_num, timestep, timestep)

        # Combine the mask
        valid_mask = center_valid_mask * other_valid_mask  # Shape: (center_agent_num, all_agent_num, timestep, timestep)

        pairwise_distances = pairwise_distances.masked_fill(valid_mask == 0, float('inf'))

        distances, _ = pairwise_distances.min(dim=-1)
        distances, _ = distances.min(dim=-1)

        # Find out the actual number of surrounding agents to select
        num_surrounding_agents = min(k, all_agent_num)

        # Find the top-k closest agents for each center agent
        topk_distances, topk_indices = torch.topk(distances, num_surrounding_agents, dim=-1, largest=False)

        # Make sure tracks_to_predict is in the first position of topk_indices
        batch_size = topk_distances.shape[0]
        # Convert numpy array to tensor
        track_index_to_predict_tensor = torch.tensor(track_index_to_predict)

        # Iterate over each batch
        for i in range(batch_size):
            # Check if track_index_to_predict[i] is in topk_indices[i, :]
            if track_index_to_predict_tensor[i] in topk_indices[i, :]:
                # Find the position of track_index_to_predict[i]
                pos = (topk_indices[i, :] == track_index_to_predict_tensor[i]).nonzero(as_tuple=True)[0].item()
                # Swap the elements to place track_index_to_predict[i] at position 0
                # Swapping values using a temporary variable
                temp1 = topk_indices[i, 0].item()
                topk_indices[i, 0] = topk_indices[i, pos]
                topk_indices[i, pos] = temp1
                # Swap the corresponding distances
                temp2 = topk_distances[i, 0].item()
                topk_distances[i, 0] = topk_distances[i, pos]
                topk_distances[i, pos] = temp2
            else:
                # Remove the last element and insert track_index_to_predict[i] at position 0
                topk_indices[i, 1:] = topk_indices[i, :-1].clone()
                topk_indices[i, 0] = track_index_to_predict_tensor[i]
                # Remove the last distance and insert 0 at position 0
                topk_distances[i, 1:] = topk_distances[i, :-1].clone()
                topk_distances[i, 0] = 0.0

        # Filter out distance above the threshold
        threshold_valid_mask = distances <= distance_threshold
        # use gather, we retrieve the value from valid_topk_mask based on the topk_indices, along axis=1
        expanded_valid_topk_mask = threshold_valid_mask.gather(1, topk_indices)  # shape: (center_agent_num, k)

        # Step 2: Use the expanded mask to get filtered_topk_indices
        filtered_topk_indices = topk_indices.clone()
        filtered_topk_indices[~expanded_valid_topk_mask] = -1

        #  Create a mask for the valid top-k agents.
        #  Shape: (center_agent_num, surrounding_agent_num, timestep)
        #  First gather the information from obj_relative_trajs_full_mask
        surrounding_obj_trajs_valid_mask = obj_relative_trajs_full_mask.gather(
            1, topk_indices.unsqueeze(-1).expand(-1, -1, all_timestep)
        )

        # Then further pad the mask based on filtered_topk_indices
        # Step 1: Create a mask where filtered_topk_indices is -1
        invalid_filtered_topk_indices_mask = (filtered_topk_indices == -1)
        # Step 2: Expand the mask to match the shape of surrounding_obj_trajs_full_valid_mask
        # This involves adding an extra dimension for timestep
        expanded_invalid_indices_mask = invalid_filtered_topk_indices_mask.unsqueeze(2).expand(-1, -1, all_timestep)
        # Step 3: Use the expanded mask to set the corresponding entries to 0
        surrounding_obj_trajs_valid_mask[expanded_invalid_indices_mask] = 0.

        # Gather the trajectories of the top-k closest agents
        #  Shape: (center_agent_num, surrounding_agent_num, timestep, obj_trajs_full.shape[-1])
        surrounding_obj_trajs_full = torch.gather(
            obj_relative_trajs_full,
            1,
            topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, all_timestep, obj_relative_trajs_full.shape[-1])
        )

        # Pad the surrounding_obj_trajs_full with 0 based on surrounding_obj_trajs_valid_mask
        # Step 1: Expand the valid mask to match the shape of surrounding_obj_trajs_full
        expanded_surrounding_obj_trajs_valid_mask = surrounding_obj_trajs_valid_mask.unsqueeze(3).expand(-1, -1, -1, obj_relative_trajs_full.shape[-1])
        # Step 2: Use the expanded mask to set the corresponding entries in surrounding_obj_trajs_full to 0
        surrounding_obj_trajs_full = surrounding_obj_trajs_full * expanded_surrounding_obj_trajs_valid_mask

        # Make sure that the surrounding_obj_trajs_full.shape[1] = k to maintain size
        current_k = num_surrounding_agents
        if current_k < k:
            surrounding_obj_trajs_full_padded = torch.zeros((surrounding_obj_trajs_full.shape[0],
                                                             k,
                                                             surrounding_obj_trajs_full.shape[2],
                                                             surrounding_obj_trajs_full.shape[3]),
                                                            dtype=surrounding_obj_trajs_full.dtype)
            filtered_topk_indices_padded = -1. * torch.ones((filtered_topk_indices.shape[0], k),
                                                            dtype=filtered_topk_indices.dtype)
            surrounding_obj_trajs_valid_mask_padded = torch.zeros((surrounding_obj_trajs_valid_mask.shape[0],
                                                                   k,
                                                                   surrounding_obj_trajs_valid_mask.shape[2]),
                                                                  dtype=surrounding_obj_trajs_valid_mask.dtype)

            surrounding_obj_trajs_full_padded[:, :current_k, :, :] = surrounding_obj_trajs_full
            filtered_topk_indices_padded[:, :current_k] = filtered_topk_indices
            surrounding_obj_trajs_valid_mask_padded[:, :current_k, :] = surrounding_obj_trajs_valid_mask

            # Convert the results back to numpy arrays
            surrounding_obj_trajs_full = surrounding_obj_trajs_full_padded.numpy()
            surrounding_obj_indices = filtered_topk_indices_padded.numpy().astype(int).tolist()
            surrounding_obj_trajs_valid_mask = surrounding_obj_trajs_valid_mask_padded.numpy()
        else:
            # Convert the results back to numpy arrays
            surrounding_obj_trajs_full = surrounding_obj_trajs_full.numpy()
            surrounding_obj_indices = filtered_topk_indices.numpy().astype(int).tolist()
            surrounding_obj_trajs_valid_mask = surrounding_obj_trajs_valid_mask.numpy()

        return surrounding_obj_trajs_full, surrounding_obj_indices, surrounding_obj_trajs_valid_mask

    def get_interested_agents(self, track_index_to_predict, obj_trajs_full, current_time_index, obj_types, scene_id):
        center_objects_list = []
        track_index_to_predict_selected = []

        for k in range(len(track_index_to_predict)):
            obj_idx = track_index_to_predict[k]

            assert obj_trajs_full[obj_idx, current_time_index, -1] > 0, f'obj_idx={obj_idx}, scene_id={scene_id}'

            center_objects_list.append(obj_trajs_full[obj_idx, current_time_index])
            track_index_to_predict_selected.append(obj_idx)

        center_objects = np.stack(center_objects_list, axis=0)  # (num_center_objects, num_attrs)
        track_index_to_predict = np.array(track_index_to_predict_selected)
        return center_objects, track_index_to_predict

    @staticmethod
    def transform_trajs_to_center_coords(obj_trajs, center_xyz, center_heading, heading_index, rot_vel_index=None):
        """
        Args:
            obj_trajs (num_objects, num_timestamps, num_attrs):
                first three values of num_attrs are [x, y, z] or [x, y]
            center_xyz (num_center_objects, 3 or 2): [x, y, z] or [x, y]
            center_heading (num_center_objects):
            heading_index: the index of heading angle in the num_attr-axis of obj_trajs
        """
        num_objects, num_timestamps, num_attrs = obj_trajs.shape
        num_center_objects = center_xyz.shape[0]
        assert center_xyz.shape[0] == center_heading.shape[0]
        assert center_xyz.shape[1] in [3, 2]

        obj_trajs = obj_trajs.clone().view(1, num_objects, num_timestamps, num_attrs).repeat(num_center_objects, 1, 1, 1)
        obj_trajs[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]
        obj_trajs[:, :, :, 0:2] = common_utils.rotate_points_along_z(
            points=obj_trajs[:, :, :, 0:2].view(num_center_objects, -1, 2),
            angle=-center_heading
        ).view(num_center_objects, num_objects, num_timestamps, 2)

        obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]

        # Rotate direction of velocity
        if rot_vel_index is not None:
            assert len(rot_vel_index) == 2
            obj_trajs[:, :, :, rot_vel_index] = common_utils.rotate_points_along_z(
                points=obj_trajs[:, :, :, rot_vel_index].view(num_center_objects, -1, 2),
                angle=-center_heading
            ).view(num_center_objects, num_objects, num_timestamps, 2)

        return obj_trajs

    @staticmethod
    def transform_trajs_to_agent_local_coords(obj_trajs_raw_data, all_agent_xy, all_agent_heading, heading_index,
                                              rot_vel_index=None, center_agent_num=None):
        """
        Args:
            obj_trajs: (agent_num, timestep, num_attr),  if 10, maybe [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid].
            heading_index: the index of heading angle in the num_attr-axis of obj_trajs
        """

        num_objects, num_timestamps, num_attrs = obj_trajs_raw_data.shape

        assert all_agent_xy.shape[0] == all_agent_heading.shape[0]
        assert all_agent_xy.shape[1] in [3, 2]
        assert num_objects == all_agent_xy.shape[0], "center num is not equal to all agent num!"

        # Only takes x, y, v_x, v_y from the obj_trajs
        obj_trajs = obj_trajs_raw_data.clone()

        obj_trajs[:, :, :2] -= all_agent_xy[:, :2].unsqueeze(1).expand(-1, num_timestamps, -1)
        obj_trajs[:, :, :2] = common_utils.rotate_points_along_z(
            points=obj_trajs[:, :, :2].view(num_objects, -1, 2),
            angle=-all_agent_heading,
        )

        obj_trajs[:, :, heading_index] -= all_agent_heading.unsqueeze(-1)

        if rot_vel_index is not None:
            assert len(rot_vel_index) == 2
            obj_trajs[:, :, rot_vel_index] = common_utils.rotate_points_along_z(
                points=obj_trajs[:, :, rot_vel_index].view(num_objects, -1, 2),
                angle=-all_agent_heading
            )

        obj_trajs = obj_trajs.unsqueeze(0).expand(center_agent_num, -1, -1, -1)

        return obj_trajs

    @staticmethod
    def transform_trajs_to_global_coords(relative_obj_trajs, center_xyz, center_heading, heading_index,
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
            points=relative_obj_trajs[:, :, :, 0:2].view(num_center_objects, -1, 2),
            angle=center_heading
        ).view(num_center_objects, num_objects, num_timestamps, 2)

        # Translate coordinates back to the global frame
        relative_obj_trajs[:, :, :, 0:center_xyz.shape[1]] += center_xyz[:, None, None, :]

        # Adjust heading angles back to the global frame
        relative_obj_trajs[:, :, :, heading_index] += center_heading[:, None, None]

        # Rotate velocity components back to the global frame (if applicable)
        if rot_vel_index is not None:
            assert len(rot_vel_index) == 2
            relative_obj_trajs[:, :, :, rot_vel_index] = common_utils.rotate_points_along_z(
                points=relative_obj_trajs[:, :, :, rot_vel_index].view(num_center_objects, -1, 2),
                angle=center_heading
            ).view(num_center_objects, num_objects, num_timestamps, 2)

        return relative_obj_trajs

    def generate_centered_trajs_for_agents(self, center_objects, obj_trajs_past, obj_types, center_indices, sdc_index, timestamps, obj_trajs_future):
        """[summary]

        Args:
            center_objects (num_center_objects, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            obj_trajs_past (num_objects, 11, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            obj_types (num_objects):
            center_indices (num_center_objects): the index of center objects in obj_trajs_past
            centered_valid_time_indices (num_center_objects), the last valid time index of center objects
            timestamps ([type]): [description]
            obj_trajs_future (num_objects, 80, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        Returns:
            ret_obj_trajs (num_center_objects, num_objects, num_timestamps, num_attrs):
            ret_obj_valid_mask (num_center_objects, num_objects, num_timestamps):
            ret_obj_trajs_future (num_center_objects, num_objects, num_timestamps_future, 4):  [x, y, vx, vy]
            ret_obj_valid_mask_future (num_center_objects, num_objects, num_timestamps_future):
        """
        assert obj_trajs_past.shape[-1] == 10
        assert center_objects.shape[-1] == 10
        num_center_objects = center_objects.shape[0]
        num_objects, num_timestamps, box_dim = obj_trajs_past.shape
        # Transform to cpu torch tensor
        center_objects = torch.from_numpy(center_objects).float()
        obj_trajs_past = torch.from_numpy(obj_trajs_past).float()
        timestamps = torch.from_numpy(timestamps)

        # Construct obj_trajs_full
        obj_trajs_full_raw_data = torch.cat((obj_trajs_past, torch.from_numpy(obj_trajs_future).float()), dim=1)
        # Transform coordinates to the centered objects
        obj_trajs_full = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_full_raw_data,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )

        # Compute acceleration for obj_trajs_full
        obj_trajs_full_vel = obj_trajs_full[:, :, :, 7:9]  # (num_centered_objects, num_objects, num_timestamps, 2)
        obj_trajs_full_vel_pre = torch.roll(obj_trajs_full_vel, shifts=1, dims=2)
        obj_trajs_full_acce = (obj_trajs_full_vel - obj_trajs_full_vel_pre) / 0.1  # (num_centered_objects, num_objects, num_timestamps, 2)
        obj_trajs_full_acce[:, :, 0, :] = obj_trajs_full_acce[:, :, 1, :]

        # Compute angular speed for obj_trajs_full
        obj_trajs_full_heading = obj_trajs_full[:, :, :, 6].unsqueeze(-1)
        obj_trajs_full_heading_pre = torch.roll(obj_trajs_full_heading, shifts=1, dims=1)
        obj_trajs_full_angular_speed = (obj_trajs_full_heading - obj_trajs_full_heading_pre) / 0.1
        obj_trajs_full_angular_speed[:, :, 0, :] = obj_trajs_full_angular_speed[:, :, 1, :]

        # ret_obj_trajs_full contains (x, y, heading, vx, vy, acc_x, acc_y, angular_speed)
        ret_obj_trajs_full = torch.cat((
            obj_trajs_full[:, :, :, [0, 1, 6, 7, 8]],
            obj_trajs_full_acce,
            obj_trajs_full_angular_speed,
        ), dim=-1)

        ret_obj_trajs_full_valid_mask = obj_trajs_full[:, :, :, -1]  # (num_center_obejcts, num_objects, num_timestamps_future)  # CHECK THIS, 20220322
        ret_obj_trajs_full[ret_obj_trajs_full_valid_mask == 0] = 0

        # Transform coordinates to the centered objects
        obj_trajs = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_past,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )

        # Generate the attributes for each object
        object_onehot_mask = torch.zeros((num_center_objects, num_objects, num_timestamps, 5))
        object_onehot_mask[:, obj_types == 'TYPE_VEHICLE', :, 0] = 1
        object_onehot_mask[:, obj_types == 'TYPE_PEDESTRIAN', :, 1] = 1
        object_onehot_mask[:, obj_types == 'TYPE_CYCLIST', :, 2] = 1
        object_onehot_mask[torch.arange(num_center_objects), center_indices, :, 3] = 1
        object_onehot_mask[:, sdc_index, :, 4] = 1

        object_time_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, num_timestamps + 1))
        object_time_embedding[:, :, torch.arange(num_timestamps), torch.arange(num_timestamps)] = 1
        object_time_embedding[:, :, torch.arange(num_timestamps), -1] = timestamps

        object_heading_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, 2))
        object_heading_embedding[:, :, :, 0] = np.sin(obj_trajs[:, :, :, 6])
        object_heading_embedding[:, :, :, 1] = np.cos(obj_trajs[:, :, :, 6])

        vel = obj_trajs[:, :, :, 7:9]  # (num_centered_objects, num_objects, num_timestamps, 2)
        vel_pre = torch.roll(vel, shifts=1, dims=2)
        acce = (vel - vel_pre) / 0.1  # (num_centered_objects, num_objects, num_timestamps, 2)
        acce[:, :, 0, :] = acce[:, :, 1, :]

        ret_obj_trajs = torch.cat((
            obj_trajs[:, :, :, 0:6], 
            object_onehot_mask,
            object_time_embedding, 
            object_heading_embedding,
            obj_trajs[:, :, :, 7:9], 
            acce,
        ), dim=-1)

        ret_obj_valid_mask = obj_trajs[:, :, :, -1]  # (num_center_obejcts, num_objects, num_timestamps)
        ret_obj_trajs[ret_obj_valid_mask == 0] = 0

        # Generate label for future trajectories
        obj_trajs_future = torch.from_numpy(obj_trajs_future).float()
        obj_trajs_future = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_future,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )

        # Compute acceleration for obj future traj
        future_vel = obj_trajs_future[:, :, :, 7:9]  # (num_centered_objects, num_objects, num_timestamps, 2)
        future_vel_pre = torch.roll(future_vel, shifts=1, dims=2)
        future_acce = (future_vel - future_vel_pre) / 0.1  # (num_centered_objects, num_objects, num_timestamps, 2)
        future_acce[:, :, 0, :] = (obj_trajs_future[:, :, 0, 7:9] - obj_trajs[:, :, -1, 7:9]) / 0.1 # First step acceleration needs to use obj_traj history data

        # Compute angular speed for obj future traj
        future_heading = obj_trajs_future[:, :, :, 6].unsqueeze(-1)
        future_heading_pre = torch.roll(future_heading, shifts=1, dims=1)
        future_angular_speed = (future_heading - future_heading_pre) / 0.1
        future_angular_speed[:, :, 0, :] = (obj_trajs_future[:, :, 0, 6].unsqueeze(-1) - obj_trajs[:, :, -1, 6].unsqueeze(-1)) / 0.1

        # ret_obj_trajs_future contains not only (x, y, vx, vy), but also the newly computed (acc_x, acc_y, angular_speed)
        ret_obj_trajs_future = obj_trajs_future[:, :, :, [0, 1, 7, 8]]  # (x, y, vx, vy)
        ret_obj_trajs_future = torch.cat((ret_obj_trajs_future, future_acce, future_angular_speed), dim=-1)

        ret_obj_valid_mask_future = obj_trajs_future[:, :, :, -1]  # (num_center_obejcts, num_objects, num_timestamps_future)  # CHECK THIS, 20220322
        ret_obj_trajs_future[ret_obj_valid_mask_future == 0] = 0

        return (ret_obj_trajs.numpy(), ret_obj_valid_mask.numpy(),
                ret_obj_trajs_future.numpy(), ret_obj_valid_mask_future.numpy(),
                ret_obj_trajs_full.numpy(), ret_obj_trajs_full_valid_mask.numpy())

    @staticmethod
    def get_all_agent_reference_states(data, type):

        valid_mask = data[:, :, -1] == 1
        valid_mask = valid_mask.float()
        if type == 'first_valid':
            valid_timestep = valid_mask.argmax(dim=1)
            valid_timestep = valid_timestep.to(torch.int64)
        elif type == 'last_valid':
            # Safety check: ensure valid_mask has correct shape and at least one valid timestep
            # in first 11 timesteps for each item
            batch_size, num_timesteps = valid_mask.shape

            # Check if each item has at least one valid timestep in first 11 steps
            first_11_valid = valid_mask[:, :11].sum(dim=1) > 0
            if not torch.all(first_11_valid):
                raise ValueError("Some items don't have any valid timesteps in first 11 steps")

            # Get the indices of valid timesteps (where value is 1)
            # Only consider first 11 timesteps
            valid_indices = torch.arange(11, device=valid_mask.device).expand(batch_size, 11)
            masked_indices = valid_indices * valid_mask[:, :11]

            # Get the last valid timestep (maximum index where value is 1)
            valid_timestep = torch.max(masked_indices, dim=1)[0].int()
        # Extract all features at the first valid timestep
        batch_size = data.shape[0]
        reference_states = data[torch.arange(batch_size), valid_timestep]

        return reference_states, valid_timestep

    @staticmethod
    def generate_batch_polylines_from_map(polylines, point_sampled_interval=1, vector_break_dist_thresh=1.0, num_points_each_polyline=20):
        """
        Args:
            polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]

        Returns:
            ret_polylines: (num_polylines, num_points_each_polyline, 7)
            ret_polylines_mask: (num_polylines, num_points_each_polyline)
        """
        point_dim = polylines.shape[-1]

        sampled_points = polylines[::point_sampled_interval]
        sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
        buffer_points = np.concatenate((sampled_points[:, 0:2], sampled_points_shift[:, 0:2]), axis=-1) # [ed_x, ed_y, st_x, st_y]
        buffer_points[0, 2:4] = buffer_points[0, 0:2]

        break_idxs = (np.linalg.norm(buffer_points[:, 0:2] - buffer_points[:, 2:4], axis=-1) > vector_break_dist_thresh).nonzero()[0]
        polyline_list = np.array_split(sampled_points, break_idxs, axis=0)
        ret_polylines = []
        ret_polylines_mask = []

        def append_single_polyline(new_polyline):
            cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
            cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
            cur_polyline[:len(new_polyline)] = new_polyline
            cur_valid_mask[:len(new_polyline)] = 1
            ret_polylines.append(cur_polyline)
            ret_polylines_mask.append(cur_valid_mask)

        for k in range(len(polyline_list)):
            if polyline_list[k].__len__() <= 0:
                continue
            for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
                append_single_polyline(polyline_list[k][idx: idx + num_points_each_polyline])

        ret_polylines = np.stack(ret_polylines, axis=0)
        ret_polylines_mask = np.stack(ret_polylines_mask, axis=0)

        ret_polylines = torch.from_numpy(ret_polylines)
        ret_polylines_mask = torch.from_numpy(ret_polylines_mask)

        return ret_polylines, ret_polylines_mask

    def create_map_data_for_center_objects(self, center_objects, map_infos, center_offset):
        """
        Args:
            center_objects (num_center_objects, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            map_infos (dict):
                all_polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]
            center_offset (2):, [offset_x, offset_y]
        Returns:
            map_polylines (num_center_objects, num_topk_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
            map_polylines_mask (num_center_objects, num_topk_polylines, num_points_each_polyline)
        """
        num_center_objects = center_objects.shape[0]

        # Transform object coordinates by center objects
        def transform_to_center_coordinates(neighboring_polylines, neighboring_polyline_valid_mask):
            neighboring_polylines[:, :, :, 0:3] -= center_objects[:, None, None, 0:3]
            neighboring_polylines[:, :, :, 0:2] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, :, 0:2].view(num_center_objects, -1, 2),
                angle=-center_objects[:, 6]
            ).view(num_center_objects, -1, batch_polylines.shape[1], 2)
            neighboring_polylines[:, :, :, 3:5] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, :, 3:5].view(num_center_objects, -1, 2),
                angle=-center_objects[:, 6]
            ).view(num_center_objects, -1, batch_polylines.shape[1], 2)

            # Use pre points to map
            # (num_center_objects, num_polylines, num_points_each_polyline, num_feat)
            xy_pos_pre = neighboring_polylines[:, :, :, 0:2]
            xy_pos_pre = torch.roll(xy_pos_pre, shifts=1, dims=-2)
            xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]
            neighboring_polylines = torch.cat((neighboring_polylines, xy_pos_pre), dim=-1)

            neighboring_polylines[neighboring_polyline_valid_mask == 0] = 0
            return neighboring_polylines, neighboring_polyline_valid_mask

        polylines = torch.from_numpy(map_infos['all_polylines'].copy())
        center_objects = torch.from_numpy(center_objects)

        batch_polylines, batch_polylines_mask = self.generate_batch_polylines_from_map(
            polylines=polylines.numpy(), point_sampled_interval=self.dataset_cfg.get('POINT_SAMPLED_INTERVAL', 1),
            vector_break_dist_thresh=self.dataset_cfg.get('VECTOR_BREAK_DIST_THRESH', 1.0),
            num_points_each_polyline=self.dataset_cfg.get('NUM_POINTS_EACH_POLYLINE', 20),
        )  # (num_polylines, num_points_each_polyline, 7), (num_polylines, num_points_each_polyline)

        # Collect a number of closest polylines for each center objects
        num_of_src_polylines = self.dataset_cfg.NUM_OF_SRC_POLYLINES

        if len(batch_polylines) > num_of_src_polylines:
            polyline_center = batch_polylines[:, :, 0:2].sum(dim=1) / torch.clamp_min(batch_polylines_mask.sum(dim=1).float()[:, None], min=1.0)
            center_offset_rot = torch.from_numpy(np.array(center_offset, dtype=np.float32))[None, :].repeat(num_center_objects, 1)
            center_offset_rot = common_utils.rotate_points_along_z(
                points=center_offset_rot.view(num_center_objects, 1, 2),
                angle=center_objects[:, 6]
            ).view(num_center_objects, 2)

            pos_of_map_centers = center_objects[:, 0:2] + center_offset_rot

            dist = (pos_of_map_centers[:, None, :] - polyline_center[None, :, :]).norm(dim=-1)  # (num_center_objects, num_polylines)
            topk_dist, topk_idxs = dist.topk(k=num_of_src_polylines, dim=-1, largest=False)
            map_polylines = batch_polylines[topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline, 7)
            map_polylines_mask = batch_polylines_mask[topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline)
        else:
            map_polylines = batch_polylines[None, :, :, :].repeat(num_center_objects, 1, 1, 1)
            map_polylines_mask = batch_polylines_mask[None, :, :].repeat(num_center_objects, 1, 1)

        map_polylines, map_polylines_mask = transform_to_center_coordinates(
            neighboring_polylines=map_polylines,
            neighboring_polyline_valid_mask=map_polylines_mask
        )

        temp_sum = (map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].float()).sum(dim=-2)  # (num_center_objects, num_polylines, 3)
        map_polylines_center = temp_sum / torch.clamp_min(map_polylines_mask.sum(dim=-1).float()[:, :, None], min=1.0)  # (num_center_objects, num_polylines, 3)

        map_polylines = map_polylines.numpy()
        map_polylines_mask = map_polylines_mask.numpy()
        map_polylines_center = map_polylines_center.numpy()

        return map_polylines, map_polylines_mask, map_polylines_center

    def generate_prediction_dicts(self, batch_dict, output_path=None):
        """

        Args:
            batch_dict:
                pred_scores: (num_center_objects, num_modes)
                pred_trajs: (num_center_objects, num_modes, num_timestamps, 7)

              input_dict:
                center_objects_world: (num_center_objects, 10)
                center_objects_type: (num_center_objects)
                center_objects_id: (num_center_objects)
                center_gt_trajs_src: (num_center_objects, num_timestamps, 10)
        """
        input_dict = batch_dict['input_dict']

        pred_scores = batch_dict['pred_scores']
        pred_trajs = batch_dict['pred_trajs']
        center_objects_world = input_dict['center_objects_world'].type_as(pred_trajs)

        num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape
        assert num_feat == 7

        pred_trajs_world = common_utils.rotate_points_along_z(
            points=pred_trajs.view(num_center_objects, num_modes * num_timestamps, num_feat),
            angle=center_objects_world[:, 6].view(num_center_objects)
        ).view(num_center_objects, num_modes, num_timestamps, num_feat)
        pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2]

        pred_dict_list = []
        batch_sample_count = batch_dict['batch_sample_count']
        start_obj_idx = 0
        for bs_idx in range(batch_dict['batch_size']):
            cur_scene_pred_list = []
            for obj_idx in range(start_obj_idx, start_obj_idx + batch_sample_count[bs_idx]):
                single_pred_dict = {
                    'scenario_id': input_dict['scenario_id'][obj_idx],
                    'pred_trajs': pred_trajs_world[obj_idx, :, :, 0:2].cpu().numpy(),
                    'pred_scores': pred_scores[obj_idx, :].cpu().numpy(),
                    'object_id': input_dict['center_objects_id'][obj_idx],
                    'object_type': input_dict['center_objects_type'][obj_idx],
                    'gt_trajs': input_dict['center_gt_trajs_src'][obj_idx].cpu().numpy(),
                    'track_index_to_predict': input_dict['track_index_to_predict'][obj_idx].cpu().numpy()
                }
                cur_scene_pred_list.append(single_pred_dict)

            pred_dict_list.append(cur_scene_pred_list)
            start_obj_idx += batch_sample_count[bs_idx]

        assert start_obj_idx == num_center_objects
        assert len(pred_dict_list) == batch_dict['batch_size']

        return pred_dict_list

