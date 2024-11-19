import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
from metadrive_util.trajectory import WaypointTrajectory
import IPython
e = IPython.embed


class PreferenceData:
    def __init__(self):
        self.num_queries = 9  # same as chunksize-1
        self.len = 0

        # include now timestep
        self.ego_pos = np.zeros((self.num_queries + 1, 2))
        self.nearest_other_pos = np.zeros((self.num_queries + 1, 2))
        self.nearest_distance = np.zeros((self.num_queries + 1, 1))
        self.has_nearest = False

        self.ego_vel_kmh = np.zeros((self.num_queries + 1, 1))
        self.ego_lane_diff_angle = np.zeros((self.num_queries + 1, 1))
        self.ego_control = np.zeros((self.num_queries + 1, 2))

    def parse(self, root, start_time, episode_len):
        self.len = min(episode_len - start_time, self.num_queries + 1)
        ego_states = root['/observations/ego_state'][start_time:start_time +
                                                     self.num_queries + 1]  # (num_queries+1, 6)
        self._parse_ego_states(ego_states)
        history = root['/observations/history_info'][
            start_time:start_time + self.num_queries + 1]  # (num_queries+1, 5, 10, 4)
        self._parse_ego_other_pos(history[-1][:, -self.len:])

    def _parse_ego_states(self, ego_states):
        ego_lane_diff_lateral_cos = np.clip(
            (ego_states[:, :1] - 0.5) * 2, -1, 1)
        self.ego_lane_diff_angle[:self.len] = np.arccos(
            ego_lane_diff_lateral_cos) - np.pi / 2  # turn left +

        self.ego_vel_kmh[:self.len] = ego_states[:, 1:2] * (80 + 1) - 1
        self.ego_control[:self.len] = ego_states[:, 3:5] * \
            2 - 1  # steering and throttle

    def _parse_ego_other_pos(self, history):
        #  只选最近的一个距离，应该不需要转化坐标？
        ego_pos_array = history[0]  # (num_queries+1, 2)
        ego_pose = ego_pos_array[0]
        position_now = ego_pose[:2]
        heading_now = np.arctan2(ego_pose[3], ego_pose[2])
        self.ego_pos[:self.len] = self._transform_past_traj_to_future(
            ego_pos_array[:, :2], position_now, heading_now)

        for i in range(1, 5):
            other_pose = history[i]  # (num_queries+1, 2)
            if np.all(other_pose == 0):
                break
            other_pos_transform = self._transform_past_traj_to_future(
                other_pose[:, :2], position_now, heading_now)
            if other_pose[0][2] < 0.5 or (other_pos_transform[0][0] < 0 and other_pose[-1][0] < 0):
                continue  # 对向车道，或是它车一直在本车后方，轨迹无效
            else:
                self.nearest_other_pos[:self.len] = other_pos_transform
                self.nearest_distance = np.linalg.norm(
                    self.ego_pos - self.nearest_other_pos, axis=1).reshape(-1, 1)
                self.has_nearest = True
                break

    def to_dict(self):
        return {
            'len': self.len,
            'has_nearest': self.has_nearest,
            'ego_pos': self.ego_pos,
            'nearest_other_pos': self.nearest_other_pos,
            'nearest_distance': self.nearest_distance,
            'ego_vel_kmh': self.ego_vel_kmh,
            'ego_lane_diff_angle': self.ego_lane_diff_angle,
            'ego_control': self.ego_control,
        }

    @staticmethod
    def _transform_past_traj_to_future(traj_array, position_now, heading_now):
        traj = WaypointTrajectory()
        local_traj = traj.global_waypoint_to_local(
            traj_array, position_now, heading_now)
        local_traj = local_traj[1:]
        return local_traj


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = True
        self.is_train = None

    def __len__(self):
        return len(self.episode_ids)

    def get_raw_traj(self, index):
        sample_full_episode = False  # hardcode
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(
            self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            self.is_sim = is_sim
            episode_len = root['/action'].shape[0]

            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len - 5)  # TODO: hardcode
            # get observation at start_ts only
            vec_data = {'lidar_scan': root['/observations/lidar_scan'][start_ts],
                        'side_detector': root['/observations/side_detector'][start_ts],
                        'lane_detector': root['/observations/lane_detector'][start_ts],
                        'navi_info': root['/observations/navi_info'][start_ts],
                        'ego_state': root['/observations/ego_state'][start_ts],
                        'history_info': root['/observations/history_info'][start_ts].reshape(5, 40)
                        }

            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts

            traj_action = root['/action'][start_ts:]
            action_len = episode_len - start_ts

            heading_now = root['/heading'][start_ts].squeeze()
            position_now = root['/position_now'][start_ts]
            preference_data = PreferenceData()
            preference_data.parse(root, start_ts, episode_len)
            steer_throttle = preference_data.ego_control[1:2].astype(np.float32)
            action_aug = {'steer_throttle': steer_throttle,
                          'traj_action': traj_action}
            action_len = action_len + 1
        return vec_data, image_dict, action_aug, action_len, position_now, heading_now, preference_data.to_dict()

    def transform_traj(self, position_array, position_now, heading_now):
        """
        transfrom traj from global to local, and diff to action, and sample
        """
        traj = WaypointTrajectory()
        local_waypoint = traj.global_waypoint_to_local(
            position_array, position_now, heading_now)
        action_array = local_waypoint[1:]
        # action_array = np.diff(local_waypoint, axis=0)
        # action_array[:,0]=np.clip(action_array[:,0],0.0,np.inf)
        return action_array

    def __getitem__(self, index):
        original_traj_shape = (400, 2)  # must same shape
        (vec_data, image_dict, actions_aug, action_aug_len,
         position_now, heading_now, preference_data) = self.get_raw_traj(index=index)
        steer_throttle, positions = actions_aug['steer_throttle'], actions_aug['traj_action']

        local_traj = self.transform_traj(positions, position_now, heading_now)

        padded_traj = np.zeros(original_traj_shape, dtype=np.float32)
        padded_traj[:action_aug_len-1] = local_traj
        is_pad = np.zeros(original_traj_shape[0])
        is_pad[action_aug_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        for key in vec_data:
            vec_data[key] = torch.from_numpy(vec_data[key]).float()
        traj_data = torch.from_numpy(padded_traj).float()
        is_pad = torch.from_numpy(is_pad).bool()
        action_data = {'steer_throttle': steer_throttle,
                       'traj_action': traj_data}

        vec_data, image_data, action_data = self.normalize_data(
            vec_data, image_data, action_data)

        return image_data, vec_data, action_data, is_pad, preference_data
    
    def normalize_data(self, vec_data, image_data, action_data):
        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data['traj_action'] = (
            action_data['traj_action'][:10] - self.norm_stats["traj_mean"]) / self.norm_stats["traj_std"]
        action_data['steer_throttle'] = (
            action_data['steer_throttle'] - self.norm_stats["steer_mean"]) / self.norm_stats["steer_std"]
        for key in vec_data:
            vec_data[key] = (vec_data[key] - self.norm_stats["vec_mean"]
                             [key]) / self.norm_stats["vec_std"][key]
        return vec_data, image_data, action_data
    
    def set_action_norm_stats(self):
        norm_sample = 100
        chunk_size = 10
        all_traj_data = []
        all_steer_data = []
        for i in range(norm_sample):
            choose_index = np.random.choice(len(self))
            action_len = 0
            while action_len < chunk_size + 1:
                _, _, actions_aug, action_len, position_now, heading_now, _ = self.get_raw_traj(
                    index=choose_index)
            steer_throttle, positions = actions_aug['steer_throttle'], actions_aug['traj_action']
            # positions, _ = self.positions_sample(positions)
            local_position = self.transform_traj(
                positions, position_now, heading_now)
            local_position = local_position[:chunk_size]
            all_traj_data.append(local_position)
            all_steer_data.append(steer_throttle)

        all_traj_data = np.stack(all_traj_data)
        all_steer_data = np.stack(all_steer_data)

        # normalize action data
        traj_mean = all_traj_data.mean(axis=0)
        steer_mean = all_steer_data.mean(axis=0)
        traj_std = all_traj_data.std(axis=0)
        steer_std = all_steer_data.std(axis=0)
        traj_std = np.clip(traj_std, 1e-2, np.inf)
        steer_std = np.clip(steer_std, 1e-2, np.inf)
        self.norm_stats["traj_mean"] = traj_mean
        self.norm_stats["traj_std"] = traj_std
        self.norm_stats["steer_mean"] = steer_mean
        self.norm_stats["steer_std"] = steer_std
        
        # self.norm_stats["traj_mean"] = np.zeros_like(traj_mean)
        # self.norm_stats["traj_std"] = np.ones_like(traj_std)
        # self.norm_stats["steer_mean"] = np.zeros_like(steer_mean)
        # self.norm_stats["steer_std"] = np.ones_like(steer_std)
        # self.norm_stats["action_mean"] = np.array([0.0, 0.0], dtype=np.float32)
        # self.norm_stats["action_std"] = np.array([1.0, 1.0], dtype=np.float32)


def get_vec_norm_stats(dataset_dir, num_episodes):
    all_vec_data = {"lidar_scan": [], "side_detector": [], "lane_detector": [], "navi_info": [], "ego_state": [],
                    "history_info": []}
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            for key in all_vec_data:
                if key == "history_info":
                    all_vec_data[key].append(torch.from_numpy(
                        root[f'/observations/{key}'][()].reshape(-1, 5, 40)))
                else:
                    all_vec_data[key].append(torch.from_numpy(
                        root[f'/observations/{key}'][()]))

    for key in all_vec_data:
        all_vec_data[key] = torch.cat(all_vec_data[key], dim=0)

    # normalize qpos data
    vec_mean = dict()
    vec_std = dict()
    for key in all_vec_data:
        vec_mean[key] = all_vec_data[key].mean(dim=[0], keepdim=True)
        vec_mean[key] = vec_mean[key].numpy().squeeze()

        vec_std[key] = all_vec_data[key].std(dim=[0], keepdim=True)
        vec_std[key] = torch.clip(vec_std[key], 1e-2, np.inf)
        vec_std[key] = vec_std[key].numpy().squeeze()

    stats = {"vec_mean": vec_mean, "vec_std": vec_std,
             "example_vec": all_vec_data}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_vec_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(
        train_indices, dataset_dir, camera_names, norm_stats)
    train_dataset.set_action_norm_stats()
    norm_stats = train_dataset.norm_stats

    val_dataset = EpisodicDataset(
        val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=0)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=0)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


# helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
