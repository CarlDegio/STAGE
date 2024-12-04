import h5py
import numpy as np
import torch
import os
import pickle
from imitate_episodes import make_policy, forward_pass
from utils import EpisodicDataset
import argparse
import matplotlib.pyplot as plt


def pre_process(vec_data, stats):
    for key in vec_data:
        vec_data[key] = (vec_data[key] - stats['vec_mean']
                         [key]) / stats['vec_std'][key]
    return vec_data

def run_train():
    policy_class='ACT'
    policy_config={'lr': 5e-05, 'num_queries': 10, 'kl_weight': 100.0, 'hidden_dim': 512, 'dim_feedforward': 3200, 'lr_backbone': 1e-05, 'backbone': 'resnet18', 'enc_layers': 4, 'dec_layers': 1, 'nheads': 8, 'camera_names': ['top_down_view']}
    ckpt_dir = './ckpt_kl100_traj0.5_prefer10'
    
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
        
    dataset = EpisodicDataset([0], 'temp_traj', ['top_down_view'], stats)
    traj_length = dataset.get_raw_traj_length(0)
    print(f'traj length = {traj_length}')
    
    image_data_list, vec_data_list, action_data_list, is_pad_list, preference_data_list = [],{'lidar_scan':[], 'side_detector':[], 'lane_detector':[], 'navi_info':[], 'ego_state':[], 'history_info':[]},{'steer_throttle':[], 'traj_action':[]},[],{'len':[], 'has_nearest':[], 'ego_pos':[], 'nearest_other_pos':[], 'nearest_distance':[], 'ego_vel_kmh':[], 'ego_lane_diff_angle':[], 'ego_control':[]}
    
    for i in range(traj_length-5):
        original_traj_shape = (400, 2)  # must same shape
        (vec_data, image_dict, actions_aug, action_aug_len,
         position_now, heading_now, preference_data) = dataset.get_raw_traj_by_ts(0, i)
        steer_throttle, positions = actions_aug['steer_throttle'], actions_aug['traj_action']

        local_traj = dataset.transform_traj(positions, position_now, heading_now)

        padded_traj = np.zeros(original_traj_shape, dtype=np.float32)
        padded_traj[:action_aug_len-1] = local_traj
        is_pad = np.zeros(original_traj_shape[0])
        is_pad[action_aug_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in dataset.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        for key in vec_data:
            vec_data[key] = torch.from_numpy(vec_data[key]).float()
        traj_data = torch.from_numpy(padded_traj).float()
        is_pad = torch.from_numpy(is_pad).bool()
        action_data = {'steer_throttle': torch.from_numpy(steer_throttle).float(),
                       'traj_action': traj_data}

        vec_data, image_data, action_data = dataset.normalize_data(
            vec_data, image_data, action_data)
        
        image_data_list.append(image_data)
        for key in vec_data:
            vec_data_list[key].append(vec_data[key])
        for key in action_data:
            action_data_list[key].append(action_data[key])
        is_pad_list.append(is_pad)
        for key in preference_data:
            preference_data_list[key].append(torch.tensor(preference_data[key]))
    
    image_data = torch.stack(image_data_list)
    vec_data = {}
    for key in vec_data_list:
        vec_data[key] = torch.stack(vec_data_list[key])
    action_data = {}
    for key in action_data_list:
        action_data[key] = torch.stack(action_data_list[key])
    is_pad = torch.stack(is_pad_list)
    preference_data = {}
    for key in preference_data_list:
        preference_data[key] = torch.stack(preference_data_list[key])
    
    
    policy = make_policy(policy_class, policy_config)
    policy.load_state_dict(torch.load(os.path.join(ckpt_dir, f'policy_best.ckpt')))
    print(f'!Loaded: {os.path.join(ckpt_dir, "policy_best.ckpt")}')
    policy.cuda()
    policy.eval()
    with torch.inference_mode():
        _, style_value = forward_pass((image_data, vec_data, action_data, is_pad, preference_data), policy)
    style_value = style_value.cpu().numpy()
    print(style_value)
    
    plt.plot(style_value)
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store',
                        type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store',
                        type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store',
                        type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store',
                        type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int,
                        help='seed', required=True)
    parser.add_argument('--num_epochs', action='store',
                        type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float,
                        help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store',
                        type=float, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store',
                        type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store',
                        type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store',
                        type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    run_train()