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


def calc_data_freq(value_array):
    """计算并绘制信号的频谱
    Args:
        value_array: shape (n,1) 的 ndarray，表示时域信号
    """
    # 准备数据
    signal = value_array.flatten()  # 将(n,1)转换为(n,)
    n = len(signal)
    sample_freq = 10  # 采样频率为10Hz (0.1s间隔)

    # 计算FFT
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, d=1/sample_freq)  # 频率轴

    # 计算幅值谱
    magnitude = np.abs(fft_result)
    # 只取正频率部分
    positive_freqs = freqs[:n//2]
    positive_magnitude = magnitude[:n//2]

    return positive_freqs, positive_magnitude


def run_data():
    policy_class = 'ACT'
    policy_config = {'lr': 5e-05, 'num_queries': 10, 'kl_weight': 100.0, 'hidden_dim': 512, 'dim_feedforward': 3200,
                     'lr_backbone': 1e-05, 'backbone': 'resnet18', 'enc_layers': 4, 'dec_layers': 1, 'nheads': 8, 'camera_names': ['top_down_view'], 'style_pattern': 'stage'}
    stage_ckpt_dir = './ckpt_kl10_prefer10_new4rules'
    classifer_ckpt_dir = './ckpt_kl10_classifer_style'

    stats_path = os.path.join(stage_ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    file_list=['5011_d0_2_stage', '5011_d0_2_human', '5011_d0_0_stage', '5011_d0_0_human', '5011_d0_3_stage', '5011_d0_3_human', '5011_d0_4_stage', '5011_d0_4_human']
    # file_list=['5023_d0.1_0_discrete_style']
    dataset = EpisodicDataset(file_list, 'temp_traj', ['top_down_view'], stats)
    policy = make_policy(policy_class, policy_config)
    policy.load_state_dict(torch.load(os.path.join(
        stage_ckpt_dir, f'policy_last.ckpt')))
    print(f'!Loaded: {os.path.join(stage_ckpt_dir, "policy_last.ckpt")}')
    policy.cuda()
    policy.eval()
    
    policy_config['style_pattern'] = 'classifier'
    policy_class_style = make_policy(policy_class, policy_config)
    policy_class_style.load_state_dict(torch.load(os.path.join(
        classifer_ckpt_dir, f'policy_last.ckpt')))
    print(f'!Loaded: {os.path.join(classifer_ckpt_dir, "policy_last.ckpt")}')
    policy_class_style.cuda()
    policy_class_style.eval()
    
    style_value_list = []
    for traj_index in range(0, 8):
        traj_length = dataset.get_raw_traj_length(traj_index)
        print(f'traj length = {traj_length}')

        image_data_list, vec_data_list, action_data_list, is_pad_list, preference_data_list = [], {'lidar_scan': [], 'side_detector': [], 'lane_detector': [], 'navi_info': [], 'ego_state': [], 'history_info': []}, {'steer_throttle': [
        ], 'traj_action': []}, [], {'start_ts': [], 'len': [], 'has_nearest': [], 'ego_pos': [], 'nearest_other_pos': [], 'nearest_distance': [], 'ego_vel_kmh': [], 'ego_lane_diff_angle': [], 'ego_control': []}

        for i in range(traj_length-5):
            original_traj_shape = (400, 2)  # must same shape
            (vec_data, image_dict, actions_aug, action_aug_len,
             position_now, heading_now, preference_data) = dataset.get_raw_traj_by_ts(traj_index, i)
            steer_throttle, positions = actions_aug['steer_throttle'], actions_aug['traj_action']

            local_traj = dataset.transform_traj(
                positions, position_now, heading_now)

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
                preference_data_list[key].append(
                    torch.tensor(preference_data[key]))

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

        with torch.inference_mode():
            _, style_value = forward_pass(
                (image_data, vec_data, action_data, is_pad, preference_data), policy)
        style_value = style_value.cpu().numpy()
        # np.save(f'./temp_traj/style_value_{file_list[traj_index]}.npy', style_value)
        style_value_list.append(style_value/10)  # for style_value norm
        print(style_value)
        
        if traj_index % 2 == 0:
            with torch.inference_mode():
                _, style_value_classifer = forward_pass(
                    (image_data, vec_data, action_data, is_pad, preference_data), policy_class_style)
                style_value_classifer = torch.nn.Softmax(dim=1)(style_value_classifer)
                style_value_classifer = torch.argmax(style_value_classifer, dim=1)-1
                style_value_classifer = style_value_classifer.cpu().numpy()
            style_value_list.append(style_value_classifer)

    # 创建两个子图
    plt.figure(0, figsize=(9, 14))
    
    # 设置全局字体大小
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'legend.fontsize': 14,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16
    })
    
    # 创建共享x轴的子图，2:1的高度比
    gs = plt.GridSpec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.05)
    
    
    for i in range(0, len(style_value_list), 3):
        ax1 = plt.subplot(gs[i//3])
        min_len = min([len(style_value_list[i]), len(style_value_list[i+1]), len(style_value_list[i+2])])
        stage_data = style_value_list[i][:min_len]
        stage_data_classifer = style_value_list[i+1][:min_len]
        human_data = style_value_list[i+2][:min_len]
        
        correlation_matrix = np.corrcoef(stage_data.squeeze(), human_data.squeeze())
        r_square_stage = correlation_matrix[0,1]**2
        print(f'R2_stage = {r_square_stage}')
        
        correlation_matrix = np.corrcoef(stage_data_classifer.squeeze(), human_data.squeeze())
        r_square_classifer = correlation_matrix[0,1]**2
        print(f'R2_classifier = {r_square_classifer}')
        
        ax1.plot([0, 28], [-1, -1], 'k--', label='Style Value Boarder')
        ax1.plot([0, 28], [1, 1], 'k--')
        time_axis = np.arange(len(style_value_list[i])) * 0.1
        ax1.plot(time_axis, style_value_list[i], label='STAGE (ours)', color='blue')
        time_axis = np.arange(len(style_value_list[i+1])) * 0.1
        ax1.plot(time_axis, style_value_list[i+1], '--', label='CVAE+Discrete Style', color='#ff7f0e')
        time_axis = np.arange(len(style_value_list[i+2])) * 0.1
        ax1.plot(time_axis, style_value_list[i+2], '--', label='HUMAN', color='green')
        ax1.text(23, 0.85, f'R² = {r_square_stage:.3f}', fontsize=18,
                verticalalignment='top', color='blue')
        ax1.text(23, 0.55, f'R² = {r_square_classifer:.3f}', fontsize=18,
                verticalalignment='top', color='#ff7f0e')
        
        # ax1.set_ylabel('STAGE & HUMAN\nStyle Value Comparison')
        if i == 0:
            ax1.set_title('Style Value Alignment Evaluation')
        
        
        if i==9:
            ax1.set_xlabel('Time (s)')
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=2, frameon=True, fancybox=True)
        else:
            ax1.set_xticklabels([])  # 隐藏x轴刻度标签

    plt.subplots_adjust(left=0.18, bottom=0.2, top=0.95)
    # plt.tight_layout()
    # plt.savefig('./style_value_alignment_evaluation.png', dpi=300, bbox_inches='tight')
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

    run_data()
