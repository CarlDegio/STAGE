from torch import nn
import torch
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from utils import set_seed, detach_dict, compute_dict_mean, load_data
from metadrive_util.collect_dataset_manual import get_other_vehicle_dict, get_history_info, parse_data, save_episode_data
import numpy as np
from einops import rearrange


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim + action_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, state_action):
        x = state_action
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class Generator(nn.Module):
    def __init__(self, state_dim, action_dim, z_dim):
        super(Generator, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.fc1 = nn.Linear(state_dim + z_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, action_dim)
        self.activation = nn.ReLU()

    def forward(self, state, is_inference=False):
        if is_inference:
            z = torch.zeros(state.size(0), self.z_dim, device=state.device)
        else:
            z = torch.randn(state.size(0), self.z_dim, device=state.device)
        x = torch.cat([state, z], dim=1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


class GAIL(nn.Module):
    def __init__(self, state_dim, action_dim, z_dim):
        super(GAIL, self).__init__()
        self.action_dim=action_dim
        self.chunk_size = 10
        self.discriminator = Discriminator(state_dim, action_dim)
        self.generator = Generator(state_dim, action_dim, z_dim)
        self.optimizer_discriminator = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0001)
        self.optimizer_generator = torch.optim.Adam(
            self.generator.parameters(), lr=0.0001)

    def forward(self, state, is_inference=False):
        return self.generator(state, is_inference)


    def discriminator_loss(self, vec_dict, action_dict):
        state = stack_vec_dict(vec_dict)
        action = stack_vec_dict(action_dict).reshape(-1,self.action_dim)
        # 鉴别为专家
        real_data = torch.cat([state, action], dim=1)
        real_output = self.discriminator(real_data)
        real_loss = nn.functional.binary_cross_entropy(
            real_output, torch.ones_like(real_output))

        # train generator
        gen_action = self.generator(state)
        fake_data = torch.cat([state, gen_action], dim=1)
        fake_output = self.discriminator(fake_data)
        fake_loss = nn.functional.binary_cross_entropy(
            fake_output, torch.zeros_like(fake_output))

        return real_loss, fake_loss
    
    def generator_loss(self, vec_dict, action_dict):
        state = stack_vec_dict(vec_dict)
        
        gen_action = self.generator(state)
        gen_steer_throttle = gen_action[:, :2].reshape(-1,1,2)
        gen_traj_action = gen_action[:, 2:].reshape(-1, self.chunk_size, 2)
        l1_loss = nn.functional.l1_loss(gen_steer_throttle, action_dict['steer_throttle']) + 0.5*nn.functional.l1_loss(gen_traj_action, action_dict['traj_action'])

        fake_data = torch.cat([state, gen_action], dim=1)
        fake_output = self.discriminator(fake_data)
        judge_loss = nn.functional.binary_cross_entropy(
            fake_output, torch.ones_like(fake_output))
        
        return l1_loss, judge_loss
    
    def total_loss_dict(self, vec_dict, action_dict):
        real_loss, fake_loss = self.discriminator_loss(vec_dict, action_dict)
        l1_loss, judge_loss = self.generator_loss(vec_dict, action_dict)
        total_loss = real_loss + fake_loss + l1_loss + 0.01*judge_loss
        return {'real_loss': real_loss, 'fake_loss': fake_loss, 'l1_loss': l1_loss, 'judge_loss': judge_loss, 'total_loss': total_loss}
    
    def train_discriminator(self, state, action):
        # train discriminator
        real_loss, fake_loss = self.discriminator_loss(state, action)
        loss = real_loss + fake_loss
        # update discriminator
        self.optimizer_discriminator.zero_grad()
        loss.backward()
        self.optimizer_discriminator.step()
        return real_loss, fake_loss

    def train_generator(self, state, action):
        l1_loss, judge_loss = self.generator_loss(state, action)
        loss = l1_loss + 0.01*judge_loss
        self.optimizer_generator.zero_grad()
        loss.backward()
        self.optimizer_generator.step()
        return l1_loss, judge_loss

    def train_gail(self, vec_dict, action_dict):
        real_loss, fake_loss = self.train_discriminator(vec_dict, action_dict)
        l1_loss, judge_loss = self.train_generator(vec_dict, action_dict)
        return {'real_loss': real_loss, 'fake_loss': fake_loss, 'l1_loss': l1_loss, 'judge_loss': judge_loss, 'total_loss': real_loss + fake_loss + l1_loss + 0.01*judge_loss}

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']


    from constants import SIM_TASK_CONFIGS
    task_config = SIM_TASK_CONFIGS[task_name]

    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 14  # TODO: check

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'onscreen_render': onscreen_render,
        'task_name': task_name,
        'seed': args['seed'],
        'camera_names': camera_names,
    }

    if is_eval:
        ckpt_names = [
            f'policy_best.ckpt',
            #   f'policy_epoch_500_seed_0.ckpt',
            f'policy_epoch_700_seed_0.ckpt',
            #   f'policy_epoch_900_seed_0.ckpt'

        ]
        results = []
        for ckpt_name in ckpt_names:
            avg_return, avg_distance = eval_bc(
                config, ckpt_name, save_episode=True)
            results.append([ckpt_name, avg_return, avg_distance])

        for ckpt_name, avg_return, avg_distance in results:
            print(f'{ckpt_name}: {avg_return=}, {avg_distance=}')
        print('Eval Done!')
    else:
        train_dataloader, val_dataloader, stats, _ = load_data(
            dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)

        # save dataset stats
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)

        best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
        best_epoch, min_val_loss, best_state_dict = best_ckpt_info

        # save best checkpoint
        ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
        torch.save(best_state_dict, ckpt_path)
        print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')

def stack_vec_dict(vec_dict):
    output_list = []
    for key, value in vec_dict.items():
        if key == 'history_info':
            output_list.append(value.reshape(-1,5*40))
        else:
            output_list.append(value)
    return torch.cat(output_list, axis=1)

def train_pre_process(data):
    image_data, vec_dict, action_dict, is_pad, preference_dict = data
    for key in action_dict:
        action_dict[key] = action_dict[key].cuda()
    for key in vec_dict:
        vec_dict[key] = vec_dict[key].cuda()
    return vec_dict, action_dict


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']

    set_seed(seed)

    obs_dim = 536
    action_dim = 22
    z_dim = 32
    policy = GAIL(obs_dim, action_dim, z_dim)
    policy.cuda()

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                vec_dict, action_dict = train_pre_process(data)
                forward_dict = policy.total_loss_dict(vec_dict, action_dict)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['total_loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss,
                                  deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        for batch_idx, data in enumerate(train_dataloader):
            vec_dict, action_dict = train_pre_process(data)
            forward_dict = policy.train_gail(vec_dict, action_dict)
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(
            train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['total_loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            # policy.save_preference_dict()
            ckpt_path = os.path.join(
                ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history,
                         epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(
        ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(
        f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info

def get_image(frame):
    curr_images = []
    curr_image = rearrange(frame, 'h w c -> c h w')
    curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(
        curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def get_topdown_config():
    top_down_config = dict(
        mode="topdown",
        screen_size=(224, 224),
        scaling=2.5,
        target_agent_heading_up=True,
        semantic_map=True,
        window=True
    )
    return top_down_config

def eval_pre_process(vec_data, stats):
    for key in vec_data:
        vec_data[key] = (vec_data[key] - stats['vec_mean']
                         [key]) / stats['vec_std'][key]
    return vec_data

def eval_post_process(a, stats):  # TODO: dict check
    steer_throttle = a['steer_throttle'] * stats['steer_std'] + stats['steer_mean']
    traj = a['traj_action'] * stats['traj_std'] + stats['traj_mean']
    return {'steer_throttle': steer_throttle, 'traj': traj}

def get_action(policy, vec_data, curr_image, stats):
    vec_data = stack_vec_dict(vec_data)
    all_actions = policy(vec_data, is_inference=True)
    all_actions=all_actions.squeeze(0).cpu().numpy()
    all_actions = {'steer_throttle': all_actions[:2], 'traj_action': all_actions[2:].reshape(policy.chunk_size,2)}
    denorm_actions = eval_post_process(all_actions, stats)
    steer_throttle, local_traj = denorm_actions['steer_throttle'], denorm_actions['traj']
    return steer_throttle, local_traj


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    max_timesteps = config['episode_len']
    task_name = config['task_name']

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    obs_dim = 536
    action_dim = 22
    z_dim = 32
    policy = GAIL(obs_dim, action_dim, z_dim)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # load environment

    from sim_env import make_sim_env
    env = make_sim_env(task_name)
    # env_max_reward = env.task.max_reward


    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks

    num_rollouts = 20
    episode_distance = []
    highest_rewards = []
    save_flag = True
    speed_kmh_list = []
    steer_throttle_list = []
    # style_value_array = np.load(f'./temp_traj/style_value_5011_d0_3.npy')
    for rollout_id in range(num_rollouts):
        rollout_id += 0

        o, info = env.reset()
        drawer = env.engine.make_point_drawer(env.agent.origin, scale=1)
        drawer.setH(90)
        # ts = env.reset()

        # evaluation loop
        rewards = []
        observations = []
        frames = []
        next_pos_actions = []
        headings = []
        now_positions = []
        other_v_history = []
        history_infos = []

        with torch.inference_mode():
            reward = 0
            for t in range(max_timesteps):
                
                observations.append(o)
                now_positions.append(env.agent.position)
                headings.append([env.agent.heading_theta])

                other_v_dict = get_other_vehicle_dict(env)
                other_v_history.append(other_v_dict)
                history_info = get_history_info(
                    env, other_v_history, now_positions, headings)
                history_infos.append(history_info)
                vec_data = {}
                vec_data['lidar_scan'] = o[96:96 + 240]
                vec_data['side_detector'] = o[:40]
                vec_data['lane_detector'] = o[46:86]
                vec_data['navi_info'] = o[86:96]
                vec_data['ego_state'] = o[40:46]
                vec_data['history_info'] = history_info.reshape(5, 40)
                speed_kmh = vec_data['ego_state'][1:2] * (80 + 1) - 1
                speed_kmh_list.append(speed_kmh)
                vec_data = eval_pre_process(vec_data, stats)
                
                
                for key in vec_data:
                    vec_data[key] = torch.from_numpy(
                        vec_data[key]).float().cuda().unsqueeze(0)

                curr_image = env.render(**get_topdown_config())
                frames.append(curr_image)
                cv_image = curr_image.copy()
                curr_image = get_image(curr_image)


                steer_throttle1, local_traj1 = get_action(policy, vec_data, curr_image, stats)

                # drawer.reset()
                # traj = WaypointTrajectory()
                # traj.set_local_waypoint(
                #     local_traj1, env.agent.position, env.agent.heading_theta)
                # traj.draw_in_sim_local(drawer)

                # traj.set_local_waypoint(
                #     local_traj2, env.agent.position, env.agent.heading_theta)
                # traj.draw_in_sim_local(drawer, rgba=np.array([1, 0, 0, 1]))

                # traj.set_local_waypoint(
                #     local_traj3, env.agent.position, env.agent.heading_theta)
                # traj.draw_in_sim_local(drawer, rgba=np.array([0, 1, 0, 1]))

                # global_waypoints1 = traj.local_waypoint_to_global(local_traj1, env.agent.position, env.agent.heading_theta)[1:]
                # agent_pix_point = env.top_down_renderer._frame_canvas.pos2pix(env.agent.position[0],env.agent.position[1])
                # traj_film_pix_point1 = transform_traj_pos2pix(env, global_waypoints1)
                # screen_pix_position1 = calc_top_down_position(traj_film_pix_point1, agent_pix_point, env.agent.heading_theta)
                
                # cv_image=cv2.resize(cv_image,(448,448))
                # # 创建透明的overlay (BGRA格式)
                # overlay1 = np.zeros((448,448,4), dtype=np.uint8)
                # overlay2 = np.zeros((448,448,4), dtype=np.uint8)
                # overlay3 = np.zeros((448,448,4), dtype=np.uint8)
                
                # for point in screen_pix_position1.astype(int):
                #     cv2.circle(cv_image, tuple(2*point), radius=2, color=(0,0,255), thickness=-1)
                
                # cv2.imshow("Top-Down with Traj Point", cv_image)
                # cv2.waitKey(1)
                
                o, r, tm, tc, info = env.step(steer_throttle1.squeeze())
                steer_throttle_list.append(steer_throttle1.squeeze())
                next_pos_actions.append(env.agent.position)
                reward += r

                if tm or tc:
                    rewards.append(reward)
                    if info['arrive_dest'] and save_flag:
                        # style value evaluation用
                        print(f"success, save rollout_id {rollout_id} episode data...")
                        # data_dict = parse_data(observations, frames, next_pos_actions, now_positions, headings,
                        #                        history_infos) # 会修改值，测试平均里程时勿跑
                        # save_episode_data(data_dict, 'temp_traj', rollout_id+10)
                    break
                    

            # plt.close()
        episode_distance.append(sum([np.linalg.norm(np.array(now_positions[i+1]) - np.array(now_positions[i])) for i in range(len(now_positions)-1)]))
        rewards = np.array(rewards)
        avg_return = np.mean(rewards)
        # print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        # if save_episode:
        #     save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    avg_distance = np.mean(episode_distance)
    print(f'Episode distance mean: {avg_distance}')
    env.close()
    # draw_speed_throttle_distribution(speed_kmh_list, steer_throttle_list, ckpt_name)
    return avg_return, avg_distance

def draw_speed_throttle_distribution(speed_kmh_list, steer_throttle_list, ckpt_name):
    plt.figure(1)
    
    plt.subplot(1, 2, 1)
    speed_kmh_array = np.concatenate(speed_kmh_list)
    plt.hist(speed_kmh_array, bins=20, edgecolor='black')
    plt.title('Speed Distribution')
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    throttle_values = [x[1] for x in steer_throttle_list]  # Get throttle values
    throttle_values = np.array(throttle_values)
    plt.hist(throttle_values, bins=20, edgecolor='black')
    plt.title('Throttle Distribution') 
    plt.xlabel('Throttle Value')
    plt.ylabel('Count')
    
    plt.tight_layout()
    np.save(f'ablation_data/gail_{ckpt_name}_speed.npy', speed_kmh_array)
    np.save(f'ablation_data/gail_{ckpt_name}_throttle.npy', throttle_values)
    plt.show()

def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)),
                 train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)),
                 val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', default=True)
    parser.add_argument('--onscreen_render',
                        action='store_true', default=False)
    parser.add_argument('--ckpt_dir', action='store',
                        type=str, help='ckpt_dir', required=False, default='ckpt_gail')
    parser.add_argument('--task_name', action='store',
                        type=str, help='task_name', required=False, default='sim_drive')
    parser.add_argument('--batch_size', action='store',
                        type=int, help='batch_size', required=False, default=1024)
    parser.add_argument('--seed', action='store', type=int,
                        help='seed', required=False, default=0)
    parser.add_argument('--num_epochs', action='store',
                        type=int, help='num_epochs', required=False, default=1000)
    parser.add_argument('--lr', action='store', type=float,
                        help='lr', required=False, default=0.0001)

    main(vars(parser.parse_args()))
