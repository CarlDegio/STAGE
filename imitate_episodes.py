import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data  # data functions
from utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE
from metadrive_util.trajectory import WaypointTrajectory
from metadrive_util.turn_left_manual import get_other_vehicle_dict, get_history_info
import cv2
import IPython
from drive_style_gui.gui import StyleGUI
e = IPython.embed


def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    # is_sim:
    from constants import SIM_TASK_CONFIGS
    task_config = SIM_TASK_CONFIGS[task_name]

    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 14  # TODO: check
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 1
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone': backbone, 'num_queries': 1,
                         'camera_names': camera_names, }
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim
    }

    if is_eval:
        ckpt_names = [f'policy_best.ckpt', 
                      f'policy_epoch_100_seed_0.ckpt',
                      f'policy_epoch_200_seed_0.ckpt',
                      f'policy_epoch_300_seed_0.ckpt',
                      f'policy_epoch_400_seed_0.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, avg_return])

        for ckpt_name, avg_return in results:
            print(f'{ckpt_name}: {avg_return=}')
        print()
        exit()

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


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


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
        semantic_map=True
    )
    return top_down_config


def pre_process(vec_data, stats):
    for key in vec_data:
        vec_data[key] = (vec_data[key] - stats['vec_mean']
                         [key]) / stats['vec_std'][key]
    return vec_data


def post_process(a, stats):  # TODO: dict check
    steer_throttle = a['steer_throttle'] * stats['steer_std'] + stats['steer_mean']
    traj = a['traj_action'] * stats['traj_std'] + stats['traj_mean']
    return {'steer_throttle': steer_throttle, 'traj': traj}


def get_action(policy, vec_data, curr_image, stats, style_control=0.0):
    all_actions = policy(
        vec_data, curr_image, style_control=style_control)
    for key in all_actions:
        all_actions[key] = all_actions[key].squeeze(0).cpu().numpy()
    denorm_actions = post_process(all_actions, stats)
    steer_throttle, local_traj = denorm_actions['steer_throttle'], denorm_actions['traj']
    style_value = 1.234
    return style_value, steer_throttle, local_traj


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
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

    query_frequency = policy_config['num_queries']
    query_frequency = 1  # TODO: every query
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks

    num_rollouts = 3
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0

        o, info = env.reset()
        drawer = env.engine.make_point_drawer(env.agent.origin, scale=1)
        drawer.setH(90)
        # ts = env.reset()

        # evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        image_list = []  # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        other_v_history = []
        headings = []
        now_positions = []

        # gui = StyleGUI((10, 30))

        with torch.inference_mode():
            reward = 0
            for t in range(max_timesteps):

                now_positions.append(env.agent.position)
                headings.append([env.agent.heading_theta])

                other_v_dict = get_other_vehicle_dict(env)
                other_v_history.append(other_v_dict)
                history_info = get_history_info(
                    env, other_v_history, now_positions, headings)
                vec_data = {}
                vec_data['side_detector'] = o[:40]
                vec_data['ego_state'] = o[40:46]
                vec_data['lane_detector'] = o[46:86]
                vec_data['navi_info'] = o[86:96]
                vec_data['lidar_scan'] = o[96:96 + 240]
                vec_data['history_info'] = history_info.reshape(5, 40)
                vec_data = pre_process(vec_data, stats)
                for key in vec_data:
                    vec_data[key] = torch.from_numpy(
                        vec_data[key]).float().cuda().unsqueeze(0)

                curr_image = env.render(**get_topdown_config())
                curr_image = get_image(curr_image)

                # query policy
                if config['policy_class'] == "ACT":
                    if temporal_agg:
                        pass
                    else:
                        # read_sv = gui.read_style()
                        read_sv = 0.0
                        style_value1, steer_throttle1, local_traj1 = get_action(
                            policy, vec_data, curr_image, stats, style_control=read_sv)
                        style_value2, steer_throttle2, local_traj2 = get_action(
                            policy, vec_data, curr_image, stats, style_control=read_sv-2)
                        style_value3, steer_throttle3, local_traj3 = get_action(
                            policy, vec_data, curr_image, stats, style_control=read_sv+2)
                else:
                    raise NotImplementedError

                drawer.reset()
                traj = WaypointTrajectory()
                traj.set_local_waypoint(
                    local_traj1, env.agent.position, env.agent.heading_theta)
                traj.draw_in_sim_local(drawer)

                traj.set_local_waypoint(
                    local_traj2, env.agent.position, env.agent.heading_theta)
                traj.draw_in_sim_local(drawer, rgba=np.array([1, 0, 0, 1]))

                traj.set_local_waypoint(
                    local_traj3, env.agent.position, env.agent.heading_theta)
                traj.draw_in_sim_local(drawer, rgba=np.array([0, 1, 0, 1]))

                
                o, r, tm, tc, info = env.step(steer_throttle1.squeeze())
                reward += r

                if tm or tc:
                    rewards.append(reward)
                    break
                # onscreen render
                # if onscreen_render:
                    # ax = plt.subplot()
                    # plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
                    # plt.ion()
                    # env.render(text={'lane_idx': env.agent.lane.index, 'style_value': style_value1})  # , 'position': env.agent.position})
                    # env.render(text={'lane_idx': env.agent.lane.index, 'steering': env.agent.steering, })
                # for visualization
                # qpos_list.append(qpos_numpy)
                # target_qpos_list.append(target_qpos)
                    

            # plt.close()

        rewards = np.array(rewards)
        avg_return = np.mean(rewards)
        # print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        # if save_episode:
        #     save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    # success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    # avg_return = np.mean(episode_returns)
    # summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    # for r in range(env_max_reward+1):
    #     more_or_equal_r = (np.array(highest_rewards) >= r).sum()
    #     more_or_equal_r_rate = more_or_equal_r / num_rollouts
    #     summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    # print(summary_str)

    # save success rate to txt
    # result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    # with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
    #     f.write(summary_str)
    #     f.write(repr(episode_returns))
    #     f.write('\n\n')
    #     f.write(repr(highest_rewards))
    env.close()
    return avg_return


def forward_pass(data, policy):
    image_data, vec_data, action_data, is_pad, preference_dict = data
    image_data, is_pad = image_data.cuda(), is_pad.cuda()
    for key in action_data:
        action_data[key] = action_data[key].cuda()
    for key in vec_data:
        vec_data[key] = vec_data[key].cuda()
    return policy(vec_data, image_data, action_data, is_pad, prefer_dict=preference_dict)


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    # policy.load_state_dict(torch.load(os.path.join(f'./ckpt1', f'policy_best.ckpt')))
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

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
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
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
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(
            train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
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

    main(vars(parser.parse_args()))
