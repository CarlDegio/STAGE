import os
import time

import numpy as np
import h5py
from metadrive import MetaDriveEnv, SafeMetaDriveEnv
from metadrive.constants import HELP_MESSAGE
from metadrive.component.pgblock.first_block import FirstPGBlock
from util.trajectory import WaypointTrajectory

METADRIVE_DEBUG = False


def get_metadrive_config():
    config = dict(
        use_render=True,
        # manual_control=True,
        # controller="steering_wheel",
        traffic_density=0.5,
        num_scenarios=200,
        start_seed=5008,
        random_agent_model=False,
        random_lane_width=False,
        random_lane_num=False,  # 3 lanes
        random_traffic=False,

        on_continuous_line_done=True,
        out_of_route_done=True,
        crash_vehicle_done=True,
        crash_object_done=True,

        vehicle_config=dict(
            show_lidar=True,
            show_navi_mark=True,
            side_detector=dict(num_lasers=40, distance=50, gaussian_noise=0.0, dropout_prob=0.0),
            lane_line_detector=dict(num_lasers=40, distance=20, gaussian_noise=0.0, dropout_prob=0.0),
            show_side_detector=True,
            show_lane_line_detector=True,
        ),
        # agent_configs={'default_agent': dict(use_special_color=True,
        #                                      spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0),
        #                                      destination='1X2_1_', )},
        map=1,
        accident_prob=0.0,
        horizon=400,

        debug=METADRIVE_DEBUG,
        debug_static_world=METADRIVE_DEBUG
    )

    top_down_config = dict(
        mode="topdown",
        screen_size=(224, 224),
        scaling=2.5,
        target_agent_heading_up=True,
        semantic_map=True,
        window=False
    )

    return config, top_down_config


def parse_data(observations, frames, positions_next, positions_now, headings, history_infos):
    """
        o[0:40] side_detector
        o[40:46] heading_error velocity steering last_action(2) yaw_rate
        o[46:86] lane_detector
        o[86:96] navi_info
        o[96:96+240] lidar_scan
        history_info: 5*10*4, ego and other vehicle history trajectory
    """
    data_dict = {
        '/observations/images/top_down_view': [],
        '/observations/lidar_scan': [],
        '/observations/side_detector': [],
        '/observations/lane_detector': [],
        '/observations/navi_info': [],
        '/observations/ego_state': [],
        '/observations/history_info': [],
        '/heading': [],
        '/position_now': [],
        '/action': []
    }
    while positions_next:
        obs = observations.pop(0)
        frame = frames.pop(0)
        action = positions_next.pop(0)
        heading = headings.pop(0)
        position_now = positions_now.pop(0)
        history_info = history_infos.pop(0)
        data_dict['/observations/side_detector'].append(obs[:40])
        data_dict['/observations/ego_state'].append(obs[40:46])
        data_dict['/observations/lane_detector'].append(obs[46:86])
        data_dict['/observations/navi_info'].append(obs[86:96])
        data_dict['/observations/lidar_scan'].append(obs[96:96 + 240])
        data_dict['/observations/history_info'].append(history_info)

        data_dict['/observations/images/top_down_view'].append(frame)

        data_dict['/action'].append(action)
        data_dict['/heading'].append(heading)
        data_dict['/position_now'].append(position_now)
    return data_dict


def save_episode_data(data_dict, dataset_dir, episode_idx):
    """
        For each timestep:
        observations
        - images
            - top_down_view     (224, 224, 3) 'uint8'    --> (bs,512,7,7)
        - lidar_scan            (240, )        'float64'
        - side_detector         (40, )         'float64'
        - lane_detector         (40, )         'float64'
        - navi_info             (10, )          'float64'
        - ego_state             (6, )          'float64'

        heading(now heading)    (1, )         'float64'
        position_now(now pos)   (2, )         'float64'
        position(next pos)      (2, )         'float64'
        history_info            (5, 10, 4)    'float64'
    """
    t0 = time.time()
    dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
    time_length = len(data_dict['/action'])
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = True
        root.attrs['length'] = time_length
        obs = root.create_group('observations')
        image = obs.create_group('images')
        _ = image.create_dataset('top_down_view', (time_length, 224, 224, 3), dtype='uint8',
                                 chunks=(1, 224, 224, 3), )
        # compression='gzip',compression_opts=2,)
        # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
        _ = obs.create_dataset('lidar_scan', (time_length, 240))
        _ = obs.create_dataset('side_detector', (time_length, 40))
        _ = obs.create_dataset('lane_detector', (time_length, 40))
        _ = obs.create_dataset('navi_info', (time_length, 10))
        _ = obs.create_dataset('ego_state', (time_length, 6))
        _ = obs.create_dataset('history_info', (time_length, 5, 10, 4))
        heading = root.create_dataset('heading', (time_length, 1))
        position_now = root.create_dataset('position_now', (time_length, 2))
        action = root.create_dataset('action', (time_length, 2))


        for name, array in data_dict.items():
            root[name][...] = array
    print(f'Saving: {time.time() - t0:.1f} secs\n')


def get_other_vehicle_dict(env: SafeMetaDriveEnv):
    surrounding_objects = env.observations['default_agent'].detected_objects
    surrounding_vehicles = list(env.engine.get_sensor("lidar").get_surrounding_vehicles(surrounding_objects))
    # print("surrounding_v:", surrounding_vehicles)
    vehicle_dict = {}
    for v in surrounding_vehicles:
        vehicle_dict[v.id] = [v.position, v.heading_theta]
    return vehicle_dict


def check_other_in_history(other_v_history, v_id):
    if len(other_v_history) < 10:
        return False
    # v_id exist 10 step in other_v_history
    for i in range(1, 11):
        if v_id not in other_v_history[-i]:
            return False
    return True


def get_history_info(env: SafeMetaDriveEnv, other_v_history, ego_pos_history, ego_heading_history, history_len=10,
                     num_other=4):
    ego_position = env.agent.position
    ego_heading = env.agent.heading_theta
    latest_other_v_dict = other_v_history[-1]
    dis_dict = {}
    for v_id, v in latest_other_v_dict.items():
        dis_dict[v_id] = np.linalg.norm(env.agent.position - v[0])
    sorted_id_dis_list = sorted(dis_dict.items(), key=lambda x: x[1])
    for _ in range(num_other):
        sorted_id_dis_list.append((None, None))

    history_info = np.zeros((num_other + 1, history_len, 4))
    history_index = 0

    waypoint = WaypointTrajectory()
    pos_history_relative = waypoint.global_waypoint_to_local(ego_pos_history[-history_len:], ego_position, ego_heading)[
                           1:]

    heading_history_relative = np.array(ego_heading_history[-history_len:]) - ego_heading
    # padding first until history_len
    while len(pos_history_relative) < history_len:
        pos_history_relative = np.insert(pos_history_relative, 0, pos_history_relative[0], axis=0)
    while len(heading_history_relative) < history_len:
        heading_history_relative = np.insert(heading_history_relative, 0, heading_history_relative[0], axis=0)

    history_info[history_index, :, 0:2] = pos_history_relative
    history_info[history_index, :, 2:3] = np.cos(heading_history_relative)
    history_info[history_index, :, 3:] = np.sin(heading_history_relative)
    history_index += 1
    for v_name, v_pos in sorted_id_dis_list:
        if history_index >= num_other + 1:
            break
        if v_name is not None:
            if check_other_in_history(other_v_history, v_name):
                for i in range(history_len, 0, -1):
                    v_pos_heading = other_v_history[-i][v_name]
                    v_pos_relative = waypoint.global_waypoint_to_local(v_pos_heading[0], ego_position, ego_heading)[1]
                    v_heading_relative = v_pos_heading[1] - ego_heading
                    history_info[history_index, 10 - i, 0:2] = v_pos_relative
                    history_info[history_index, 10 - i, 2:3] = np.cos(v_heading_relative)
                    history_info[history_index, 10 - i, 3:] = np.sin(v_heading_relative)
            else:
                continue
        else:
            history_info[history_index, :, 0:2] = np.zeros(2)
            history_info[history_index, :, 2] = np.zeros(1)
            history_info[history_index, :, 3] = np.zeros(1)
        history_index += 1

    return history_info


def main():
    config, top_down_config = get_metadrive_config()
    dataset_dir = 'scene/turn_left_dataset'
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    env = SafeMetaDriveEnv(config)

    try:
        o, info = env.reset()

        print(HELP_MESSAGE)

        if METADRIVE_DEBUG:
            env.engine.toggleDebug()

        env.agent.expert_takeover = False
        log_episode = 338
        for episode in range(log_episode, 400):
            observations = []
            frames = []
            next_pos_actions = []
            headings = []
            now_positions = []
            other_v_history = []
            history_infos = []

            while True:
                # env.render(text={'lane_idx': env.agent.lane.index})
                frame = env.render(**top_down_config)

                observations.append(o)
                frames.append(frame)

                pos_now = env.agent.position
                theta_now = env.agent.heading_theta
                # print(theta_now)
                now_positions.append(pos_now)
                headings.append([theta_now])

                other_v_dict = get_other_vehicle_dict(env)
                other_v_history.append(other_v_dict)
                history_info = get_history_info(env, other_v_history, now_positions, headings)
                history_infos.append(history_info)

                o, r, tm, tc, info = env.step(np.array([0.1, 0.2]))
                # print(env.agent.heading_theta)
                # print(env.agent.dist_to_left_side, env.agent.dist_to_right_side)
                print("ego_states",o[40:46])
                pos_next = env.agent.position
                next_pos_actions.append(pos_next)
                # action：下一时刻的绝对位置

                if tm or tc:
                    if info['arrive_dest']:
                        print(f"success, save {log_episode} episode data...")
                        data_dict = parse_data(observations, frames, next_pos_actions, now_positions, headings,
                                               history_infos)
                        # save_episode_data(data_dict, dataset_dir, log_episode)
                        log_episode += 1

                    env.config["traffic_density"] = np.random.uniform(0.0, 0.3)
                    o, info = env.reset()
                    env.agent.expert_takeover = False
                    break

    except Exception as e:
        raise e
    finally:
        env.close()


if __name__ == "__main__":
    main()
