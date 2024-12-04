import os
import time

import numpy as np
import h5py
from metadrive import MetaDriveEnv, SafeMetaDriveEnv
from metadrive.constants import HELP_MESSAGE
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive_util.trajectory import WaypointTrajectory
import cv2

METADRIVE_DEBUG = False


def get_metadrive_config():
    config = dict(
        use_render=True,
        manual_control=True,
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
        window=True
    )

    return config, top_down_config


def calc_top_down_position(pix_position, agent_pix_position, heading):
    def rotation_matrix(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])

    pix_position, agent_pix_position = np.array(pix_position), np.array(agent_pix_position)
    offset = pix_position - agent_pix_position
    rotation = heading - np.pi / 2
    rotation_offset = np.dot(rotation_matrix(rotation), offset.T).T
    screen_pix_position = rotation_offset + np.array([224, 224]) / 2
    return screen_pix_position

def transform_traj_pos2pix(env, global_waypoints: np.ndarray):
    global_pix_waypoints = np.zeros_like(global_waypoints)
    for i in range(global_pix_waypoints.shape[0]):
        global_pix_waypoints[i] = env.top_down_renderer._frame_canvas.vec2pix(global_waypoints[i])
    return global_pix_waypoints

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

            while True:
                traj = WaypointTrajectory()
                global_waypoints = traj.local_waypoint_to_global(np.array([[1.0, 0.0], [1.8, 0.0], [3.0, 0.0], [4.5, 0.5],
                                                                           [6.0, 1.0], [8.0, 1.3], [10.0, 1.6], [12.0, 1.8],
                                                                           [14.0, 2.0], [16.0, 2.0]]),
                                              env.agent.position, env.agent.heading_theta)[1:]

                # env.render(text={'lane_idx': env.agent.lane.index})
                frame = env.render(**top_down_config)
                frame2 = frame.copy()
                film_agent_point = env.top_down_renderer._frame_canvas.pos2pix(env.agent.position[0],
                                                                               env.agent.position[1])
                # global coordinate to pixel coordinate
                film_draw_point = transform_traj_pos2pix(env, global_waypoints)
                screen_pix_position = calc_top_down_position(film_draw_point, film_agent_point, env.agent.heading_theta)
                for point in screen_pix_position.astype(int):
                    cv2.circle(frame2, tuple(point), radius=1, color=(255, 0, 0), thickness=-1)
                frame2=cv2.resize(frame2, (448,448))
                # cv2.circle(frame2, screen_pix_position.astype(int), radius=2, color=(0, 255, 0), thickness=-1)
                cv2.imshow("Image with Point", frame2)
                cv2.waitKey(1)

                observations.append(o)
                frames.append(frame)

                pos_now = env.agent.position
                theta_now = env.agent.heading_theta
                print(pos_now)
                now_positions.append(pos_now)
                headings.append([theta_now])

                o, r, tm, tc, info = env.step(np.array([0.0, 0.0]))
                pos_next = env.agent.position
                next_pos_actions.append(pos_next)
                # action：下一时刻的绝对位置

                if tm or tc:
                    if info['arrive_dest']:
                        print(f"success, save {log_episode} episode data...")
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
