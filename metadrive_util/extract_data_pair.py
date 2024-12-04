import os
import numpy as np
import h5py
camera_names = ['top_down_view']
import cv2
def get_raw_traj(dataset_path):
    with h5py.File(dataset_path, 'r') as root:
        episode_len = root['/action'].shape[0]

        start_ts = np.random.choice(episode_len)
        # get observation at start_ts only
        vec_data = {'lidar_scan': root['/observations/lidar_scan'][start_ts],
                    'side_detector': root['/observations/side_detector'][start_ts],
                    'lane_detector': root['/observations/lane_detector'][start_ts],
                    'navi_info': root['/observations/navi_info'][start_ts],
                    'ego_state': root['/observations/ego_state'][start_ts]
                    }

        image_dict = dict()
        for cam_name in camera_names:
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
        # get all actions after and including start_ts

        action = root['/action'][start_ts:]
        action_len = episode_len - start_ts

        heading_now = root['/heading'][start_ts].squeeze()
        position_now = root['/position_now'][start_ts]
    return vec_data, image_dict, action, action_len, position_now, heading_now

def get_pair():
    dataset_dir='../turn_left_dataset_manual'
    np.random.seed(5)
    total_episodes = len(os.listdir(dataset_dir))
    episode_id = np.random.choice(total_episodes)
    data1_path=os.path.join(dataset_dir, f'episode_{episode_id}.hdf5')
    episode_id = np.random.choice(total_episodes)
    data2_path=os.path.join(dataset_dir, f'episode_{episode_id}.hdf5')
    vec_data1, image_dict1, action1, action_len1, position_now1, heading_now1=get_raw_traj(data1_path)
    vec_data2, image_dict2, action2, action_len2, position_now2, heading_now2=get_raw_traj(data2_path)
    # image_dict1['top_down_view']=cv2.cvtColor(image_dict1['top_down_view'], cv2.COLOR_BGR2RGB)
    # image_dict2['top_down_view']=cv2.cvtColor(image_dict2['top_down_view'], cv2.COLOR_BGR2RGB)
    image_dict1['top_down_view'] = cv2.resize(image_dict1['top_down_view'], (300, 300))
    image_dict2['top_down_view'] = cv2.resize(image_dict2['top_down_view'], (300, 300))
    while cv2.waitKey(1) & 0xFF != ord('q'):
        cv2.imshow('image1', image_dict1['top_down_view'])
        cv2.imshow('image2', image_dict2['top_down_view'])
        image_shape = image_dict1['top_down_view'].shape


def main():
    get_pair()


if __name__ == "__main__":
    main()