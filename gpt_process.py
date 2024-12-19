from openai import OpenAI
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import List, Dict, Any
import tenacity
import numpy as np
import logging

# 禁用 httpx 的 INFO 日志
logging.getLogger("httpx").setLevel(logging.WARNING)

key = ""
# CHECK: remove this before upload to git


def create_client():
    return OpenAI(
        api_key=key,
        base_url="https://api.chatanywhere.tech/v1",
    )


@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),  # 最多重试5次
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),  # 指数退避重试
    retry=tenacity.retry_if_exception_type((Exception)) | tenacity.retry_if_result(lambda x: x['content'] not in ['1', '2']),  # 异常重试或结果不符合要求重试
    before_sleep=lambda retry_state: print(
        f"重试第 {retry_state.attempt_number} 次...")
)
def single_api_call(client: OpenAI, message: str) -> Dict[str, Any]:
    """单个API调用任务，带有重试机制"""
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": message
                }
            ]
        )
        
        result = {
            'content': completion.choices[0].message.content.strip(),
            'usage': {
                'completion_tokens': completion.usage.completion_tokens,
                'prompt_tokens': completion.usage.prompt_tokens
            }
        }
        
        if result['content'] not in ['1', '2']:
            print(f"Invalid response: {result['content']}, retrying...")
            
        return result
        
    except Exception as e:
        print(f"Call API error: {str(e)}")
        raise  # 重新抛出异常以触发重试


def multithread_api_call(messages: List[str], workers: int = 3, persistent_dict=None):
    """
    并行处理多个API调用，支持结果持久化

    Args:
        messages: 要发送的消息列表，每两个元素为一组(prompt, key_tuple)
        workers: 并行工作线程数
        persistent_dict: 持久化的字典，用于存储结果

    Returns:
        tuple: (所有回复列表, 总completion_tokens, 总prompt_tokens)
    """
    client = create_client()
    results = []
    total_completion_tokens = 0
    total_prompt_tokens = 0

    # 使用线程池并行处理请求
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for i in range(0, len(messages), 1):
            prompt = messages[i][0]
            key_tuple = messages[i][1]
            
            # 检查是否已有缓存结果
            if persistent_dict is not None and key_tuple in persistent_dict:
                results.append(str(persistent_dict[key_tuple]))
                continue
                
            # 没有缓存结果，提交API调用任务
            future = executor.submit(single_api_call, client, prompt)
            futures.append((future, key_tuple))

        # 收集结果
        for future, key_tuple in futures:
            try:
                result = future.result()
                results.append(result['content'])
                total_completion_tokens += result['usage']['completion_tokens']
                total_prompt_tokens += result['usage']['prompt_tokens']
                
                # 保存结果到持久化字典
                if persistent_dict is not None:
                    persistent_dict[key_tuple] = int(result['content'])
                    
            except Exception as e:
                print(f"multithread error: {str(e)}")
                raise e

    return results, total_completion_tokens, total_prompt_tokens

def format_data(array):
        return np.array2string(np.round(array, decimals=3), separator=',', suppress_small=True)

def generate_chinese_prompt(prefer_dict1: Dict[str, Any], prefer_dict2: Dict[str, Any]):
    description = "[Description] \n1. 我将以固定格式提供两段车辆驾驶的信息data1和data2，你需要结合data中的状态和车辆的行为，判断其中哪个data的驾驶policy更激进，并输出更激进的组号（1或2）。\n\
2. 数据以0时刻帧自身车辆位置为原点(0,0)，以自身车头朝向为x轴，以自身左侧为y轴的自动驾驶信息，包含未来的最多10个帧，帧间距0.1秒，作为判别的依据 \n\
3. 每一组数据中，ego_trajectory代表车辆从此刻到未来的轨迹，单位米，形状为(n,2)，n代表未来帧数，2代表x,y坐标，x代表向前，y代表向左；\
has_nearest代表50米内是否有其他车辆，True是有，则需要关注周围车辆对驾驶policy的影响，False是没有，则可以认为离自己最近的车辆也很远，对自己的policy不产生影响；\
nearest_other_pos代表在0时刻自身车辆坐标系下，相距最近的一个其他车未来的坐标，单位米，形状为(n,2)，代表时间和x,y坐标；\
nearest_distance代表在未来每个时刻，自身车辆到最近的其他车辆的距离，单位米；\
ego_vel_kmh代表在未来每个时刻，自身车辆的速度，单位km/h；\
ego_lane_diff_angle代表在未来每个时刻，自身车辆与道路方向的夹角，单位rad，有绝对值大于0.1的值表示可能存在变道的情况；\
ego_control代表在未来每个时刻，自身车辆的控制量，形状为(n,2)，其中每个时间刻为(steering, throttle)，steering代表方向盘转角，范围-1到1，throttle代表油门与刹车，范围-1到1，负数代表刹车正数代表油门。throttle与加速度成比例"

    rules = "[Rules] \n\
1. Only output 1 or 2, indicating that data 1 is more aggressive or data 2 is more aggressive, and do not output anything else \n\
2. 距离其他车辆小于10m时仍然继续缩短距离并超越具有最高的激进等级，其次大的激进等级是是更大的油门值，其次是更高的速度。\n\
3. 如果has_nearest为False，或其他车辆距离自己始终大于10m，无需注意提供的nearest_other_pos，nearest_distance的值\
"
    data = f"[Input]: \n\
[data1]: ego_trajectory: {format_data(prefer_dict1['ego_pos'])},\nhas_nearest: {prefer_dict1['has_nearest']},\nnearest_other_pos: {format_data(prefer_dict1['nearest_other_pos'])},\nnearest_distance: {format_data(prefer_dict1['nearest_distance'])},\nego_vel_kmh: {format_data(prefer_dict1['ego_vel_kmh'])},\nego_lane_diff_angle: {format_data(prefer_dict1['ego_lane_diff_angle'])},\nego_control: {format_data(prefer_dict1['ego_control'])} \n\
[data2]: ego_trajectory: {format_data(prefer_dict2['ego_pos'])},\nhas_nearest: {prefer_dict2['has_nearest']},\nnearest_other_pos: {format_data(prefer_dict2['nearest_other_pos'])},\nnearest_distance: {format_data(prefer_dict2['nearest_distance'])},\nego_vel_kmh: {format_data(prefer_dict2['ego_vel_kmh'])},\nego_lane_diff_angle: {format_data(prefer_dict2['ego_lane_diff_angle'])},\nego_control: {format_data(prefer_dict2['ego_control'])}"
        
    return description + "\n" + rules + "\n" + data

def generate_english_prompt(prefer_dict1: Dict[str, Any], prefer_dict2: Dict[str, Any]):
    description = "[Description]\n1. I will provide two vehicle driving data, data1 and data2, in a fixed format. You need to analyze the states and behaviors in each data to determine which data has a more aggressive driving policy, and output the more aggressive group number (1 or 2).\n\
2. The data is based on a driving scenario where the position of the vehicle at time 0 is set as the origin (0,0), with the vehicle's front direction as the x-axis and the left side as the y-axis. The data includes information for up to 10 future frames, with a frame interval of 0.1 seconds, which serves as the basis for making judgments.\n\
3. In each data set: **ego_trajectory** represents the vehicle's trajectory from the current moment to the future. The unit is meters, with a shape of (n,2), where n is the number of future frames, and 2 corresponds to the x and y coordinates. The x-axis represents forward movement, and the y-axis represents movement to the left;\n\
**has_nearest** indicates whether there are other vehicles within 50 meters. True means there is another vehicle, and the influence of surrounding vehicles on the driving policy needs to be considered. False means there are no other vehicles nearby, and the closest vehicle can be considered far enough away not to impact the driving policy;\n\
**nearest_other_pos** represents the future coordinates (in the vehicle's coordinate system at time 0) of the closest other vehicle. The unit is meters, and the shape is (n,2), representing time and x, y coordinates\n\
**nearest_distance** represents the distance between the vehicle and the closest other vehicle at each future time step, in meters;\n\
**ego_vel_kmh** represents the speed of the vehicle at each future time step, in km/h;\n\
**ego_lane_diff_angle** represents the angle between the vehicle's direction and the road direction at each future time step, in radians. Absolute values greater than 0.1 radians indicate a possible lane change;\n\
**ego_control** represents the control inputs of the vehicle at each future time step. Its shape is (n,2), where each time step is represented by (steering, throttle). Steering represents the steering wheel angle, with a range from -1 to 1. Throttle represents the throttle and brake input, with a range from -1 to 1. A negative value indicates braking, while a positive value indicates acceleration. Throttle is proportional to acceleration."

    rules = "[Rules] \n\
1. Only output 1 or 2, indicating that data 1 is more aggressive or data 2 is more aggressive, and do not output anything else; \n\
2. When the distance to other vehicles now is less than 10 meters, continuing to shorten the distance or overtaking represents the highest level of aggressiveness. The next level of aggressiveness is higher throttle values, followed by higher speed. \n\
3. If has_nearest is False, or if other vehicles are always more than 10 meters away, the values of nearest_other_pos and nearest_distance can be ignored."

    data = f"[Input]: \n\
[data1]: ego_trajectory: {format_data(prefer_dict1['ego_pos'])},\nhas_nearest: {prefer_dict1['has_nearest']},\nnearest_other_pos: {format_data(prefer_dict1['nearest_other_pos'])},\nnearest_distance: {format_data(prefer_dict1['nearest_distance'])},\nego_vel_kmh: {format_data(prefer_dict1['ego_vel_kmh'])},\nego_lane_diff_angle: {format_data(prefer_dict1['ego_lane_diff_angle'])},\nego_control: {format_data(prefer_dict1['ego_control'])} \n\
[data2]: ego_trajectory: {format_data(prefer_dict2['ego_pos'])},\nhas_nearest: {prefer_dict2['has_nearest']},\nnearest_other_pos: {format_data(prefer_dict2['nearest_other_pos'])},\nnearest_distance: {format_data(prefer_dict2['nearest_distance'])},\nego_vel_kmh: {format_data(prefer_dict2['ego_vel_kmh'])},\nego_lane_diff_angle: {format_data(prefer_dict2['ego_lane_diff_angle'])},\nego_control: {format_data(prefer_dict2['ego_control'])}"

    return description + "\n" + rules + "\n" + data

def test_api():
    # 测试用例
    test_messages = [
        "What is ChatGPT",
        "Hello",
        "How are you",
        "Who are you",
    ]

    results, completion_tokens, prompt_tokens = multithread_api_call(
        test_messages, workers=250)

    print("\n结果:")
    for i, result in enumerate(results, 1):
        print(f"\n回复 {i}:")
        print(result)

    print(f"\n总计使用:")
    print(
        f"Completion Tokens: {completion_tokens}, Prompt Tokens: {prompt_tokens}, price {completion_tokens / 1000 * 0.0042 + prompt_tokens / 1000 * 0.00105} CA")


if __name__ == "__main__":
    prefer_dict1 = {
        'episode_id': 100,
        'start_ts':1,
        'ego_pos': np.array(
        [[0.00000000e+00, 0.00000000e+00],
       [1.07903981e+00, 2.19584792e-03],
       [2.15443921e+00, 5.94835589e-03],
       [3.22619081e+00, 1.12319216e-02],
       [4.29569006e+00, 1.80096291e-02],
       [5.36867142e+00, 2.59136315e-02],
       [6.45193434e+00, 3.34817171e-02],
       [7.55311584e+00, 3.89650166e-02],
       [8.67763519e+00, 4.18379158e-02],
       [9.82976341e+00, 4.20278981e-02]]),
        'nearest_other_pos': np.array(
            [[ 6.39670801,  3.2288332 ],
            [ 7.22662067,  3.22115493],
            [ 8.05718994,  3.21347046],
            [ 8.88828468,  3.20578122],
            [ 9.71981525,  3.19808745],
            [10.55169678,  3.19039154],
            [11.38385296,  3.18269372],
            [12.21622181,  3.17499495],
            [13.0487833 ,  3.16729498],
            [13.88147926,  3.1595943 ]]
        ),
        'nearest_distance': np.array(
            [[7.16541954],
            [6.9393406 ],
            [6.71793601],
            [6.50111162],
            [6.28760917],
            [6.07269893],
            [5.85161153],
            [5.61954099],
            [5.37358518],
            [5.11230101]]),
        'has_nearest': True,
        'ego_vel_kmh': 
            np.array([[38.8716011 ],
            [38.74067688],
            [38.60974884],
            [38.47883224],
            [38.55764389],
            [38.86470032],
            [39.47439957],
            [40.27904892],
            [41.2623291 ],
            [42.32032013]]),
        'ego_lane_diff_angle': np.array(
            [[0.00934219],
            [0.0107975 ],
            [0.0122472 ],
            [0.0136863 ],
            [0.01510096],
            [0.01628292],
            [0.01648247],
            [0.01517856],
            [0.01295173],
            [0.01043975]]),
        'ego_control': np.array(
            [[ 0.00489807,  0.        ],
            [ 0.00489807,  0.        ],
            [ 0.00489807,  0.        ],
            [ 0.00480652,  0.        ],
            [ 0.00471497,  0.07450867],
            [ 0.0017395 ,  0.29019165],
            [-0.00498962,  0.57646179],
            [-0.00823975,  0.76077271],
            [-0.00823975,  0.92939758],
            [-0.00823975,  0.99998474]])
        }
    
    prefer_dict2 = {
        'episode_id': 200,
        'start_ts':5,
        'ego_pos': np.array(
        [[ 0.        , -0.        ],
       [ 0.98491371, -0.02581839],
       [ 1.97891891, -0.0689934 ],
       [ 2.98150539, -0.13185062],
       [ 3.99153113, -0.21969461],
       [ 5.00614405, -0.33916777],
       [ 6.01976442, -0.49375805],
       [ 7.02708244, -0.68096817],
       [ 8.02586174, -0.89284122],
       [ 9.0188303 , -1.11509299]]),
        'nearest_other_pos': np.array(
            [[0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.]]
        ),
        'nearest_distance': np.array(
            [[0.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.],
       [0.]]),
        'has_nearest': False,
        'ego_vel_kmh': 
            np.array([[35.39986038],
            [35.74900055],
            [36.09733582],
            [36.43878174],
            [36.75505066],
            [36.92254257],
            [36.91878128],
            [36.77838516],
            [36.65719223],
            [36.52869415]]),
        'ego_lane_diff_angle': np.array(
            [[0.06037986],
            [0.0618149 ],
            [0.06282544],
            [0.06268597],
            [0.05919898],
            [0.05046248],
            [0.03723609],
            [0.02323222],
            [0.01341617],
            [0.01422811]]),
        'ego_control': np.array(
            [[-0.0635376 ,  0.34117126],
            [-0.06344604,  0.34117126],
            [-0.0652771 ,  0.34117126],
            [-0.07699585,  0.34117126],
            [-0.09796143,  0.32940674],
            [-0.11878967,  0.19607544],
            [-0.12112427,  0.01960754],
            [-0.11000061,  0.        ],
            [-0.06893921,  0.        ],
            [-0.01295471,  0.        ]])
        }
    print(generate_chinese_prompt(prefer_dict1, prefer_dict2))
    # 加载或创建持久化字典
    try:
        persistent_dict = np.load("result_dict.npy", allow_pickle=True).item()
    except:
        persistent_dict = {}
        
    test_messages_with_id = [
        (generate_chinese_prompt(prefer_dict1, prefer_dict2), 
        (prefer_dict1['episode_id'], prefer_dict1['start_ts'], prefer_dict2['episode_id'], prefer_dict2['start_ts'])),
        (generate_english_prompt(prefer_dict1, prefer_dict2),
        (prefer_dict1['episode_id'], prefer_dict1['start_ts'], prefer_dict2['episode_id'], prefer_dict2['start_ts'])),
    ]

    results, completion_tokens, prompt_tokens = multithread_api_call(
        test_messages_with_id, workers=2, persistent_dict=persistent_dict)
    
    print("\n结果:")
    for i, result in enumerate(results, 1):
        print(f"\n回复 {i}:")
        print(result)

    print(f"\n总计使用:")
    print(
        f"Completion Tokens: {completion_tokens}, Prompt Tokens: {prompt_tokens}, price {completion_tokens / 1000 * 0.0042 + prompt_tokens / 1000 * 0.00105} CA")
    
    # 保存更新后的字典
    print(persistent_dict)
    np.save("result_dict.npy", persistent_dict)