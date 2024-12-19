import numpy as np
import matplotlib.pyplot as plt

def get_speed_and_throttle(speed_path, throttle_path):
    speed_data = np.load(speed_path)
    throttle_data = np.load(throttle_path)
    throttle_data = np.clip(throttle_data, -1, 1)
    return speed_data, throttle_data

def plot_throttle_distribution(throttle_list):
    plt.figure(figsize=(12, 6))
    plt.rcParams.update({
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'legend.fontsize': 14,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16
    })
    
    # 定义区间
    bins = [-np.inf, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['<0', '0~0.2', '0.2~0.4', '0.4~0.6', '0.6~0.8', '0.8~1.0']
    
    # 设置柱状图的位置
    x = np.arange(len(labels))
    width = 0.25  # 每个柱子的宽度
    
    # 计算每组数据的百分比
    percentages_list = []
    for throttle_data in throttle_list:
        hist, _ = np.histogram(throttle_data, bins=bins)
        percentages = hist / len(throttle_data) * 100
        percentages_list.append(percentages)
    
    # 绘制三组柱状图
    plt.bar(x - width, percentages_list[0], width, label='Without Action Modality', color='lightcoral', edgecolor='black')
    plt.bar(x, percentages_list[1], width, label='without VAE', color='lightgreen', edgecolor='black')
    plt.bar(x + width, percentages_list[2], width, label='STAGE(ours)', color='lightskyblue', edgecolor='black')
    
    plt.title('Acceleration and Deceleration Behavior Comparison')
    plt.xlabel('Throttle and Brake Value(-1~1)')
    plt.ylabel('Percentage (%)')
    plt.xticks(x, labels, rotation=45)
    
    # 在柱子上标注百分比值
    for i in range(len(labels)):
        plt.text(x[i] - width, percentages_list[0][i], f'{percentages_list[0][i]:.1f}', 
                ha='center', va='bottom', fontsize=13)
        plt.text(x[i], percentages_list[1][i], f'{percentages_list[1][i]:.1f}', 
                ha='center', va='bottom', fontsize=13)
        plt.text(x[i] + width, percentages_list[2][i], f'{percentages_list[2][i]:.1f}', 
                ha='center', va='bottom', fontsize=13)
    
    plt.legend()
    plt.tight_layout()

if __name__ == '__main__':
    speed_path = 'ablation_data/novae_noprefer_policy_best.ckpt_speed.npy'
    throttle_path = 'ablation_data/novae_noprefer_policy_best.ckpt_throttle.npy'
    speed_data1, throttle_data1 = get_speed_and_throttle(speed_path, throttle_path)
    
    speed_path = 'ablation_data/novae_policy_best.ckpt_speed.npy'
    throttle_path = 'ablation_data/novae_policy_best.ckpt_throttle.npy'
    speed_data2, throttle_data2 = get_speed_and_throttle(speed_path, throttle_path)
    
    speed_path = 'ablation_data/kl10_prefer10_policy_best.ckpt_speed.npy'
    throttle_path = 'ablation_data/kl10_prefer10_policy_best.ckpt_throttle.npy'
    speed_data3, throttle_data3 = get_speed_and_throttle(speed_path, throttle_path)
    
    throttle_list = [throttle_data1, throttle_data2, throttle_data3]
    plot_throttle_distribution(throttle_list)
    plt.show()
