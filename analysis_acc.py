import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def get_speed_and_throttle(speed_path, throttle_path):
    speed_data = np.load(speed_path)
    throttle_data = np.load(throttle_path)
    throttle_data = np.clip(throttle_data, -1, 1)
    return speed_data, throttle_data

def plot_throttle_distribution(throttle_list):
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(14, 7))
    plt.rcParams.update({
        'axes.titlesize': 24,
        'axes.labelsize': 22,
        'legend.fontsize': 22,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'font.family': 'Arial'
    })
    
    bins = [-np.inf, 0, 0.2, 0.5, 0.8, np.inf]
    labels = ['-1~0(Brake)', '0~0.2(Light Throttle)', '0.2~0.5', '0.5~0.8', '0.8~1.0(Heavy Throttle)']
    labels = [labels[i] for i in [0, 4, 3, 2, 1]]
    x = np.arange(len(labels))
    width = 0.14
    
    colors = sns.color_palette("Blues", 6)  # Gradient from light to dark blue
    
    percentages_list = []
    for throttle_data in throttle_list:
        hist, _ = np.histogram(throttle_data, bins=bins)
        percentages = hist / len(throttle_data) * 100
        rearranged_percentages = [percentages[i] for i in [0, 4, 3, 2, 1]]
        percentages_list.append(rearranged_percentages)
        
    
    plt.bar(x - 2.5*width, percentages_list[0], width, label='GAIL', color=colors[0], edgecolor='black')
    plt.bar(x - 1.5*width, percentages_list[1], width, label='BC', color=colors[1], edgecolor='black')
    plt.bar(x - 0.5*width, percentages_list[2], width, label='CVAE', color=colors[2], edgecolor='black')
    plt.bar(x + 0.5*width, percentages_list[3], width, label='CVAE+Discrete Style', color=colors[3], edgecolor='black')
    plt.bar(x + 1.5*width, percentages_list[4], width, label='BC+Preference Style', color=colors[4], edgecolor='black')
    plt.bar(x + 2.5*width, percentages_list[5], width, label='STAGE(ours)', color=colors[5], edgecolor='black')

    plt.title('Comparison of Driving Comfort Across Methods')
    plt.xlabel('Throttle/Brake Value (-1: Full Brake, 1: Full Throttle)')
    plt.ylabel('Control Signal Distribution (%)')
    plt.xticks(x, labels, rotation=0.0)
    
    for i in range(len(labels)):
        for j, percentages in enumerate(percentages_list):
            offset = (j - 2.5) * width
            plt.text(x[i] + offset, percentages[i], f'{percentages[i]:.1f}', 
                     ha='center', va='bottom', fontsize=12)
    
    # max_percentage = max([percentages_list[j][-1] for j in range(len(percentages_list))])
    # max_idx = [percentages_list[j][-1] for j in range(len(percentages_list))].index(max_percentage)
    plt.arrow(1.25,50,1.5,0,head_width=2,head_length=0.1,fc='black',ec='black',linewidth=3)
    plt.annotate(f'comfort increase', 
                 xy=(2.25,40), 
                 xytext=(1.55,52),
                 fontsize=24)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', borderaxespad=0., ncol=3)
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.savefig('throttle_distribution.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    speed_path = 'ablation_data/gail_policy_best.ckpt_speed.npy'
    throttle_path = 'ablation_data/gail_policy_best.ckpt_throttle.npy'
    speed_data1, throttle_data1 = get_speed_and_throttle(speed_path, throttle_path)
    
    speed_path = 'ablation_data/novae_noprefer_policy_best.ckpt_speed.npy'
    throttle_path = 'ablation_data/novae_noprefer_policy_best.ckpt_throttle.npy'
    speed_data2, throttle_data2 = get_speed_and_throttle(speed_path, throttle_path)
    
    speed_path = 'ablation_data/cvae_policy_best.ckpt_speed.npy'
    throttle_path = 'ablation_data/cvae_policy_best.ckpt_throttle.npy'
    speed_data3, throttle_data3 = get_speed_and_throttle(speed_path, throttle_path)
    
    speed_path = 'ablation_data/kl10_class_style_policy_best.ckpt_speed.npy'
    throttle_path = 'ablation_data/kl10_class_style_policy_best.ckpt_throttle.npy'
    speed_data4, throttle_data4 = get_speed_and_throttle(speed_path, throttle_path)
    
    speed_path = 'ablation_data/novae_policy_best.ckpt_speed.npy'
    throttle_path = 'ablation_data/novae_policy_best.ckpt_throttle.npy'
    speed_data5, throttle_data5 = get_speed_and_throttle(speed_path, throttle_path)
    
    speed_path = 'ablation_data/kl10_prefer10_policy_best.ckpt_speed.npy'
    throttle_path = 'ablation_data/kl10_prefer10_policy_best.ckpt_throttle.npy'
    speed_data6, throttle_data6 = get_speed_and_throttle(speed_path, throttle_path)
    
    throttle_list = [throttle_data1, throttle_data2, throttle_data3, throttle_data4, throttle_data5, throttle_data6]
    plot_throttle_distribution(throttle_list)
    plt.show()
