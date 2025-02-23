import os
import numpy as np
from matplotlib import pyplot as plt

def morphDemo():
    np.random.seed(42)
    ksz = 5
    n = 20
    nstep = n // ksz
    data = np.random.randint(0, 100, n)

    # Compute G and H
    G = data.copy()
    H = data.copy()
    for i in range(nstep):
        start_index = i * ksz + 1
        end_index = (i + 1) * ksz
        for j in range(start_index, end_index):
            G[j] = max(G[j - 1], G[j])
        start_index = i * ksz - 1
        end_index = (i + 1) * ksz - 2
        for j in range(end_index, start_index, -1):
            H[j] = max(H[j + 1], H[j])

    # 改进后的绘图设置
    # plt.style.use('seaborn')  # 使用更现代的样式
    plt.style.use('seaborn-v0_8')  # 使用 seaborn 样式
    fig, axes = plt.subplots(3, 1, figsize=(12, 4), dpi=100)
    vertical_shift = 120  # 各层之间的垂直间距
    
    # 配置颜色主题
    colors = plt.cm.tab10.colors  # 使用标准色环
    segment_colors = [colors[i%10] for i in range(nstep)]
    
    # 公共绘图参数
    line_params = {
        'linewidth': 2.5,
        'marker': 'o',
        'markersize': 8,
        'markerfacecolor': 'w',
        'markeredgewidth': 2
    }
    
    text_params = {
        'fontsize': 9,
        'ha': 'center',
        'va': 'bottom',
        'bbox': dict(facecolor='w', alpha=0.8, edgecolor='none', pad=2)
    }

    # 自定义 x 轴刻度值
    custom_xticks = [0, 5, 10, 15, 20]  # 指定你想显示的刻度值
    custom_xticklabels = ['0', '5', '10', '15', '20']  # 对应的刻度标签

    for ax_idx, (array, offset) in enumerate(zip([data, G, H], [200, 100, 0])):
        ax = axes[ax_idx]
        
        # 绘制背景辅助线
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)  # 网格线在数据下方
        
        # 均匀分布x轴坐标
        x_base = np.arange(n)  # 增加x轴间距
        
        for seg in range(nstep):
            start = seg * ksz
            end = (seg + 1) * ksz
            x_segment = x_base[start:end]
            y_segment = array[start:end] + offset
            
            # 绘制线段
            ax.plot(x_segment, y_segment, 
                    color=segment_colors[seg],
                    **line_params)
            
            # 添加数值标签
            for x, y in zip(x_segment, y_segment):
                ax.text(x, y, f"{y-offset}", 
                        color=segment_colors[seg],
                        **text_params)
                
        # 设置自定义 x 轴刻度
        ax.set_xticks([i for i in custom_xticks])  # 设置刻度位置
        ax.set_xticklabels(custom_xticklabels)  # 设置刻度标签
        
        # 美化坐标轴
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', which='both', length=0)
        ax.set_yticks([])
        ax.set_ylim(offset - 20, offset + 120)
        
        # 添加图层标签
        ax.text(-2, offset + 80, 
                ['Original', 'Stair Up', 'Stair Down'][ax_idx],
                ha='right', va='center', fontsize=12)

    plt.tight_layout(pad=3.0)
    os.makedirs("supplementary", exist_ok=True)
    plt.savefig("supplementary/stair.png", bbox_inches='tight', dpi=200)
    plt.close()

if __name__ == "__main__":
    morphDemo()