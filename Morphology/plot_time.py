import os
import re
import numpy as np
from matplotlib import pyplot as plt


def demo1():
    x = range(1, 6)
    y = [368.146, 297.507, 257.611, 158.593, 111.451]
    # plt.figure(figsize=(10, 8))
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.xlabel("Methods", fontdict={"family": "Times New Roman", "size": 16})
    plt.ylabel("Time(us)", fontdict={"family": "Times New Roman", "size": 16})
    plt.xticks(ticks=[1, 2, 3, 4, 5], labels=[1, 2, 3, 4, 5], fontproperties="Times New Roman", size=12)
    plt.yticks(fontproperties="Times New Roman", size=12)
    plt.ylim(0, 400)
    ax.plot(x, y)
    ax.scatter(x, y, color="red", marker="*")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.savefig("data/time_cost.png", dpi=200)


def autoRun():
    for radius in range(1, 31):
        os.system(f"/mnt/d/Project/CudaImageProcessing/Morphology/build/cuda_mophology {radius} 1 500")


def plotTime():
    # 1. 假设文档内容存储在一个字符串中（以示例说明）
    time_path = "build/cost.txt"
    with open(time_path, "r") as f:
        document = f.read().strip()

    # 2. 定义正则表达式来匹配 Radius 和 Running Time 数据
    pattern_radius = re.compile(r'Radius: (\d+)')
    pattern_running_time = re.compile(r'CUDA-Mophology: (\d+\.\d+)ms')

    # 3. 从文档中提取数据
    radius_data = pattern_radius.findall(document)
    running_time_data = pattern_running_time.findall(document)

    # 4. 转换为整数列表，并转换为毫秒
    radius_data = [int(value) for value in radius_data]
    running_time_data = [float(value) for value in running_time_data]

    # 5. 绘制折线图
    plt.figure(figsize=(10, 6))
    # plt.plot(range(len(radius_data)), radius_data, color='blue', marker='o')
    plt.plot(radius_data, running_time_data, color='red', marker='s')

    plt.xlabel("Radius")
    plt.ylabel("Time (ms)")
    plt.title("Time cost")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(radius_data, [str(i) for i in radius_data])
    plt.yticks(np.arange(0, 2.5, 0.5), [str(i) for i in np.arange(0, 2.5, 0.5)])
    plt.savefig("supplementary/time-cost.png", dpi=200)


if __name__ == "__main__":
    # demo1()
    # autoRun()
    plotTime()
