import matplotlib.pyplot as plt
import os
import numpy as np
from config_parser import ConfigParser
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 假设你的梯度值列表

def drawgram(config, gradients, num):
    # 最简单的折线图
    grad_values = [g.cpu().item() for g in gradients]
    plt.figure(figsize=(10, 6))
    plt.plot(grad_values, marker='o', linestyle='-', color='b', linewidth=2)
    # 添加100的基准线
    plt.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Danger (100)')
    plt.title('Gradient Norm Over Time')
    plt.xlabel('Batch')
    plt.ylabel('Gradient Norm')
    output_dir = os.path.join(config.get("output", "test_path"),
                              config.get("output", "model_name"))
    # 确保目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, "total") + "-all_grams-" + str(num) + ".png"

    plt.savefig(file_path, dpi=600, bbox_inches='tight', format="png")
    # 显示图形
    plt.close()
