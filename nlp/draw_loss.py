import matplotlib.pyplot as plt
import numpy as np
import os
from config_parser import ConfigParser
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
def plot_multi_losses(config, losses, num):
    """
    losses: list of lists, 形状为 [batch_num, task_num]
    """
    # 转换为numpy数组便于处理
    losses = np.array(losses)
    batch_num = len(losses)
    task_num = losses.shape[1]

    # 创建图形
    plt.figure(figsize=(12, 6))
    task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
    # 为每个任务绘制折线
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    task_names = task_name

    for i in range(task_num):
        plt.plot(range(batch_num),
                 losses[:, i],
                 color=colors[i % len(colors)],
                 linewidth=2,
                 label=task_names[i] if i < len(task_names) else f'Task {i + 1} Loss')

    # 设置图形属性
    plt.xlabel('Batch', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title(f'the {num} epoch Loss Curves', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_dir = os.path.join(config.get("output", "test_path"),
                              config.get("output", "model_name"))
    # 确保目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, "total") + "-all_losses" + str(num) + ".png"

    plt.savefig(file_path, dpi=600, bbox_inches='tight', format="png")
    # 显示图形
    plt.close()


def plot_losses_subplots(config, losses, num):
    """
    使用子图分别显示每个任务的loss
    """
    losses = np.array(losses)
    batch_num = len(losses)
    task_num = losses.shape[1]
    task_name = config.get("data", "type_of_label").replace(" ", "").split(",")
    fig, axes = plt.subplots(task_num, 1, figsize=(12, 4 * task_num))

    if task_num == 1:
        axes = [axes]

    task_names = task_name
    colors = ['blue', 'orange', 'green']

    for i in range(task_num):
        ax = axes[i]
        ax.plot(range(batch_num), losses[:, i],
                color=colors[i % len(colors)],
                linewidth=2)

        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title(f'the {num} epoch {task_names[i]} Loss Curve', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 添加最小值标记
        min_loss_idx = np.argmin(losses[:, i])
        min_loss = losses[min_loss_idx, i]
        ax.scatter(min_loss_idx, min_loss, color='red', s=100, zorder=5)
        ax.annotate(f'Min: {min_loss:.4f}',
                    xy=(min_loss_idx, min_loss),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=10)

    plt.tight_layout()

    output_dir = os.path.join(config.get("output", "test_path"),
                              config.get("output", "model_name"))
    # 确保目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, "total") + "-subplots_losses " + str(num) + ".png"

    plt.savefig(file_path, dpi=600, bbox_inches='tight', format="png")

    plt.close()

