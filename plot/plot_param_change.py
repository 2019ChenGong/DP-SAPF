import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter

# Set up figure and subplots
fig = plt.figure(figsize=(17, 3.3))
axs = fig.subplots(1, 4)

# Define methods, privacy budgets, colors, and markers
methods = ["Acc", "FID"]
eps = ['0.2', '1.0', '5.0', '10', '15', '20']
ratio = ['5', '10', '20', '30', '40', '50', '60', '70']
colors = ['#D76364', '#05B9E2']
markers = ['v', 'D']

# Process data
cifars = [[69.5, 73.1, 75.7, 74.6, 75.4, 75.9],
          [33.8, 32.1, 31.1, 26.6, 26.1, 25.9]]
celebas = [[78.8, 84.2, 87.1, 90.2, 89.9, 90.6],
          [26.9, 25.0, 24.0, 23.6, 23.1, 22.7]]
cifar_ss = [[66.5, 68.5, 71.1, 74.6, 74.0, 70.7, 73.6, 15.7],
          [43.0, 30.0, 33.2, 26.6, 26.3, 32.2, 25.9, 192.3]]
celeba_ss = [[82.1, 86.3, 87.7, 90.2, 89.5, 90.1, 88.1, 62.7],
          [39.9, 28.9, 24.2, 23.6, 25.3, 25.2, 29.9, 257.3]]

# Set line width and font size
lw = 1.3
fontsize = 14

# Plot MNIST subplot
color_ref = 'black'

def plot_one(ax, x1, data1, x2, data2, idx):
    ax_right = ax.twinx()
    max_acc = np.max(data1)
    min_fid = np.min(data2)
    max_acc_epoch = x1[np.argmax(data1)]  # 达到最大Acc的epoch
    min_fid_epoch = x2[np.argmin(data2)]  # 达到最小FID的epoch

    ax.axhline(y=max_acc, color=color_ref, linestyle='--', linewidth=1.2, alpha=0.7, zorder=0)
    
    # 标注位置：放在右侧边缘（避免遮挡数据），略高于虚线
    t_acc = [[-0.2, 0], [0.3, 0], [-0.2, 0], [-0.2, 0]]
    t_fid = [[-0.4, 0.5], [-0.4, 0.3], [-0.6, 10], [-0.6, 15]]
    ax.text(
        x1[0] + t_acc[idx][0],  # x位置：最后一个epoch右侧
        max_acc + t_acc[idx][1],  # y位置：虚线上方偏移
        f'{max_acc:.1f}', 
        fontsize=9, 
        fontweight='bold',
        color=color_ref,
        ha='left', 
        va='bottom',
    )
    
    # 2. FID 最小值水平线（右侧轴）
    ax_right.axhline(y=min_fid, color=color_ref, linestyle='--', linewidth=1.2, alpha=0.7, zorder=0)
    
    # 标注位置：放在右侧边缘，略低于虚线（FID越小越好，标注在下方更合理）
    ax_right.text(
        x2[-1] + t_fid[idx][0], 
        min_fid + t_fid[idx][1],  # y位置：虚线下方偏移
        f'{min_fid:.1f}', 
        fontsize=9, 
        fontweight='bold',
        color=color_ref,
        ha='left', 
        va='top',
    )

    ax.plot(x1, data1, 
                    color=colors[0], 
                    marker=markers[0], 
                    markersize=6.5, 
                    linewidth=lw, 
                    label='Acc')
    if idx == 0:
        ax.set_ylabel('Acc (%)', fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.grid(color='lightgrey', linewidth=1.0, zorder=0)
    # ax.set_title(exp['name'], fontsize=12, fontweight='bold', pad=10)

    # 设置 Accuracy 轴范围（可选）
    # ax_left.set_ylim(70, 100)

    # --- 右侧Y轴：FID ---
    ax_right.plot(x2, data2, 
                    color=colors[1], 
                    marker=markers[1], 
                    markersize=6.5, 
                    linewidth=lw, 
                    label='FID')
    if idx == 3:
        ax_right.set_ylabel('FID', fontsize=fontsize)
    ax_right.tick_params(axis='y', labelsize=fontsize)
    ax_right.tick_params(axis='x', labelsize=fontsize)

    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    
    # 合并并统一显示在右下角
    loc = 'lower left' if idx < 2 else 'upper left'
    if idx == 3:
        loc = 'center left'
    ax.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc=loc,          # 右下角位置（根据数据趋势选择）
        fontsize=fontsize*0.7,
        framealpha=0.92,            # 半透明背景避免遮挡数据
        edgecolor='lightgray',
        fancybox=True,
        ncol=1,                     # 单列布局
        handlelength=2.5            # 图例线长度
    )


plot_one(axs[0], [i for i in range(6)], cifars[0], [i for i in range(6)], cifars[1], 0)
plot_one(axs[1], [i for i in range(6)], celebas[0], [i for i in range(6)], celebas[1], 1)
plot_one(axs[2], [i for i in range(len(ratio))], cifar_ss[0], [i for i in range(len(ratio))], cifar_ss[1], 2)
plot_one(axs[3], [i for i in range(len(ratio))], celeba_ss[0], [i for i in range(len(ratio))], celeba_ss[1], 3)
axs[0].set_xticks([i for i in range(6)], eps)
axs[1].set_xticks([i for i in range(6)], eps)
axs[2].set_xticks([i for i in range(len(ratio))], ratio)
axs[3].set_xticks([i for i in range(len(ratio))], ratio)
axs[0].set_xlabel('Privacy Budget $\epsilon$', fontsize=fontsize)
axs[1].set_xlabel('Privacy Budget $\epsilon$', fontsize=fontsize)
axs[2].set_xlabel('Selection Ratio $c$ (%)', fontsize=fontsize)
axs[3].set_xlabel('Selection Ratio $c$ (%)', fontsize=fontsize)
axs[0].set_title('CIFAR-10', fontsize=fontsize)
axs[1].set_title('CelebA', fontsize=fontsize)
axs[2].set_title('CIFAR-10', fontsize=fontsize)
axs[3].set_title('CelebA', fontsize=fontsize)
# method_idx = [0, 1]
# for idx in method_idx:
#     method = methods[idx]
#     cifar = cifars[idx]
#     axs[0].plot(cifar, label=method, lw=lw, markersize=6.5, color=colors[idx], marker=markers[idx])
# axs[0].set_xticks([i for i in range(6)], eps)
# axs[0].set_ylabel("Acc (%)", fontsize=fontsize, labelpad=2)
# axs[0].set_xlabel("Privacy Budget $\epsilon$ (CIFAR-10)", fontsize=fontsize)
# # axs[0].set_yticks([0, 20, 40, 60, 80, 100])  # Set y-axis ticks
# # axs[0].set_ylim([-5, 105])  # Add padding to y-axis range
# axs[0].yaxis.tick_right()  # Move y-axis ticks to right
# axs[0].tick_params(axis='both', which='major', labelsize=14.5)
# axs[0].legend(fontsize=12.5)

# # Plot F-MNIST subplot
# for idx in method_idx:
#     method = methods[idx]
#     celeba = celebas[idx]
#     axs[1].plot(celeba, label=method, lw=lw, markersize=6.5, color=colors[idx], marker=markers[idx])
# axs[1].set_xticks([i for i in range(6)], eps)
# axs[1].set_xlabel("Privacy Budget $\epsilon$ (CelebA)", fontsize=fontsize, ha='left', position=(0, 0))
# axs[1].set_ylabel("FID", fontsize=fontsize, labelpad=2)
# # axs[1].set_yticks([0, 20, 40, 60, 80, 100])  # Set y-axis ticks
# # axs[1].set_ylim([-5, 105])  # Add padding to y-axis range
# axs[1].yaxis.set_label_position("right")  # FID label on right
# axs[1].yaxis.tick_right()  # Move y-axis ticks to right
# axs[1].tick_params(axis='both', which='major', labelsize=14.5)
# axs[1].legend(fontsize=12.5)

# # Plot F-MNIST subplot
# for idx in method_idx:
#     method = methods[idx]
#     cifar_s = cifar_ss[idx]
#     axs[2].plot(cifar_s, label=method, lw=lw, markersize=6.5, color=colors[idx], marker=markers[idx])
# axs[2].set_xticks([i for i in range(6)], eps)
# axs[2].set_xlabel("Privacy Budget $\epsilon$ (CelebA)", fontsize=fontsize, ha='left', position=(0, 0))
# axs[2].set_ylabel("FID", fontsize=fontsize, labelpad=2)
# # axs[1].set_yticks([0, 20, 40, 60, 80, 100])  # Set y-axis ticks
# # axs[1].set_ylim([-5, 105])  # Add padding to y-axis range
# axs[2].yaxis.set_label_position("right")  # FID label on right
# axs[2].yaxis.tick_right()  # Move y-axis ticks to right
# axs[2].tick_params(axis='both', which='major', labelsize=14.5)
# axs[2].legend(fontsize=12.5)

# # Plot F-MNIST subplot
# for idx in method_idx:
#     method = methods[idx]
#     celeba_s = celeba_ss[idx]
#     axs[3].plot(celeba_s, label=method, lw=lw, markersize=6.5, color=colors[idx], marker=markers[idx])
# axs[3].set_xticks([i for i in range(6)], eps)
# axs[3].set_xlabel("Privacy Budget $\epsilon$ (CelebA)", fontsize=fontsize, ha='left', position=(0, 0))
# axs[3].set_ylabel("FID", fontsize=fontsize, labelpad=2)
# # axs[1].set_yticks([0, 20, 40, 60, 80, 100])  # Set y-axis ticks
# # axs[1].set_ylim([-5, 105])  # Add padding to y-axis range
# axs[3].yaxis.set_label_position("right")  # FID label on right
# axs[3].yaxis.tick_right()  # Move y-axis ticks to right
# axs[3].tick_params(axis='both', which='major', labelsize=14.5)
# axs[3].legend(fontsize=12.5)

# Add grids
axs[0].grid(color='lightgrey', linewidth=1.0, zorder=0)
axs[1].grid(color='lightgrey', linewidth=1.0, zorder=0)
axs[2].grid(color='lightgrey', linewidth=1.0, zorder=0)
axs[3].grid(color='lightgrey', linewidth=1.0, zorder=0)

# Adjust subplot spacing
fig.subplots_adjust(wspace=0.4, hspace=0.27)

# Save figures
fig.savefig("param_change.png", bbox_inches='tight')
fig.savefig("param_change.pdf", bbox_inches='tight')