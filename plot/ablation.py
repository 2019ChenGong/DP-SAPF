import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import mpl_toolkits.axisartist as axisartist

fontsize=19
matplotlib.rcParams.update({'font.size': fontsize, 'font.weight': 'normal'})

colors = ['#FFBE7A', '#8ECFC9', '#82B0D2', '#E0CBEF', '#FA7F6F', '#F7B7D2']
def create_multi_bars(ax, labels, datas, method, tick_step=1, group_gap=0.2, bar_gap=0, metric='fid'):

    ax.grid(color='lightgrey', linewidth=1, zorder=0)
    bwith = 0.7

    ticks = np.arange(len(labels)) * tick_step
    group_num = len(datas)
    group_width = tick_step - group_gap
    bar_span = group_width / group_num
    bar_width = bar_span - bar_gap
    baseline_x = ticks - (group_width - bar_span) / 2

    if metric == 'fid':
        best_idx = np.argmax(-datas[:, 0])
    else:
        best_idx = np.argmax(datas[:, 0])
    for index, y in enumerate(datas):
        ax.bar(baseline_x + index*bar_span, y, bar_width*0.7, label=method[index], color=colors[index], 
                zorder=200, edgecolor='black')

        if metric == 'fid':
            improve = datas[index, 0]
            if improve < 100:
                x_shift = 0.05
            else:
                x_shift = 0.06
        else:
            improve = datas[index, 0]
            x_shift = 0.05
        if metric == 'fid':
            y_shift = 1.3
            if datas[index, 0] > 100:
                y_shift = 3.5
        else:
            y_shift = 0.8
        if index == best_idx:
            ax.text(baseline_x + index*bar_span - x_shift, y+y_shift, str(improve), fontsize=13, fontweight='bold')
        else:
            ax.text(baseline_x + index*bar_span - x_shift, y+y_shift, str(improve), fontsize=13)

    ax.set_xticks([])
    ax.set_title(labels[0], fontsize=fontsize)
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    for spine in ax.spines.values():
        spine.set_zorder(300)

methods = ['DP-SAPF', 'Random', 'Noisy', 'w/o LoRA', 'Layer-Level', 'All Param.']
colors = ['#FFBE7A', '#E0CBEF', '#82B0D2', '#8ECFC9', '#FA7F6F', '#F7B7D2']
xlabels = ['CIFAR-10', 'CelebA', 'CIFAR-10', 'CelebA']


accs_cifar = [74.6, 55.2, 16.2, 69.3, 16.0, 41.0]
fids_cifar = [26.6, 65.3, 146.3, 30.1, 231.5, 63.0]
accs_celeba = [90.2, 60.4, 64.3, 88.8, 64.0, 76.3]
fids_celeba = [23.6, 71.3, 188.9, 28.7, 121.6, 76.3]

    
fig = plt.figure(figsize=(19, 4.), dpi=200)
axes = fig.subplots(1, 4)
data = np.array([fids_cifar, fids_celeba, accs_cifar, accs_celeba]).T
create_multi_bars(axes[0], xlabels, data, methods)
lines, labels = axes[0].get_legend_handles_labels()
axes[0].cla()

data = np.array([fids_cifar, fids_celeba, accs_cifar, accs_celeba][:1]).T
create_multi_bars(axes[0], xlabels[:1], data, methods)
data = np.array([fids_cifar, fids_celeba, accs_cifar, accs_celeba][1:2]).T
create_multi_bars(axes[1], xlabels[1:2], data, methods)
data = np.array([fids_cifar, fids_celeba, accs_cifar, accs_celeba][2:3]).T
create_multi_bars(axes[2], xlabels[2:3], data, methods, metric='acc')
data = np.array([fids_cifar, fids_celeba, accs_cifar, accs_celeba][3:]).T
create_multi_bars(axes[3], xlabels[3:], data, methods, metric="acc")

axes[0].set_ylabel('FID', fontsize=fontsize, weight='normal', labelpad=10)
axes[1].set_ylabel('FID', fontsize=fontsize, weight='normal', labelpad=10)
axes[2].set_ylabel('Acc (%)', fontsize=fontsize, weight='normal', labelpad=4.5)
axes[3].set_ylabel('Acc (%)', fontsize=fontsize, weight='normal', labelpad=1)

axes[0].set_ylim([0, 260])
axes[1].set_ylim([0, 210])
axes[2].set_ylim([0, 90])
axes[3].set_ylim([0, 110])

# leg = fig.legend(lines, labels, loc='upper center', ncol=6, facecolor='white', edgecolor='black', shadow=True, columnspacing=1, borderaxespad=0.1, fontsize=20)
# for legobj in leg.legendHandles:
#     legobj.set_linewidth(1.0)

fig.subplots_adjust(top=0.85)  # 👈 调整此值控制间隔大小

leg = fig.legend(
    lines, labels,
    loc='lower center',          # 👈 注意：此时 loc 参考 bbox_to_anchor 的位置
    bbox_to_anchor=(0.5, 0.95),   # 👈 (x, y) 相对于 figure 坐标系，1.0 表示顶部
    ncol=6,
    facecolor='white',
    edgecolor='black',
    shadow=True,
    columnspacing=1,
    borderaxespad=0.5,           # 此时作用变小，可保留默认
    fontsize=20
)

plt.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.8, wspace=0.3, hspace=0.3)
fig.savefig("ablation.png", bbox_inches='tight')
fig.savefig("ablation.pdf", bbox_inches='tight')
print(matplotlib.get_cachedir())