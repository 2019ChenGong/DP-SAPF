import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import numpy as np
import matplotlib.pyplot as plt

# ---- Layout ----
datasets = ['CIFAR-10', 'CelebA', 'OCTMNIST']
models   = ['SD-v1-5', 'SD-2-1-base', 'Realistic-v6', 'Prompt2med']
ratio    = [5, 10, 20, 30, 40, 50, 60, 70]

# ---- Acc data ----
acc_data = [
    # CIFAR-10
    [[66.5, 68.5, 71.1, 74.6, 74.0, 70.7, 73.6, 15.7],   # SD-v1-5      
     [69.1, 68.8, 72.8, 72.3, 73.6, 72.3, 71.8, 15.1],   # SD-2-1-base  
     [62.8, 66.4, 66.9, 68.0, 70.6, 71.5, 71.8, 13.8],   # Realistic-v6 
     [68.1, 69.8, 72.6, 72.8, 76.2, 73.8, 75.4, 15.7]],  # Prompt2med   
    # CelebA
    [[82.1, 86.3, 87.7, 90.2, 89.5, 90.1, 88.1, 62.7],   # SD-v1-5      
     [84.3, 86.5, 90.3, 90.5, 90.6, 90.4, 90.2, 64.1],   # SD-2-1-base  
     [84.8, 89.2, 92.5, 92.1, 91.4, 91.0, 91.8, 65.7],   # Realistic-v6 
     [81.8, 83.9, 87.8, 88.5, 89.7, 90.4, 90.9, 61.5]],  # Prompt2med   
    # OCTMNIST
    [[39.2, 44.5, 45.3, 46.2, 45.9, 46.5, 42.7, 24.3],   # SD-v1-5      
     [35.8, 40.1, 41.3, 42.9, 44.8, 45.4, 44.6, 25.7],   # SD-2-1-base  
     [32.6, 36.8, 37.2, 38.5, 38.4, 37.6, 37.4, 24.5],   # Realistic-v6 
     [37.6, 42.7, 43.1, 43.4, 45.6, 44.2, 46.5, 25.8]],  # Prompt2med   
]

# ---- FID data ----
fid_data = [
    # CIFAR-10
    [[43.0, 30.0, 33.2, 26.6, 26.3, 32.2, 25.9, 192.3],  # SD-v1-5      
     [41.2, 31.5, 34.8, 27.2, 29.1, 33.6, 26.4, 188.5],  # SD-2-1-base  
     [46.5, 33.8, 37.1, 30.8, 31.5, 35.2, 29.3, 197.6],  # Realistic-v6 
     [39.8, 28.3, 31.5, 25.4, 27.6, 31.8, 24.9, 183.2]], # Prompt2med   
    # CelebA
    [[39.9, 28.9, 24.2, 23.6, 25.3, 25.2, 29.9, 257.3],  # SD-v1-5      
     [38.5, 27.4, 25.0, 24.0, 24.4, 26.2, 31.5, 251.7],  # SD-2-1-base  
     [36.2, 25.8, 18.4, 19.2, 19.4, 23.2, 27.4, 245.8],  # Realistic-v6 
     [40.8, 29.5, 27.3, 26.0, 27.5, 26.8, 32.1, 263.4]], # Prompt2med   
    # OCTMNIST
    [[105.3, 88.6, 81.2, 77.9, 75.5, 79.2, 73.8, 315.4], # SD-v1-5      
     [108.7, 91.3, 83.5, 80.0, 78.2, 81.5, 76.3, 322.6], # SD-2-1-base  
     [110.2, 93.8, 80.3, 81.7, 81.9, 83.0, 77.9, 331.8], # Realistic-v6 
     [126.5, 108.4, 101.7, 99.1, 96.8, 101.3, 94.5, 357.9]], # Prompt2med 
]

# ---- Style ----
color_acc = '#D76364'
color_fid = '#05B9E2'
marker_acc = 'v'
marker_fid = 'D'
lw       = 1.3
fontsize = 13
x = list(range(len(ratio)))

fig, axs = plt.subplots(3, 4, figsize=(17, 9))


def plot_one(ax, row, col, acc, fid):
    ax_r = ax.twinx()

    # --- Plot both lines ---
    ax.plot(x, acc, color=color_acc, marker=marker_acc,
            markersize=6.5, linewidth=lw, label='Acc')
    ax_r.plot(x, fid, color=color_fid, marker=marker_fid,
              markersize=6.5, linewidth=lw, label='FID')

    max_acc = max(acc)
    min_fid = min(fid)

    # --- Set FID ylim first (expand bottom for annotation) ---
    fid_ymin0, fid_ymax0 = ax_r.get_ylim()
    fid_ymin = fid_ymin0 - (fid_ymax0 - fid_ymin0) * 0.10
    fid_ymax = fid_ymax0
    ax_r.set_ylim(fid_ymin, fid_ymax)

    # --- Set Acc ylim: expand top, then lower bottom so min(acc) >= FID min line ---
    acc_ymin0, acc_ymax0 = ax.get_ylim()
    acc_ymax = acc_ymax0 + (acc_ymax0 - acc_ymin0) * 0.10

    # physical fraction of FID min line in the FID axis
    frac_fid = (min_fid - fid_ymin) / (fid_ymax - fid_ymin)

    # solve for acc_ymin such that min(acc) maps to frac_fid + margin (slightly above the line)
    acc_min = min(acc)
    target_frac = frac_fid + 0.06
    required_ymin = (acc_min - target_frac * acc_ymax) / (1 - target_frac)
    # take the lower of auto-min and required (i.e. expand if needed, never shrink)
    acc_ymin = min(acc_ymin0, required_ymin)
    ax.set_ylim(acc_ymin, acc_ymax)

    # --- Reference lines and annotations ---
    ax.axhline(y=max_acc, color='black', linestyle='--', linewidth=1.0, alpha=0.6, zorder=0)
    ax.text(x[0] - 0.15, max_acc, f'{max_acc:.1f}',
            fontsize=8, fontweight='bold', color='black', ha='left', va='bottom')

    ax_r.axhline(y=min_fid, color='black', linestyle=':', linewidth=1.0, alpha=0.6, zorder=0)
    ax_r.text(x[-1] + 0.15, min_fid, f'{min_fid:.1f}',
              fontsize=8, fontweight='bold', color='black', ha='right', va='top')

    # --- Labels ---
    if row == 0:
        ax.set_title(models[col], fontsize=fontsize, pad=6)
    if col == 0:
        ax.set_ylabel(f'{datasets[row]}\n\nAcc (%)', fontsize=fontsize, labelpad=12)
    if col == 3:
        ax_r.set_ylabel('FID', fontsize=fontsize)
    else:
        ax_r.set_yticklabels([])
    if row == 2:
        ax.set_xlabel('Selection Ratio (%)', fontsize=fontsize)

    ax.set_xticks(x, ratio)
    ax.tick_params(axis='both', labelsize=fontsize - 1)
    ax_r.tick_params(axis='y', labelsize=fontsize - 1)
    ax.grid(color='lightgrey', linewidth=1.0, zorder=0)

    # --- Legend (every subplot, center left) ---
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=fontsize * 0.75,
              loc='center left', framealpha=0.9, edgecolor='lightgray',
              fancybox=True, handlelength=2.2)


for row in range(3):
    for col in range(4):
        plot_one(axs[row][col], row, col,
                 acc_data[row][col], fid_data[row][col])

fig.subplots_adjust(wspace=0.25, hspace=0.35)
fig.savefig("selection_ratio_combined.png", bbox_inches='tight', dpi=150)
fig.savefig("selection_ratio_combined.pdf", bbox_inches='tight')
print("Saved selection_ratio_combined.png / .pdf")
