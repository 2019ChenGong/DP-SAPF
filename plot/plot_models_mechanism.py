import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import numpy as np
import matplotlib.pyplot as plt

# ---- Data (DPSaFP always last) ----
left_methods  = ['DP-LoRA',  'DP-SAPF']
left_acc      = [61.2,  78.4]
left_fid      = [105.6, 61.1]

right_methods = ['EM',   'PTR', 'DP-SAPF']
right_acc     = [90.0,  89.9,   90.2]
right_fid     = [23.4,  24.6,   23.6]

# ---- Colors: one per method, DPSaFP always '#FFBE7A' ----
method_colors = {
    'DP-SAPF': '#FA7F6F',
    'DP-LoRA': '#82B0D2',
    'EM':      '#82B0D2',
    'PTR':     '#E0CBEF',
}

fontsize   = 15
TOTAL_SPAN = 3.0   # fixed x-data range for both subplots → same pixel bar width

fig, axs = plt.subplots(1, 2, figsize=(10, 4.0))


def plot_bars(ax, methods, acc_vals, fid_vals, title, acc_ylim, fid_ylim,
              bar_scale=1.0, show_acc_label=True, show_fid_label=True):
    ax_r = ax.twinx()
    n      = len(methods)
    colors = [method_colors[m] for m in methods]

    # spacing_w fixes bar positions & xlim; bar_scale only changes thickness
    spacing_w  = TOTAL_SPAN / (2 * n + 2.25)
    actual_bar = spacing_w * bar_scale
    group_gap  = spacing_w * 0.75

    acc_x = np.arange(n, dtype=float) * actual_bar
    fid_x = acc_x + n * actual_bar + group_gap

    for i in range(n):
        ax.bar(acc_x[i], acc_vals[i], actual_bar,
               color=colors[i], edgecolor='black', linewidth=0.8,
               label=methods[i], zorder=3)
        ax_r.bar(fid_x[i], fid_vals[i], actual_bar,
                 color=colors[i], edgecolor='black', linewidth=0.8,
                 zorder=3)

    ax.set_ylim(*acc_ylim)
    ax_r.set_ylim(*fid_ylim)

    # value labels
    def label_bars(axis, xs, vals):
        ylo, yhi = axis.get_ylim()
        offset = (yhi - ylo) * 0.018
        for xi, v in zip(xs, vals):
            axis.text(xi, v + offset, f'{v:.1f}',
                      ha='center', va='bottom', fontsize=fontsize * 0.72)

    label_bars(ax,   acc_x, acc_vals)
    label_bars(ax_r, fid_x, fid_vals)

    # x-axis: group-center labels
    ax.set_xticks([acc_x.mean(), fid_x.mean()])
    ax.set_xticklabels(['Acc (%)', 'FID'], fontsize=fontsize)
    margin = spacing_w * 0.75
    ax.set_xlim(acc_x[0] - margin, fid_x[-1] + margin)

    if show_acc_label:
        ax.set_ylabel('Acc (%)', fontsize=fontsize, labelpad=6)
    if show_fid_label:
        ax_r.set_ylabel('FID',   fontsize=fontsize, labelpad=6)
    ax.tick_params(axis='y', labelsize=fontsize - 1)
    ax_r.tick_params(axis='y', labelsize=fontsize - 1)
    ax.set_title(title, fontsize=fontsize, pad=8)
    ax.yaxis.grid(color='lightgrey', linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)

    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, fontsize=fontsize * 0.78, loc='upper right',
              framealpha=0.92, edgecolor='lightgray',
              fancybox=True, handlelength=1.6, handletextpad=0.5)


plot_bars(axs[0], left_methods,  left_acc,  left_fid,
          'DiT (CIFAR-10)',
          acc_ylim=(40, 105), fid_ylim=(0, 150), bar_scale=0.55,
          show_fid_label=False)

plot_bars(axs[1], right_methods, right_acc, right_fid,
          'DP Mechanism (CelebA)',
          acc_ylim=(88.5, 91.0), fid_ylim=(20, 33),
          show_acc_label=False)

fig.subplots_adjust(wspace=0.4)
fig.savefig("models_mechanism.png", bbox_inches='tight', dpi=150)
fig.savefig("models_mechanism.pdf", bbox_inches='tight')
print("Saved models_mechanism.png / .pdf")
