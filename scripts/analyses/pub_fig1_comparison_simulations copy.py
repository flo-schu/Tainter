import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, patches
import numpy as np


# fig 2A








# folder = "20190424_0141"
# folder = "20191121_1225"
# folder = "20191121_1250"
# folder = "20191121_1254"
# folder = "20191121_1448"
folder = "20191121_1734"

# path = "./results/model/"
path = "../../results/model/"
data_expl = pd.read_csv(path + folder + "/t5_explore.csv")
data_base = pd.read_csv(path + folder + "/t5_base.csv")
data_inte = pd.read_csv(path + folder + "/t5_intermediate.csv")

cmap_a = cm.get_cmap("Blues", 4)
cmap_e = cm.get_cmap("Oranges", 4)
textwidth = 12.12537
plt.rcParams.update({'font.size': 14})
fig = plt.figure(constrained_layout=True, figsize=(textwidth, textwidth / 2))
gs =  fig.add_gridspec(3, 3)
ax1 = fig.add_subplot(gs[0,0:2])
ax2 = ax1.twinx()
ax1b= fig.add_subplot(gs[0,2])
ax3 = fig.add_subplot(gs[1,0:3])
ax4 = ax3.twinx()
ax5 = fig.add_subplot(gs[2,0:3])
ax6 = ax5.twinx()

for i, ax, d in zip(range(3), [(ax1, ax2), (ax3, ax4), (ax5, ax6)], [data_base, data_inte, data_expl]):
    ax[0].plot(np.array(d['x']), np.array(d['Admin']), ls='-', fillstyle='none',
               color=cmap_a(3), label="$A_{total}$")
    ax[0].fill_between(np.array(d['x']), np.array(d['A_pool_shk']), alpha=.5,
                       color=cmap_a(3), label="$A_{shock}$")
    ax[0].fill_between(np.array(d['x']),
                       np.array(d['A_pool_exp']) + np.array(d['A_pool_shk']),
                       np.array(d['A_pool_shk']),
                       alpha=.25, color=cmap_a(3), label="$A_{exploration}$")
    ax[1].plot(np.array(d['x']), np.array(d['Ecap']), ls="-", lw=.5,
               color=cmap_e(2), label="$E_{prodcution}$")
    ax[0].set_ylim(0, 1.25)
    ax[1].set_ylim(0, 1.25)
    if i == 0:
        ax[0].set_xlim(0, 3250)
    else:
        ax[0].set_xlim(0, 5000)


# compute theoretical diminishing returns
def output(complexity, eff, loss):
    return complexity * eff - (complexity) ** loss + 0.93

def marginal(arr):
    return np.diff(arr)

complexity = np.linspace(0,10, num =100)
returns = output(complexity, 1.1, 1.11)

# compute moving average of empirical ecap
ecap = np.convolve(np.array(data_base['Ecap']), np.ones(5), 'valid') / 5
x = np.array(data_base['Admin'])
# plt.plot(complexity/10,returns, color="black")
# plt.plot(x[2:len(x)-2],ecap,'o', color=cmap_e(2), alpha=.5)
# plt.ylim(0.85,1.1)
# plt.xlim(0,0.3)
# patch = patches.Rectangle((0,0.85), 250, 1.1, facecolor="none", edgecolor="black")
# ax2.add_patch(patch)

ax1b.plot(x[2:len(x)-2], ecap, 'o', lw=1,
          color=cmap_e(2), label="$E_{prodcution}$")
ax1b.plot(complexity/10, returns, color="black", lw=1.5)

ax1b.set_ylim(0.85,1.1)
ax1b.set_xlim(0,0.3)

plt.setp(ax1.get_xticklabels(), visible=False)
ax1b.set_xticks([])
ax1b.set_yticks([])
ax1b.set_xlabel('administration')
ax1b.set_ylabel('energy')
# plt.setp(ax1b.get_xticklabels(), visible=False)
# plt.setp(ax1b.get_xticks(), visible=False)
# plt.setp(ax1b.get_yticks(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)

fig.text(0.01, 0.5, 'Administrator share', ha='center',
         va='center', rotation='vertical')
fig.text(0.99, 0.5, 'Energy per capita', ha='center',
         va='center', rotation='vertical')
fig.text(0.5, 0.01, 'Time', ha='center', va='center')
fig.subplots_adjust(bottom=0.08, left=0.06, right=0.94, top=.95)

ahandles, alabels = ax1.get_legend_handles_labels()
ehandles, elabels = ax2.get_legend_handles_labels()

handles = [ahandles[0], ehandles[0], ahandles[1], ahandles[2]]
labels = [alabels[0], elabels[0], alabels[1], alabels[2]]
ax3.legend(handles, labels, loc="right", ncol=2, frameon=True, fancybox=False)
           #bbox_to_anchor=(.3, .95))

ax1.annotate("A", xy=(0.005, 0.88), xycoords="axes fraction")
ax1b.annotate("B", xy=(0.94, 0.88), xycoords="axes fraction")
ax3.annotate("C", xy=(0.005, 0.88), xycoords="axes fraction")
ax5.annotate("D", xy=(0.005, 0.88), xycoords="axes fraction")
ax1.annotate("$p_{e}$ = 0", xy=(0.995, 0.88), xycoords="axes fraction",horizontalalignment="right")
ax3.annotate("$p_{e}$ = 0.00275", xy=(0.995, 0.88), xycoords="axes fraction",horizontalalignment="right")
ax5.annotate("$p_{e}$ = 0.02", xy=(0.995, 0.88), xycoords="axes fraction",horizontalalignment="right")
# fig.tight_layout()
fig.savefig("../../results/model/" + folder + "/pub_figure2.pdf")
fig.savefig("../../results/model/" + folder + "/pub_figure2.png", dpi=300)
plt.show()

