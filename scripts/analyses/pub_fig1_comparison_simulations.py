import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import numpy as np

# folder = "20190424_0141"
# folder = "20191121_1225"
# folder = "20191121_1250"
# folder = "20191121_1254"
# folder = "20191121_1448"
folder = "20191121_1734"

data_explore = pd.read_csv("../../results/model/" + folder + "/t5_explore.csv")
data_base = pd.read_csv("../../results/model/" + folder + "/t5_base.csv")
data_interme = pd.read_csv("../../results/model/" + folder + "/t5_intermediate.csv")

textwidth = 12.12537
plt.rcParams.update({'font.size': 14})
fig, (ax1, ax3, ax5) = plt.subplots(3, 1, sharex=True,
                                    figsize=(textwidth, textwidth / 2))
# norm = mpl.colors.Normalize(vmin=0, vmax=6)
cmap = cm.get_cmap("tab20", 20)
# cmap = cm.get_cmap("Dark2",8)
ax2 = ax1.twinx()
ax1.plot(np.array(data_base['x']),
         np.array(data_base['Admin']),
         ls='-', fillstyle='none', color=cmap(0))

ax1.fill_between(np.array(data_base['x']), np.array(data_base['A_pool_shk']),
                 alpha=.5, color=cmap(0))
ax1.fill_between(np.array(data_base['x']),
                 np.array(data_base['A_pool_exp']) + np.array(data_base['A_pool_shk']),
                 np.array(data_base['A_pool_shk']),
                 alpha=.5, color=cmap(1))
ax1.set_ylim(0, 1.25)
ax2.set_ylim(0, 1.25)
ax2.plot(np.array(data_base['x']),
         np.array(data_base['Ecap']),
         ls="-", lw=.5, color=cmap(2))

ax4 = ax3.twinx()
ax3.plot(data_interme['x'], data_interme['Admin'],
         ls='-', fillstyle='none',
         color=cmap(0))
ax3.fill_between(data_interme['x'], data_interme['A_pool_shk'],
                 alpha=.5, color=cmap(0))
ax3.fill_between(data_interme['x'],
                 data_interme['A_pool_exp'] + data_interme['A_pool_shk'],
                 data_interme['A_pool_shk'],
                 alpha=.5, color=cmap(1))
ax3.set_ylim(0, 1.25)
ax4.set_ylim(0, 1.25)
ax4.plot(data_interme['x'], data_interme['Ecap'],
         ls="-", lw=.5, color=cmap(2))

ax6 = ax5.twinx()
ax5.plot('x', 'Admin', ls='-', fillstyle='none',
         data=data_explore,
         label="$A_{total}$",
         color=cmap(0))
ax5.fill_between(data_explore['x'], data_explore['A_pool_shk'],
                 label="$A_{shock}$", alpha=.5, color=cmap(0))
ax5.fill_between(data_explore['x'], data_explore['A_pool_exp'] + data_explore['A_pool_shk'],
                 data_explore['A_pool_shk'],
                 label="$A_{exploration}$", alpha=.5, color=cmap(1))
ax5.set_ylim(0, 1.25)
ax6.set_ylim(0, 1.25)
ax6.plot(data_explore['x'], data_explore['Ecap'],
         label="$E_{prodcution}$", ls="-", lw=.5, color=cmap(2))

plt.setp(ax1.get_xticklabels(), visible=False)
ax4.set_ylim(0, max(np.append(data_base["Ecap"], data_explore['Ecap'])) + .1)
ax2.set_ylim(0, max(np.append(data_base["Ecap"], data_explore['Ecap'])) + .1)
fig.text(0.01, 0.5, 'Administrator share', ha='center',
         va='center', rotation='vertical')
fig.text(0.99, 0.5, 'Energy per capita', ha='center',
         va='center', rotation='vertical')
fig.text(0.5, 0.01, 'Time', ha='center', va='center')
fig.subplots_adjust(bottom=0.08, left=0.06, right=0.94, top=.95)
fig.legend(loc="upper center", bbox_to_anchor=(.5, .95), ncol=4,
           frameon=False)

ax1.annotate("A", xy=(0.01, 0.83), xycoords="axes fraction")
ax3.annotate("B", xy=(0.01, 0.83), xycoords="axes fraction")
ax5.annotate("C", xy=(0.01, 0.83), xycoords="axes fraction")
ax1.annotate("$p_{e}$ = 0", xy=(0.85, 0.83), xycoords="axes fraction")
ax3.annotate("$p_{e}$ = 0.00275", xy=(0.85, 0.83), xycoords="axes fraction")
ax5.annotate("$p_{e}$ = 0.02", xy=(0.85, 0.83), xycoords="axes fraction")
# fig.tight_layout()
plt.show()
fig.savefig("../../results/model/" + folder + "/pub_figure2.png")
