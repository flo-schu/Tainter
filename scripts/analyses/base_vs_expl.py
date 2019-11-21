import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import numpy as np

# folder = "20190424_0141"
folder = "20191121_1152"


data_explore = pd.read_csv("../../results/model/"+folder+"/t5_explore.csv")
data_base    = pd.read_csv("../../results/model/"+folder+"/t5_base.csv")

fig, (ax1,ax3) = plt.subplots(2,1,sharex=True)
# norm = mpl.colors.Normalize(vmin=0, vmax=6)
cmap = cm.get_cmap("tab20",20)
# cmap = cm.get_cmap("Dark2",8)
ax3.plot('x','Admin',ls='-',fillstyle='none', data = data_explore,
    label = "$A_{total}$", color = cmap(0))
ax3.fill_between(data_explore['x'],data_explore['A_pool_shk'],
    label = "$A_{shock}$",alpha = .5,color = cmap(0))
ax3.fill_between(data_explore['x'],data_explore['A_pool_exp']+data_explore['A_pool_shk'],
    data_explore['A_pool_shk'],
    label = "$A_{exploration}$",alpha = .5,color = cmap(1))
ax4 = ax3.twinx()
ax4.plot(data_explore['x'], data_explore['Ecap'],
    label = "$E_{prodcution}$", c = "g", ls = "-", lw = .5,color = cmap(2))

ax3.set_ylim(0,1)
ax2= ax1.twinx()
ax1.plot(np.array(data_base['x']),np.array(data_base['Admin']),ls='-',fillstyle='none',color = cmap(0))

ax1.fill_between(np.array(data_base['x']),np.array(data_base['A_pool_shk']),
    alpha = .5,color = cmap(0))
ax1.fill_between(np.array(data_base['x']),np.array(data_base['A_pool_exp'])+np.array(data_base['A_pool_shk']),
    np.array(data_base['A_pool_shk']),
    alpha = .5,color = cmap(1))
ax2.plot(np.array(data_base['x']), np.array(data_base['Ecap']), c = "g", ls = "-", lw = .5,color = cmap(2))

ax1.set_ylim(0,1)
plt.setp(ax1.get_xticklabels(), visible=False)
ax4.set_ylim(0,max(np.append(data_base["Ecap"],data_explore['Ecap']))+.1)
ax2.set_ylim(0,max(np.append(data_base["Ecap"],data_explore['Ecap']))+.1)
fig.legend(loc = "upper right", bbox_to_anchor = (.9,.88),ncol = 1)
fig.text(0.05, 0.5, 'Administrator share', ha='center',
    va='center', rotation='vertical')
fig.text(0.98, 0.5, 'Energy per capita', ha='center',
    va='center', rotation='vertical')
fig.text(0.5, 0.04, 'Time', ha='center', va='center')

ax1.annotate("A", xy=(0.01, 0.9), xycoords="axes fraction")
ax3.annotate("B", xy=(0.01, 0.9), xycoords="axes fraction")
plt.show()
fig.savefig("../../results/model/"+folder+"/Admin_Ecap_twocases.png")
