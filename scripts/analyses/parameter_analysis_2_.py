import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import sys
sys.path.append('../helpers/')
from shifted_cmap import shiftedColorMap


folder = "20190419_1134"
fb = "parscan_base"
fe = "parscan_expl0.02"
data_b = pd.read_csv("../../results/model/"+folder+"/"+fb+".csv")
data_e = pd.read_csv("../../results/model/"+folder+"/"+fe+".csv")
data_diff = np.array(data_e.te) - np.array(data_b.te)

print(data_b)
rho     = np.linspace(0,0.1,101)  # link density in erdos renyi network
phi     = np.linspace(1,1.5,101)   # efficiency of coordinated Workers
# pargrid = it.product(rho, phi)

orig_cmap = mpl.cm.RdBu
midpoint = (np.absolute(np.min([np.array(data_b.te),np.array(data_e.te),data_diff])) /
    (np.max([np.array(data_b.te),np.array(data_e.te),data_diff]) -
    np.min([np.array(data_b.te),np.array(data_e.te),data_diff])))
print(midpoint)
shifted_cmap = shiftedColorMap(orig_cmap, midpoint=midpoint, name='shiftedcmap')


fig, (ax1,ax2,ax3) = plt.subplots(nrows = 1,ncols = 3, sharey = True)
# produced energy
grid = np.array(data_b.te).reshape((len(rho), len(phi)))
grid = np.flipud(grid.T)
im = ax1.imshow(grid, extent= (data_b.rho.min(), data_b.rho.max(), data_b.phi.min(), data_b.phi.max()),
    interpolation = "nearest", cmap = shifted_cmap, aspect = "auto",
    vmin = np.min([np.array(data_b.te),np.array(data_e.te),data_diff]),
    vmax = np.max([np.array(data_b.te),np.array(data_e.te),data_diff]))

grid = np.array(data_e.te).reshape((len(rho), len(phi)))
grid = np.flipud(grid.T)
im = ax2.imshow(grid, extent= (data_e.rho.min(), data_e.rho.max(), data_e.phi.min(), data_e.phi.max()),
    interpolation = "nearest", cmap = shifted_cmap, aspect = "auto",
    vmin = np.min([np.array(data_b.te),np.array(data_e.te),data_diff]),
    vmax = np.max([np.array(data_b.te),np.array(data_e.te),data_diff]))

grid = data_diff.reshape((len(rho), len(phi)))
grid = np.flipud(grid.T)
im = ax3.imshow(grid, extent= (data_e.rho.min(), data_e.rho.max(), data_e.phi.min(), data_e.phi.max()),
    interpolation = "nearest", cmap = shifted_cmap, aspect = "auto",
    vmin = np.min([np.array(data_b.te),np.array(data_e.te),data_diff]),
    vmax = np.max([np.array(data_b.te),np.array(data_e.te),data_diff]))


fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.11, 0.05, .77])
fig.colorbar(im, cax = cbar_ax)

fig.text(0.05, 0.5, 'Efficiency ($\\phi$)', ha='center',
    va='center', rotation='vertical')
fig.text(0.5, 0.04, 'Link Density ($\\rho$)', ha='center', va='center')
ax1.annotate("A", xy=(0.03, 0.95), xycoords="axes fraction", color = "black")
ax2.annotate("B", xy=(0.03, 0.95), xycoords="axes fraction", color = "black")
ax3.annotate("C", xy=(0.03, 0.95), xycoords="axes fraction", color = "black")

plt.savefig("../../results/model/"+folder+"/parscan_te.png")
plt.show()


# fig, (ax1,ax3) = plt.subplots(1,2)
# for i in np.arange(0,len(data),int(np.sqrt(len(data)))):
#     d = data.sort_values(["phi","rho"])[i:i+11]
#     d.index = np.arange(11)
#     ax1.plot(d.rho, d.te, label = "phi = "+str(d.phi[1]))
#
# for i in np.arange(0,121,11):
#     d = data[i:i+11]
#     ax3.plot(d.phi, d.te, label = "rho = "+str(d.rho[i]))
# ax1.legend()
# ax3.legend()
# plt.subplots_adjust(wspace = .5)
# #plt.ylabel("produced energy per capita")
# ax3.set_xlabel("efficiency (phi)")
# ax1.set_xlabel("link density (rho)")
