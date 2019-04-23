import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

folder = "20190419_1134"
file = "parscan_base"
data = pd.read_csv("../../results/model/"+folder+"/"+file+".csv")

rho     = np.linspace(0,0.1,101)  # link density in erdos renyi network
phi     = np.linspace(1,1.5,101)   # efficiency of coordinated Workers
# pargrid = it.product(rho, phi)
print(data)

fig, (ax1,ax2) = plt.subplots(nrows = 1,ncols = 2, sharey = True)
# produced energy
grid = np.array(data.s).reshape((len(rho), len(phi)))
grid = np.flipud(grid.T)
im = ax2.imshow(grid, extent= (data.rho.min(), data.rho.max(), data.phi.min(), data.phi.max()),
    interpolation = "nearest", cmap = "Spectral_r", aspect = "auto",
    vmin = np.min([data.te,data.s]), vmax = np.max([data.te,data.s]))

grid = np.array(data.te).reshape((len(rho), len(phi)))
grid = np.flipud(grid.T)
im = ax1.imshow(grid, extent= (data.rho.min(), data.rho.max(), data.phi.min(), data.phi.max()),
    interpolation = "nearest", cmap = "Spectral_r", aspect = "auto",
    vmin = np.min([data.te,data.s]), vmax = np.max([data.te,data.s]))
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.11, 0.05, .77])
fig.colorbar(im, cax = cbar_ax)

fig.text(0.05, 0.5, 'Efficiency ($\\phi$)', ha='center',
    va='center', rotation='vertical')
fig.text(0.5, 0.04, 'Link Density ($\\rho$)', ha='center', va='center')
ax1.annotate("A", xy=(0.03, 0.95), xycoords="axes fraction", color = "black")
ax2.annotate("B", xy=(0.03, 0.95), xycoords="axes fraction", color = "black")
plt.savefig("../../results/model/"+folder+"/"+file+".png")
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
