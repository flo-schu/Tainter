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
data_diff_st = np.array(data_e.s) - np.array(data_b.s)
rho     = np.linspace(0,0.1,101)  # link density in erdos renyi network
phi     = np.linspace(1,1.5,101)   # efficiency of coordinated Workers
# pargrid = it.product(rho, phi)
# print(pd.DataFrame(data_diff_st))
data_b["expl_te"] = data_e.te
data_b["diff_te"] = data_diff
data_b["diff_te_abs"] = np.abs(data_b.diff_te)
# for i, j in data_diff, data_b:

limits_up_rho = []
limits_up_phi = []
limits_low_phi = []
limits_low_rho = []




for i in np.arange(0,len(data_b),len(rho)):
    low_phi = low_rho = up_phi = up_rho = -99999999
    # print(temp)
    temp = data_b.iloc[i:i+len(rho)]
    # temp = temp[np.abs(temp.diff_te) == np.min(np.abs(temp.diff_te))]
    # temp_2 = temp.sort_values(["diff_te_abs"]).iloc[0:2]
    # temp_3 = temp_2.sort_values(["phi"])
    for i in np.arange(0, len(temp)-2):
        temp_1 = temp.iloc[i:(i+2)]
        temp_2 = np.array(temp_1.diff_te)
        # print(temp_2)

        if temp_2[0] >= 0 and temp_2[1] <= 0:
            temp_low = temp_1[temp_1.diff_te_abs == np.min(np.abs(temp_2))]
            low_rho = np.array(temp_low.rho)
            low_phi = np.array(temp_low.phi)
            # print(low_phi,low_rho)

    if low_phi == -99999999:
        # temp_default = temp[np.abs(temp.diff_te) == np.min(np.abs(temp.diff_te))]
        # low_rho = np.array([temp_default.rho.iloc[0]])
        low_rho = np.array([float("nan")])
        low_phi = np.array([float("nan")])

    for i in np.arange(0, len(temp)-2):
        temp_1 = temp.iloc[i:(i+2)]
        temp_2 = np.array(temp_1.diff_te)
        # print(temp_2)

        if temp_2[0] <= 0 and temp_2[1] >= 0:
            temp_low = temp_1[temp_1.diff_te_abs == np.min(np.abs(temp_2))]
            up_rho = np.array(temp_low.rho)
            up_phi = np.array(temp_low.phi)
            # print(low_phi,low_rho)

    if up_phi == -99999999:
        # temp_default = temp[np.abs(temp.diff_te) == np.min(np.abs(temp.diff_te))]
        # up_rho = np.array([temp_default.rho.iloc[0]])
        up_rho = np.array([float("nan")])
        up_phi = np.array([float("nan")])


    # print(low_phi,low_rho, up_phi, up_rho)


    limits_up_phi.append(up_phi[0])
    limits_up_rho.append(up_rho[0])
    limits_low_phi.append(low_phi[0])
    limits_low_rho.append(low_rho[0])

# print(limits_low_rho, limits_up_phi)
# input()

epmin = np.min([np.array(data_b.te),np.array(data_e.te),data_diff])
epmax = np.max([np.array(data_b.te),np.array(data_e.te),data_diff])
stmin = np.min([np.array(data_b.s),np.array(data_e.s),data_diff_st])
stmax = np.max([np.array(data_b.s),np.array(data_e.s),data_diff_st])
orig_cmap = mpl.cm.RdBu
orig_cmap_st = mpl.cm.Blues
print(stmin, stmax)
midpoint = np.absolute(epmin) / (epmax - epmin)
midpoint_st = np.absolute(stmin) / (stmax - stmin)
print(midpoint, midpoint_st)
shifted_cmap = shiftedColorMap(orig_cmap, midpoint=midpoint, name='shiftedcmap')
# shifted_cmap_st = orig_cmap
shifted_cmap_st = shiftedColorMap(orig_cmap_st, midpoint=0.5, name='shiftedcmap_st')

fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(nrows = 2,ncols = 3, sharey = True, sharex = True)
# produced energy
grid = np.array(data_b.te).reshape((len(rho), len(phi)))
grid = np.flipud(grid.T)
im = ax4.imshow(grid, extent= (data_b.rho.min(), data_b.rho.max(), data_b.phi.min(), data_b.phi.max()),
    interpolation = "nearest", cmap = shifted_cmap, aspect = "auto",
    vmin = epmin, vmax = epmax)
# ax4.plot(np.arange(0,0.1, 0.010), np.arange(1,1.5, 0.05), marker = "o", color = "red")
# ax4.plot()

grid = np.array(data_e.te).reshape((len(rho), len(phi)))
grid = np.flipud(grid.T)
im = ax5.imshow(grid, extent= (data_e.rho.min(), data_e.rho.max(), data_e.phi.min(), data_e.phi.max()),
    interpolation = "nearest", cmap = shifted_cmap, aspect = "auto",
    vmin = epmin, vmax = epmax)

grid = data_diff.reshape((len(rho), len(phi)))
grid = np.flipud(grid.T)
im = ax6.imshow(grid, extent= (data_e.rho.min(), data_e.rho.max(), data_e.phi.min(), data_e.phi.max()),
    interpolation = "nearest", cmap = shifted_cmap, aspect = "auto",
    vmin = epmin, vmax = epmax)
ax6.plot(limits_up_rho, limits_up_phi)
ax6.plot(limits_low_rho, limits_low_phi)

grid = np.array(data_b.s).reshape((len(rho), len(phi)))
grid = np.flipud(grid.T)
im2 = ax1.imshow(grid, extent= (data_b.rho.min(), data_b.rho.max(), data_b.phi.min(), data_b.phi.max()),
    interpolation = "nearest", cmap = shifted_cmap_st, aspect = "auto",
    vmin = stmin, vmax = stmax)

grid = np.array(data_e.s).reshape((len(rho), len(phi)))
grid = np.flipud(grid.T)
im2 = ax2.imshow(grid, extent= (data_e.rho.min(), data_e.rho.max(), data_e.phi.min(), data_e.phi.max()),
    interpolation = "nearest", cmap = shifted_cmap_st, aspect = "auto",
    vmin = stmin, vmax = stmax)

grid = data_diff_st.reshape((len(rho), len(phi)))
grid = np.flipud(grid.T)
im2 = ax3.imshow(grid, extent= (data_e.rho.min(), data_e.rho.max(), data_e.phi.min(), data_e.phi.max()),
    interpolation = "nearest", cmap = shifted_cmap_st, aspect = "auto",
    vmin = stmin, vmax = stmax)


fig.subplots_adjust(right=0.75)
cbar_ax_ep = fig.add_axes([0.81, 0.525, 0.05, .355])
cbar_ax_st = fig.add_axes([0.81, 0.11 , 0.05, .355])
fig.colorbar(im2, cax = cbar_ax_ep)
fig.colorbar(im, cax = cbar_ax_st)

fig.text(0.05, 0.5, 'Efficiency ($\\phi$)', ha='center',
    va='center', rotation='vertical')
fig.text(0.5, 0.04, 'Link Density ($\\rho$)', ha='center', va='center')
fig.text(0.78, 0.7, 'Survival time', ha = 'center', va = "center", rotation = "vertical")
fig.text(0.78, 0.3, 'Energy production', ha = 'center', va = "center", rotation = "vertical")
ax1.annotate("A", xy=(0.03, 0.9), xycoords="axes fraction", color = "white")
ax2.annotate("B", xy=(0.03, 0.9), xycoords="axes fraction", color = "white")
ax3.annotate("C", xy=(0.03, 0.9), xycoords="axes fraction", color = "black")
ax4.annotate("D", xy=(0.03, 0.9), xycoords="axes fraction", color = "black")
ax5.annotate("E", xy=(0.03, 0.9), xycoords="axes fraction", color = "black")
ax6.annotate("F", xy=(0.03, 0.9), xycoords="axes fraction", color = "black")


plt.savefig("../../results/model/"+folder+"/parscan_st_te.png")
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
