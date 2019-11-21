import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import sys
sys.path.append('../helpers/')
from shifted_cmap import shiftedColorMap


folder = "20191027_0101"
# folder = "20191112_0937"

data   = pd.read_csv("../../results/model/"+folder+"/parscan.csv")
# data   = pd.read_csv("../../results/model/"+folder+"/parscan_plus_p0.csv")

### FILTER DATA SET ###################################################
data   = data.query("rho < 1")
#
p_e = np.array(data.p_e)
rho = np.array(data.rho)
phi = np.array(data.phi)
npe = np.unique(p_e)
nrho = np.unique(rho)
nphi = np.unique(phi)

dat = np.array(data)[:,1:]
names = np.array(data.columns[1:])

te_0 = dat[p_e == 0, names == "te"]
print(te_0)
# replicate te of pe = 0 npe times and reshape to array of arrays with 1 element each
te_0_arr = np.reshape(np.array(te_0.tolist() * len(npe)),(len(data),1))
# add new arrays to existing array
dat = np.append(dat, te_0_arr, axis = 1)
names = np.append(names,"te0")

te_diff = dat[:,names == "te"] - dat[:, names == "te0"]
te_diff_arr = np.reshape(te_diff, (len(data),1))
dat = np.append(dat, te_diff_arr, axis = 1)
names = np.append(names,"tediff")


print(names)

input("abort? Press ctrl+C or enter to continue")



### COLORPLOTS #################################################################
# better colorbar
epmin = np.min(dat[:,names == "tediff"])
epmax = np.max(dat[:,names == "tediff"])
orig_cmap = mpl.cm.RdBu
midpoint = np.absolute(epmin) / (epmax - epmin)
shifted_cmap = shiftedColorMap(orig_cmap, midpoint=midpoint, name='shiftedcmap')

# pe on x axis
fig, axes = plt.subplots(nrows = 4, ncols = 5, sharex=True, sharey=True)

# select p range
# pe_range = np.arange(0,.02,.001)
pe_range = npe
for i, ax in zip(pe_range, axes.flatten()):
    my_sub = ( i - 0.00001 < p_e ) & ( p_e < i + 0.00001 )
    print(i, np.sum(my_sub))
    d = dat[my_sub,:]
    grid = d[:,names == "tediff"].reshape(len(nrho), len(nphi))
    # grid = d[:,5].reshape(len(rho.unique()), len(phi.unique()))
    grid = np.flipud(grid.T)

    im = ax.imshow(grid, extent=(min(nrho), max(nrho), min(nphi), max(nphi)),
        aspect = "auto", interpolation = "nearest", cmap = shifted_cmap,
        vmin = epmin, vmax = epmax)
    ax.text(x = max(nrho)*.7, y = max(nphi)*.92, s = str(np.round(np.log10(i),1)),
            fontsize = 8)


fig.subplots_adjust(right=0.75)
cbar_ax_ep = fig.add_axes([0.8, 0.2, 0.05, .6])
fig.colorbar(im, cax = cbar_ax_ep)

plt.savefig("../../results/model/"+folder+"/tediff_for_pe.jpg")
plt.show()

# pe on x axis
fig, axes = plt.subplots(nrows = 3, ncols = 3, sharex=True, sharey=True)
for i, ax in zip(np.arange(0,max(nrho),.1), axes.flatten()):
    my_sub = ( i - 0.00001 < rho ) & ( rho < i + 0.00001 )
    d = dat[my_sub,:]
    grid = d[:,names == "te"].reshape(len(npe), len(nphi))
    grid = np.flipud(grid.T)
    ax.imshow(grid, extent=(min(npe), max(npe), min(nphi), max(nphi)),
        aspect = "auto", interpolation = "nearest", cmap = shifted_cmap,
        vmin = epmin, vmax = epmax)


fig.subplots_adjust(right=0.75)
cbar_ax_ep = fig.add_axes([0.8, 0.2, 0.05, .6])
fig.colorbar(im, cax = cbar_ax_ep)

plt.show()


###
