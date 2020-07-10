# created by Florian Schunck in December 2019
# Project: tainter
# Short description of the feature:
# 1. create Plot 4 for publication. The plot displays the paramter grids in
#    terms of survival time and energy production and contrasts p_e=0 und p_e>0
#    to illustrate divergent model outcomes dependent on the degree of
#    exploration
# 2. data is transformed to 4D array which is then subset to get a 2D array
#    of one outcome variable (st, te, tediff) which is reshaped into
#    rho * phi (1000 x 100). The resulting output is transposed and flipped
#    upside down
# ------------------------------------------------------------------------------
# Open tasks:
# TODO: add lines for p_e values displayed in grids
#
# ------------------------------------------------------------------------------

import numpy as np
from matplotlib import pyplot as plt


# data input and processing for upper subplots ---------------------------------
print("Hello! Starting import...")
# data = np.loadtxt("./cluster/parameter_analysis_20200630/output_corrected.txt", delimiter=",", skiprows=1)
data = np.load("./cluster/parameter_analysis_20200630/output.pkl", allow_pickle=True)
colnames = np.array(["p_e", "rho", "phi", "te", "st"])
print("Import complete")

p_e = data[:, 0]
rho = data[:, 1]
phi = data[:, 2]
npe = np.unique(p_e)
nrho = np.unique(rho)
nphi = np.unique(phi)

print("import complete")
print(len(data), len(npe), len(nrho), len(nphi))


# recycle pe=0 over all unique pe values and add column to the end of the array
te_0 = np.tile(data[p_e == 0, colnames == "te"], len(npe))
dat = np.column_stack((data, te_0))  # append a 1d array to a 2d array (via column)
colnames = np.append(colnames, "te0")

# calculate difference of pe0 and peX and add column to the end of the array
te_diff = (dat[:, colnames == "te"] - dat[:, colnames == "te0"]).squeeze()
dat = np.column_stack((dat, te_diff))
colnames = np.append(colnames, "tediff")

# remove te0 column
dat = dat[:, colnames != "te0"]
colnames = colnames[colnames != "te0"]
print(colnames, '\n', dat)

# reshape data to array of 4 dimensions (one for each parameter and one for the
# data entries.
# access like:
# a) darr[p_e][rho][phi][colnames]  entries in brackets are replaced by integers
# b) darr[p_e, rho, phi, colnames]  to select all choose :
darr = dat.reshape((len(npe),len(nrho),len(nphi),6))

# PLOT #########################################################################

# parameters
plot_pe = npe.searchsorted([0, 1e-4, 0.02])  # indices of tested pe values
labels = {"st": ("A", "B", "C"),
          "te": ("D", "E", "F")
          }

# color ranges
c_st = darr[plot_pe, :, :, colnames == "st"].flatten()
c_te = darr[plot_pe, :, :, colnames == "te"].flatten()

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(2, 6)

# survival time
fu0 = fig.add_subplot(gs[0, 0])
fu1 = fig.add_subplot(gs[0, 1])
fu2 = fig.add_subplot(gs[0, 2])

for pe, ax, lab in zip(plot_pe, [fu0, fu1, fu2], labels["st"]):
    grid = np.flipud(darr[pe, :, :, np.where(colnames == "st")[0][0]].T)
    im = ax.imshow(grid, extent=(min(nrho), max(nrho), min(nphi), max(nphi)),
                   vmin=c_st.min(), vmax=c_st.max())
    # plot annotations
    textstr = ' - '.join((lab, r'$p_e=%.5f$' % (npe[pe],)))
    ax.text(.0, 1.05, textstr, transform=ax.transAxes, fontsize=12,
            ha="left", va="top")


# energy
fu3 = fig.add_subplot(gs[0, 3])
fu4 = fig.add_subplot(gs[0, 4])
fu5 = fig.add_subplot(gs[0, 5])

for pe, ax, lab in zip(plot_pe, [fu3, fu4, fu5], labels["te"]):
    grid = np.flipud(darr[pe, :, :, np.where(colnames == "te")[0][0]].T)
    im = ax.imshow(grid, extent=(min(nrho), max(nrho), min(nphi), max(nphi)),
                   vmin=c_te.min(), vmax=c_te.max())
    # plot annotations
    textstr = ' - '.join((lab, r'$p_e=%.5f$' % (npe[pe],)))
    ax.text(.0, 1.05, textstr, transform=ax.transAxes, fontsize=12,
            ha="left", va="top")
