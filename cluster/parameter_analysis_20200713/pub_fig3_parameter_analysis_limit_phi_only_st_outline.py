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
from matplotlib import cm
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# data input and processing for upper subplots ---------------------------------
print("Hello! Starting import...")
# data = np.load("./cluster/parameter_analysis_20200713/output_interpol.npy", allow_pickle=True)
data = np.load("output_interpol.npy", allow_pickle=True)
colnames = np.array(["p_e", "rho", "phi", "te", "st"])
print("Import complete")

p_e = data[:, 0]
rho = data[:, 1]
phi = data[:, 2]
npe = np.unique(p_e)
nrho = np.unique(rho)
nphi = np.unique(phi)

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

# reshape data to array of 4 dimensions (one for each parameter and one for the
# data entries.
# access like:
# a) darr[p_e][rho][phi][colnames]  entries in brackets are replaced by integers
# b) darr[p_e, rho, phi, colnames]  to select all choose :
darr = dat.reshape((len(npe), len(nrho), len(nphi), 6))

# lower subplots
# filter darr
lim_phi = 1.3
nphi = nphi[nphi < lim_phi]

darr = darr[:, :, range(len(nphi)), :]

# PLOT #########################################################################

# parameters
plot_pe = npe.searchsorted([0, 1e-4, 2.4e-3])  # indices of tested pe values
labels = {"st": ("A1", "A2", "A3"),
          "te": ("B1", "B2", "B3")
          }

# color ranges
c_st = darr[plot_pe, :, :, colnames == "st"].flatten()
c_te = darr[plot_pe, :, :, colnames == "te"].flatten()
c_te_lim = np.where(c_st < 10000, c_te, 0)

cmap = cm.get_cmap('plasma')
cmap.set_over("white")

fig = plt.figure(constrained_layout=False, figsize=(12, 8))
gs = fig.add_gridspec(2, 3)
plt.draw()


# survival time
fu0 = fig.add_subplot(gs[0, 0])
fu1 = fig.add_subplot(gs[0, 1])
fu2 = fig.add_subplot(gs[0, 2])

for pe, ax, lab in zip(plot_pe, [fu0, fu1, fu2], labels["st"]):
    grid = np.flipud(darr[pe, :, :, np.where(colnames == "st")[0][0]].T)
    contour_rho = np.flipud(darr[pe, :, :, np.where(colnames == "rho")[0][0]].T)
    contour_phi = np.flipud(darr[pe, :, :, np.where(colnames == "phi")[0][0]].T)
    contour_stlim = np.flipud(darr[pe, :, :, np.where(colnames == "st")[0][0]].T)
    im = ax.imshow(grid, extent=(min(nrho), max(nrho), min(nphi), max(nphi)),
                   vmin=c_st.min(), vmax=c_st.max() - 1, cmap=cmap)
    ax.contour(contour_rho, contour_phi, contour_stlim,
               levels=np.array([9999]), linestyles="-", colors="black")
    if lab != "A1": ax.yaxis.set_ticklabels([])
    if lab == "A1": ax.set_ylabel("output elasticity ($\\phi$)")
    if lab == "A2": ax.text(x=.15, y=.94, s="link density ($\\rho$)", ha="center")
    if lab == "A1":
        axins = inset_axes(ax,
                           width="6%",  # width = 50% of parent_bbox width
                           height="40%",  # height : 5%
                           loc='upper right')
        fu3cb = fig.colorbar(im, cax=axins, ticklocation="left", extend="max")
        fu3cb.set_label('survival time')

    # plot annotations
    textstr = ' - '.join((lab, r'$p_e=%.5f$' % (npe[pe],)))
    ax.text(.0, .95, textstr, transform=ax.transAxes, fontsize=12,
            ha="left", va="top")

# exploration (lower left subplot)
fm1 = fig.add_subplot(gs[1, 0:3])
fm1.cla()

fm1.set_xlim(9e-6, 0.003)
fm1.set_ylim(0, 10500)
fm1.set_yticks([0, 2000, 4000, 6000, 8000, 10000])
fm1.set_yticklabels(['0', '2000', '4000', '6000', '8000', '$\\geq$ 10000'])
fm1.set_xscale('log')

darr_contour = np.flipud(np.median(darr[:, :, :, :], axis=1))
contour_pe = darr_contour[  :, :, np.where(colnames == "p_e")[0][0]]
contour_st = darr_contour[  :, :, np.where(colnames == "st")[0][0]]
contour_phi = darr_contour[  :, :, np.where(colnames == "phi")[0][0]]

levels = [1, 1.05, 1.075, 1.1, 1.15, 1.18, 1.19, 1.195, 1.2, 1.25]

CS = fm1.contour(contour_pe, contour_st, contour_phi, levels=levels, linestyles="-", colors="black")
manual_locations = np.asarray(plt.ginput(len(levels), timeout=-1))

fm1.clabel(CS, inline=1, fontsize=10, manual=manual_locations)
fm1.set_ylabel("median survival time (t)")
fm1.set_xlabel("exploration ($p_e$)")
fm1.text(.0, .95, "B", transform=fm1.transAxes, fontsize=12,
         ha="left", va="top")


plt.savefig('pub_figure4_limit_phi_only_st_contour.pdf', dpi=200)
plt.show()
# fl1 = fig.add_subplot(gs[2, 0:3])
# fl1.cla()
# pe_frame = npe < 0.0026
# plot_rho = np.arange(len(nrho))
# plot_rho_col = cm.viridis(np.linspace(0,1,len(nrho)))
# for irho, icol in zip(plot_rho,plot_rho_col):
#     temp = darr[pe_frame, irho, :, np.where(colnames == "st")[0][0]]
#     y = temp.mean(axis=1)
#     fl1.plot(npe[pe_frame], y, color=icol)
#     fl1.set_ylabel("survival time")
#     fl1.set_xlabel("exploration ($p_e$)")
#
#
# fl2 = fig.add_subplot(gs[2, 3:6])
# fl2.cla()
# plot_rho = np.arange(len(nrho))
# pe_frame = npe < 0.2
# plot_rho_col = cm.viridis(np.linspace(0,1,len(nrho)))
# for irho, icol in zip(plot_rho, plot_rho_col):
#     temp = darr[pe_frame, irho, :, np.where(colnames == "te")[0][0]]
#     y = temp.mean(axis=1)
#     fl2.plot(npe[pe_frame], y, color=icol)
#     fl2.set_ylabel("produced energy")
#     fl2.set_xlabel("exploration ($p_e$)")
#     fl2.set_xscale('log')
#     fl2.set_yscale('log')
