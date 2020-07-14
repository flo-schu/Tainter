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
# data = np.load("./cluster/parameter_analysis_20200630/output_interpol.npy", allow_pickle=True)
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
darr = dat.reshape((len(npe),len(nrho),len(nphi),6))

# lower subplots


# PLOT #########################################################################

# parameters
plot_pe = npe.searchsorted([0, 2e-4, 0.02])  # indices of tested pe values
labels = {"st": ("A1", "A2", "A3"),
          "te": ("B1", "B2", "B3")
          }

# color ranges
c_st = darr[plot_pe, :, :, colnames == "st"].flatten()
c_te = darr[plot_pe, :, :, colnames == "te"].flatten()

fig = plt.figure(constrained_layout=True, figsize=(20, 12))
gs = fig.add_gridspec(2, 6)

# survival time
fu0 = fig.add_subplot(gs[0, 0])
fu1 = fig.add_subplot(gs[0, 1])
fu2 = fig.add_subplot(gs[0, 2])

for pe, ax, lab in zip(plot_pe, [fu0, fu1, fu2], labels["st"]):
    grid = np.flipud(darr[pe, :, :, np.where(colnames == "st")[0][0]].T)
    im = ax.imshow(grid, extent=(min(nrho), max(nrho), min(nphi), max(nphi)),
                   vmin=c_st.min(), vmax=c_st.max())

    if lab != "A1": ax.yaxis.set_ticklabels([])
    if lab == "A1": ax.set_ylabel("output elasticity ($\\phi$)")
    if lab == "A2": ax.text(x=.15, y=.94, s="link density ($\\rho$)", ha="center")
    if lab == "A3":
        axins = inset_axes(ax,
                             width="6%",  # width = 50% of parent_bbox width
                             height="40%",  # height : 5%
                             loc='upper right')
        fu3cb = fig.colorbar(im, cax=axins, ticklocation="left")
        fu3cb.set_label('survival time')

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

    contour_rho = np.flipud(darr[pe, :, :, np.where(colnames == "rho")[0][0]].T)
    contour_phi = np.flipud(darr[pe, :, :, np.where(colnames == "phi")[0][0]].T)
    contour_diff = np.flipud(darr[pe, :, :, np.where(colnames == "tediff")[0][0]].T)
    im = ax.imshow(grid, extent=(min(nrho), max(nrho), min(nphi), max(nphi)),
                   vmin=c_te.min(), vmax=c_te.max())
    ax.contour(contour_rho, contour_phi, contour_diff,
               levels=np.array([0]), linestyles="--", colors="white")
    if lab != "B1": ax.yaxis.set_ticklabels([])
    if lab == "B1": ax.set_ylabel("output elasticity ($\\phi$)")
    if lab == "B2": ax.text(x=.15, y=.94, s="link density ($\\rho$)", ha="center")
    if lab == "B3":
        axins = inset_axes(ax,
                             width="6%",  # width = 50% of parent_bbox width
                             height="40%",  # height : 5%
                             loc='upper right')
        fu3cb = fig.colorbar(im, cax=axins, ticklocation="left")
        fu3cb.set_label('output (E)', color='white')
        fu3cb.ax.yaxis.set_tick_params(color='white')
        fu3cb.outline.set_edgecolor('white')
        plt.setp(plt.getp(fu3cb.ax.axes, 'yticklabels'), color='white')    # plot annotations

    textstr = ' - '.join((lab, r'$p_e=%.5f$' % (npe[pe],)))
    ax.text(.0, 1.05, textstr, transform=ax.transAxes, fontsize=12,
            ha="left", va="top")

# exploration (lower left subplot)
fm1 = fig.add_subplot(gs[1, 0:3])
fm1.cla()
plot_phi = np.arange(len(nphi))
pe_frame = npe <= 0.02

# fm1.set_xlim(0, 0.0027)
fm1.set_xlim(9e-6, 0.02)
fm1.set_ylim(0, 10100)
fm1.set_xscale('log')
fm1.set_ylabel("survival time")
fm1.set_xlabel("exploration ($p_e$)")
fm1.text(.0, 1.05, "A4", transform=fm1.transAxes, fontsize=12,
         ha="left", va="top")

lc = list()
for iphi in plot_phi:
    temp = darr[pe_frame, :, iphi, np.where(colnames == "st")[0][0]]
    lc.append(np.column_stack([npe[pe_frame], temp.mean(axis=1)]))

line_segments = LineCollection(lc)
line_segments.set_array(nphi)
fm1.add_collection(line_segments)

axins1 = inset_axes(fm1,
                    width="2%",  # width = 50% of parent_bbox width
                    height="40%",  # height : 5%
                    loc='lower right')
fm1cb = fig.colorbar(line_segments, cax=axins1, ticklocation="left")
fm1cb.set_label('output elasticity ($\\phi$)')


# total energy over p_e
fm2 = fig.add_subplot(gs[1, 3:6])
fm2.cla()
plot_phi = np.arange(len(nphi))
pe_frame = npe < 0.2

fm2.set_xlim(9e-6, 0.02)
fm2.set_ylim(1e2, 2e5)
fm2.set_ylabel("output (E)")
fm2.set_xlabel("exploration ($p_e$)")
fm2.set_xscale('log')
fm2.set_yscale('log')
fm2.text(.0, 1.05, "B4", transform=fm2.transAxes, fontsize=12,
         ha="left", va="top")

lc = list()
for iphi in plot_phi:
    temp = darr[pe_frame, :, iphi, np.where(colnames == "te")[0][0]]
    lc.append(np.column_stack([npe[pe_frame], temp.mean(axis=1)]))

axins2 = inset_axes(fm2,
                    width="2%",  # width = 50% of parent_bbox width
                    height="40%",  # height : 5%
                    loc='lower right')

line_segments = LineCollection(lc)
line_segments.set_array(nphi)
fm2.add_collection(line_segments)
fm2cb = fig.colorbar(line_segments, cax=axins2, ticklocation="left")
fm2cb.set_label('output elasticity ($\\phi$)')

plt.savefig('pub_figure4.pdf', dpi=200)

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
