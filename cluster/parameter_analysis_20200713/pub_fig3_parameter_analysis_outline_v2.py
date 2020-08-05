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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

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
plot_pe = npe.searchsorted([0, 1e-4, 5e-4, 5e-3])  # indices of tested pe values
labels = {"st": ("$B_1$", "$B_2$", "$B_3$", "$B_4$")}

# color ranges
c_st = darr[plot_pe, :, :, colnames == "st"].flatten()

cmap = cm.get_cmap('plasma')
cmap.set_over("grey")

fig = plt.figure(constrained_layout=False, figsize=(16, 8))
gs = fig.add_gridspec(2, 4)
plt.draw()

# survival time
fu0 = fig.add_subplot(gs[1, 0])
fu1 = fig.add_subplot(gs[1, 1])
fu2 = fig.add_subplot(gs[1, 2])
fu3 = fig.add_subplot(gs[1, 3])

for pe, ax, lab in zip(plot_pe, [fu0, fu1, fu2, fu3], labels["st"]):
    grid = np.flipud(darr[pe, :, :, np.where(colnames == "st")[0][0]].T)
    contour_rho = np.flipud(darr[pe, :, :, np.where(colnames == "rho")[0][0]].T)
    contour_phi = np.flipud(darr[pe, :, :, np.where(colnames == "phi")[0][0]].T)
    contour_stlim = np.flipud(darr[pe, :, :, np.where(colnames == "st")[0][0]].T)
    im = ax.imshow(grid, extent=(min(nrho), max(nrho), min(nphi), max(nphi)),
                   vmin=0, vmax=c_st.max() - 1, cmap=cmap)
    ax.contour(contour_rho, contour_phi, contour_stlim,
               levels=np.array([9999]), linestyles="-", colors="black")
    if lab != "$B_1$": ax.yaxis.set_ticklabels([])
    if lab == "$B_1$": ax.set_ylabel("output elasticity ($\\phi$)")
    if lab == "$B_2$": ax.text(x=.325, y=.95, s="link density ($\\rho$)", ha="center")
    if lab == "$B_4$":
        axins = inset_axes(ax,
                           width="6%",  # width = 50% of parent_bbox width
                           height="40%",  # height : 5%
                           loc='lower right')
        fu3cb = fig.colorbar(im, cax=axins, ticklocation="left", extend="max")
        fu3cb.set_label('survival time')
        fu3cb.set_ticks([0, 5000, 10000])
        fu3cb.set_ticklabels(['0', '5000', '$\\geq 10000$'])

    # plot annotations
    # pe_lab = r'$p_e=%.5f$' % (npe[pe],)
    pe_lab = ""
    textstr = ''.join((lab, pe_lab))
    ax.text(.03, .97, textstr, transform=ax.transAxes, fontsize=14,
            ha="left", va="top")

# exploration (lower left subplot)
fm1 = fig.add_subplot(gs[0, 0:4])
fm1.cla()

fm1.set_xlim(9e-6, 0.003)
fm1.set_ylim(0, 10500)
fm1.set_yticks([0, 2000, 4000, 6000, 8000, 10000])
fm1.set_yticklabels(['0', '2000', '4000', '6000', '8000', '$\\geq$ 10000'])
fm1.set_xscale('log')
fm1.set_ylabel("median survival time (t)")
fm1.set_xlabel("exploration ($p_e$)")
fm1.text(.0075, .94, "A", transform=fm1.transAxes, fontsize=14,
         ha="left", va="top")
# fm1.axvline(x=0, ymin=0, ymax=1, color="grey", linestyle="--")
# fm1.axvline(x=1e-4, ymin=0, ymax=1, color="grey", linestyle="--")
# fm1.axvline(x=5e-4, ymin=0, ymax=1, color="grey", linestyle="--")
# fm1.axvline(x=3e-3, ymin=0, ymax=1, color="grey", linestyle="--")

# pe and phi arrays are the same for any rho value. Hence any can be chosen in
# this case [0]
contour_pe = darr[:, 0, :, np.where(colnames == "p_e")[0][0]]
contour_phi = darr[:, 0, :, np.where(colnames == "phi")[0][0]]
# median is computed for st along the rho axis
contour_st = np.median(darr[:, :, :, np.where(colnames == "st")[0][0]], axis=1)

levels = [1, 1.05, 1.075, 1.1, 1.15, 1.18, 1.19, 1.195, 1.2, 1.25]

CS = fm1.contour(contour_pe, contour_st, contour_phi, levels=levels, linestyles="-", colors="black")
# manual_locations = np.asarray(plt.ginput(len(levels), timeout=-1))
manual_locations = np.array([
    [1.37200179e-05, 4.12500000e+02],
    [3.00000000e-05, 1.38750000e+03],
    [6.17240961e-05, 3.00000000e+03],
    [1.30473382e-04, 3.80000000e+03],
    [1.70591944e-04, 5.40000000e+03],
    [2.35691822e-04, 6.63750000e+03],
    [2.82939199e-04, 7.61250000e+03],
    [3.21089992e-04, 8.55000000e+03],
    [3.47707952e-04, 9.67500000e+03],
    [4.19370434e-04, 1.00875000e+04]
])
fm1.clabel(CS, inline=1, fontsize=10, manual=manual_locations)

# connections
pe_ax = [0, 0.415, 0.694, 1]
for pe, ax in zip(pe_ax, [fu0, fu1, fu2, fu3]):
    for side in [0, 1]:
        cp = ConnectionPatch((pe, 0), (side, 1), "axes fraction", "axes fraction", axesA=fm1, axesB=ax, color="grey",
                             linestyle="--")
        fm1.add_artist(cp)

plt.savefig('pub_figure4_contour_v2.pdf', dpi=200)
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
