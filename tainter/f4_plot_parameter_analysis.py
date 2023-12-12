# created by Florian Schunck in December 2019
# Project: tainter
# Short description of the feature:
# 1. create Plot 4 for publication. The plot displays the paramter grids in
#    terms of survival time and energy production and contrasts par_1=0 und par_1>0
#    to illustrate divergent model outcomes dependent on the degree of
#    exploration
# 2. data is transformed to 4D array which is then subset to get a 2D array
#    of one outcome variable (st, te, tediff) which is reshaped into
#    par_2 * par_3 (1000 x 100). The resulting output is transposed and flipped
#    upside down
# ------------------------------------------------------------------------------
# Open tasks:
# TODO: add lines for par_1 values displayed in grids
#
# ------------------------------------------------------------------------------

import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
from labellines import labelLines
from tainter.cluster.parameter_scan_odeint import process_output

# data input and processing for upper subplots ---------------------------------

def fig4_parameter_analysis(
    data_file="data/parameter_analysis/results_combined.txt",
    parameters=["\rho", "p_e", "c"],
    panel_steps=[0, 1e-4, 5e-4, 5e-3],
    multiline_steps=[],
):
    data = np.loadtxt(data_file)
    colnames = np.array(["par_1","par_2", "par_3", "te", "st"])
    print("Import complete")

    par_1 = data[:, 0]
    par_2 = data[:, 1]
    par_3 = data[:, 2]

    npar_1 = np.unique(par_1)
    npar_2 = np.unique(par_2)
    npar_3 = np.unique(par_3)

    if len(multiline_steps) == 0:
        multiline_steps = np.linspace(
            npar_3.min(), npar_3.max(), num=10
        )


    print(len(data), len(npar_1), len(npar_2), len(npar_3))

    # recycle pe=0 over all unique pe values and add column to the end of the array
    te_0 = np.tile(data[par_1 == 0, colnames == "te"], len(npar_1))
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
    # a) darr[par_1][par_2][par_3][colnames]  entries in brackets are replaced by integers
    # b) darr[par_1, par_2, par_3, colnames]  to select all choose :
    darr = dat.reshape((len(npar_1), len(npar_2), len(npar_3), 6))

    # lower subplots
    # filter darr
    # npar_3 = npar_3[npar_3 < lim_phi]

    darr = darr[:, :, range(len(npar_3)), :]

    # PLOT #########################################################################

    # parameters
    plot_pe = npar_1.searchsorted(panel_steps)  # indices of tested pe values
    labels = {"st": ("$B_1$", "$B_2$", "$B_3$", "$B_4$")}

    # color ranges
    c_st = darr[plot_pe, :, :, colnames == "st"].flatten()

    cmap = cm.get_cmap('plasma')
    cmap.set_over("grey")

    textwidth = 12.12537
    fig = plt.figure(constrained_layout=False, figsize=(textwidth, textwidth / 2), frameon=True)
    plt.rcParams.update({'font.size': 14})

    # fig.set_tight_layout(tight=True)
    gs = fig.add_gridspec(2, 4)
    plt.draw()

    # survival time
    fu0 = fig.add_subplot(gs[1, 0])
    fu1 = fig.add_subplot(gs[1, 1])
    fu2 = fig.add_subplot(gs[1, 2])
    fu3 = fig.add_subplot(gs[1, 3])
    fm1 = fig.add_subplot(gs[0, 0:4])

    for pe, ax, lab in zip(plot_pe, [fu0, fu1, fu2, fu3], labels["st"]):
        ax.cla()
        grid = np.flipud(darr[pe, :, :, np.where(colnames == "st")[0][0]].T)
        contour_rho = np.flipud(darr[pe, :, :, np.where(colnames == "par_2")[0][0]].T)
        contour_phi = np.flipud(darr[pe, :, :, np.where(colnames == "par_3")[0][0]].T)
        contour_stlim = np.flipud(darr[pe, :, :, np.where(colnames == "st")[0][0]].T)
        im = ax.imshow(grid, extent=(min(npar_2), max(npar_2), min(npar_3), max(npar_3)),
                    vmin=0, vmax=9999, cmap=cmap,
                    aspect="auto")
        ax.contour(contour_rho, contour_phi, contour_stlim,
                levels=np.array([9999]), linestyles="-", colors="black")
        if lab != "$B_1$": ax.yaxis.set_ticklabels([])
        if lab == "$B_1$": ax.set_ylabel(f"productivity ($c$)")
        if lab == "$B_4$":
            axins = inset_axes(ax,
                            width="8%",  # width = 50% of parent_bbox width
                            height="60%",  # height : 5%
                            loc='lower right')
            fu3cb = fig.colorbar(im, cax=axins, ticklocation="left", extend="max")
            fu3cb.set_label('survival time')
            fu3cb.set_ticks([0, 5000, 10000])
            fu3cb.set_ticklabels(['0', '5000', '$\\geq 10000$'])
        # plot annotations
        # pe_lab = r'$par_1=%.5f$' % (npar_1[pe],)
        pe_lab = ""
        textstr = ''.join((lab, pe_lab))
        ax.text(.03, .97, textstr, transform=ax.transAxes, fontsize=14,
                ha="left", va="top")

    # exploration (lower left subplot)
    fm1.cla()

    fm1.set_xlim(9e-6, 0.003)
    fm1.set_ylim(0, 10000)
    fm1.set_yticks([0, 2000, 4000, 6000, 8000, 10000])
    fm1.set_yticklabels(['0', '2000', '4000', '6000', '8000', '$\\geq 10000$'])
    fm1.set_xscale('log')
    fm1.set_xticks([1e-5, 1e-4, 1e-3, 2.5e-3])
    fm1.set_xticklabels(["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "${}^1{/}_N$"])
    fm1.set_ylabel("median survival time (t)")
    fm1.set_xlabel("exploration ($p_e$)")
    fm1.text(.0075, .94, "A", transform=fm1.transAxes, fontsize=14,
            ha="left", va="top")
    fm1.axvline(x=2.5e-3, ymin=0, ymax=1, color="grey", linestyle="--")
    # fm1.axvline(x=1e-4, ymin=0, ymax=1, color="grey", linestyle="--")
    # fm1.axvline(x=5e-4, ymin=0, ymax=1, color="grey", linestyle="--")
    # fm1.axvline(x=3e-3, ymin=0, ymax=1, color="grey", linestyle="--")

    # pe and par_3 arrays are the same for any par_2 value. Hence any can be chosen in
    darr_median = np.median(darr, axis=1)

    levels = npar_3.searchsorted(multiline_steps)
    for lvl in levels:
        lbl = f"${parameters[2]} = " + str(np.round(npar_3[lvl], 2)) + "$"
        st = darr_median[:, lvl, 4]
        st = np.where(st >= 10000, 1e100, st)
        pe = darr_median[:, lvl, 0]
        fm1.plot(pe, st, color="black", linewidth=.75, label=lbl)

    # x_locations = np.asarray(plt.ginput(len(levels), timeout=-1))[:,0]
    x_locations = np.array([1.5e-05, 2.5e-05, 5.0e-05, 7.0e-05, 1.1e-04,
                            1.5e-04, 2.0e-04, 2.75e-04, 3.5e-4])
    labelLines(fm1.get_lines(), xvals=x_locations, zorder=2.5, fontsize=10,
            bbox={'pad': .5, 'color': 'white'})

    # connections
    pe_ax = [0, 0.415, 0.694, 1]
    for pe, ax in zip(pe_ax, [fu0, fu1, fu2, fu3]):
        for side in [0, 1]:
            cp = ConnectionPatch((pe, 0), (side, 1), "axes fraction", "axes fraction", axesA=fm1, axesB=ax, color="grey",
                                linestyle="--", alpha=.5)
            fm1.add_artist(cp)

    fig.text(x=.55, y=.035, s="link density ($\\rho$)", ha="center")

    plt.subplots_adjust(0.12, 0.12, 0.98, 0.98, 0.13, 0.31)

    return fig
    # fl1 = fig.add_subplot(gs[2, 0:3])
    # fl1.cla()
    # pe_frame = npar_1 < 0.0026
    # plot_rho = np.arange(len(npar_2))
    # plot_rho_col = cm.viridis(np.linspace(0,1,len(npar_2)))
    # for irho, icol in zip(plot_rho,plot_rho_col):
    #     temp = darr[pe_frame, irho, :, np.where(colnames == "st")[0][0]]
    #     y = temp.mean(axis=1)
    #     fl1.plot(npar_1[pe_frame], y, color=icol)
    #     fl1.set_ylabel("survival time")
    #     fl1.set_xlabel("exploration ($par_1$)")
    #
    #
    # fl2 = fig.add_subplot(gs[2, 3:6])
    # fl2.cla()
    # plot_rho = np.arange(len(npar_2))
    # pe_frame = npar_1 < 0.2
    # plot_rho_col = cm.viridis(np.linspace(0,1,len(npar_2)))
    # for irho, icol in zip(plot_rho, plot_rho_col):
    #     temp = darr[pe_frame, irho, :, np.where(colnames == "te")[0][0]]
    #     y = temp.mean(axis=1)
    #     fl2.plot(npar_1[pe_frame], y, color=icol)
    #     fl2.set_ylabel("produced energy")
    #     fl2.set_xlabel("exploration ($par_1$)")
    #     fl2.set_xscale('log')
    #     fl2.set_yscale('log')

if __name__ == "__main__":
    data_file = sys.argv[1]
    fig4_parameter_analysis(data_file=data_file)