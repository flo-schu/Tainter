# created by Florian Schunck in December 2019
# Project: tainter
# Short description of the feature:
# 1. create Plot 4 for publication. The plot displays the paramter grids in
#    terms of survival time and energy production and contrasts p_e=0 und p_e>0
#    to illustrate divergent model outcomes dependent on the degree of
#    exploration
# ------------------------------------------------------------------------------
# Open tasks:
# TODO: add lines for p_e values displayed in grids
#
# ------------------------------------------------------------------------------

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as st
import matplotlib as mpl
import pandas as pd
import sys
sys.path.append('../helpers/')
from shifted_cmap import shiftedColorMap

# data input and processing for upper subplots #################################


# folder = "20191027_0101" # model run for first paper draft (rho[0,1], p_E[low resolution
# folder = "20191112_0937"
# folder = "20200625_1350" # model for second paper draft (log scale for pe, link density (rho) only until .3)
folder = "20200625_1615"

data   = pd.read_csv("../../results/model/"+folder+"/parscan.csv")
# data   = pd.read_csv("../../results/model/"+folder+"/parscan_plus_p0.csv")
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
# print(te_0)
# replicate te of pe = 0 npe times and reshape to array of arrays with 1 element each
te_0_arr = np.reshape(np.array(te_0.tolist() * len(npe)),(len(data),1))
# add new arrays to existing array
dat = np.append(dat, te_0_arr, axis = 1)
names = np.append(names,"te0")

te_diff = dat[:,names == "te"] - dat[:, names == "te0"]
te_diff_arr = np.reshape(te_diff, (len(data),1))
dat = np.append(dat, te_diff_arr, axis = 1)
names = np.append(names,"tediff")



# data input and processing for lower subplots #################################

out_e  = list()
out_s  = list()
out_s_conf = list()
out_e_conf = list()
out_ed_conf = list()

for i in npe:
    sube  = dat[p_e == i, names == "te"]
    subs  = dat[p_e == i, names == "s"]
    subed = dat[p_e == i, names == "tediff"]
    out_e.append(np.mean(sube))
    out_s.append(np.mean(subs))
    out_s_conf.append(st.t.interval(0.99, df=len(subs)-1,
                                    loc = np.mean(subs), scale = st.sem(subs)))
    out_e_conf.append(st.t.interval(0.99, df=len(sube)-1,
                                    loc = np.mean(sube), scale = st.sem(sube)))


print(out_e_conf)

mysub = dat[:,names == "tediff"] < 0
comp = dat[mysub[:,0], : ]

size_badarea = list()
magn_badarea = list()

for i in npe:
    te = comp[comp[:,names == "p_e"][:,0] == i, names == "tediff"]
    size_badarea.append(len(te))
    magn_badarea.append(np.sum(te)/-10000)
    out_ed_conf.append(st.t.interval(0.99, df=len(te)-1,
                                     loc = np.sum(te)/-10000,
                                     scale = st.sem(te)))
    # avg_st.append(st*-1)


# PLOT #########################################################################

# better colorbar --------------------------------------------------------------
epmin = np.min(dat[:,names == "tediff"])
epmax = np.max(dat[:,names == "tediff"])
orig_cmap = mpl.cm.RdBu
midpoint = np.absolute(epmin) / (epmax - epmin)
shifted_cmap = shiftedColorMap(orig_cmap, midpoint=midpoint, name='shiftedcmap')


# initialize figure ------------------------------------------------------------
textwidth = 12.12537
plt.rcParams.update({'font.size': 14})
# cmap = cm.get_cmap("tab20",20)

f1 = plt.figure(constrained_layout=True, figsize = (textwidth, textwidth/2))
gs = f1.add_gridspec(2, 6)

# plot grids -------------------------------------------------------------------
f1_ax1 = f1.add_subplot(gs[0, 0])
grid = dat[p_e == p_e[np.searchsorted(p_e,0.00)],
           names == "s"].reshape(len(nrho), len(nphi))
grid = np.flipud(grid.T)
im = f1_ax1.imshow(grid, extent=(min(nrho), max(nrho), min(nphi), max(nphi)),
                   aspect = "auto", interpolation = "nearest", cmap = shifted_cmap,
                   vmin = epmin, vmax = epmax)
f1_ax1.set_ylabel("efficiency ($\\phi$)")
f1_ax1.set_xlabel(" ", horizontalalignment = "right")
f1.text(x = .2 , y= .52, s = "link density ($\\rho$)", ha = "center")

f1_ax2 = f1.add_subplot(gs[0, 1], sharey = f1_ax1)
grid = dat[p_e == p_e[np.searchsorted(p_e,0.02)],
           names == "s"].reshape(len(nrho), len(nphi))
grid = np.flipud(grid.T)
im = f1_ax2.imshow(grid, extent=(min(nrho), max(nrho), min(nphi), max(nphi)),
                   aspect = "auto", interpolation = "nearest", cmap = shifted_cmap,
                   vmin = epmin, vmax = epmax)


f1_ax3 = f1.add_subplot(gs[0, 2], sharey = f1_ax1)
grid = dat[p_e == p_e[np.searchsorted(p_e,0.00)],
           names == "te"].reshape(len(nrho), len(nphi))
grid = np.flipud(grid.T)
im = f1_ax3.imshow(grid, extent=(min(nrho), max(nrho), min(nphi), max(nphi)),
                   aspect = "auto", interpolation = "nearest", cmap = shifted_cmap,
                   vmin = epmin, vmax = epmax)
f1_ax3.set_ylabel("efficiency ($\\phi$)")
f1.text(x = .52 , y= .52, s = "link density ($\\rho$)", ha = "center")

f1_ax4 = f1.add_subplot(gs[0, 3], sharey = f1_ax1)
grid = dat[p_e == p_e[np.searchsorted(p_e,0.02)],
           names == "te"].reshape(len(nrho), len(nphi))
grid = np.flipud(grid.T)
im = f1_ax4.imshow(grid, extent=(min(nrho), max(nrho), min(nphi), max(nphi)),
                   aspect = "auto", interpolation = "nearest", cmap = shifted_cmap,
                   vmin = epmin, vmax = epmax)



f1_ax5 = f1.add_subplot(gs[0, 4], sharey = f1_ax1)
grid = dat[p_e == p_e[np.searchsorted(p_e,0.00275)],
names == "tediff"].reshape(len(nrho), len(nphi))
grid = np.flipud(grid.T)
im = f1_ax5.imshow(grid, extent=(min(nrho), max(nrho), min(nphi), max(nphi)),
                   aspect = "auto", interpolation = "nearest", cmap = shifted_cmap,
                   vmin = epmin, vmax = epmax)
f1_ax5.set_ylabel("efficiency ($\\phi$)")
f1.text(x = .85 , y= .52, s = "link density ($\\rho$)", ha = "center")
# f1_ax5.text(x = max(nrho)*.7, y = max(nphi)*.92,
#             s = str(np.round(np.log10(i),1)),
#             fontsize = 8)

f1_ax6 = f1.add_subplot(gs[0, 5], sharey = f1_ax1)
grid = dat[p_e   == p_e[np.searchsorted(p_e,0.02)],
           names == "tediff"].reshape(len(nrho), len(nphi))
grid = np.flipud(grid.T)
im = f1_ax6.imshow(grid, extent=(min(nrho), max(nrho), min(nphi), max(nphi)),
                   aspect = "auto", interpolation = "nearest", cmap = shifted_cmap,
                   vmin = epmin, vmax = epmax)
# f1_ax6.text(x = max(nrho)*.7, y = max(nphi)*.92,
#             s = str(np.round(np.log10(i),1)),
#             fontsize = 8)

# plot colorbar ----------------------------------------------------------------
# fig.subplots_adjust(right=0.75)
# cbar_ax_ep = fig.add_axes([0.8, 0.2, 0.05, .6])
# fig.colorbar(im, cax = cbar_ax_ep)


# plot st and te over p_e ------------------------------------------------------
f1_ax7 = f1.add_subplot(gs[1, 0:2])
f1_ax7.plot(npe, out_s, label = "survival")
f1_ax7.fill_between(npe, [x[0] for x in out_s_conf], [x[1] for x in out_s_conf],
                    alpha = .5)
f1_ax7.set_xlabel("$p_e$")
# f1_ax7.set_ylim(np.min(out_s)*0.9, np.max(out_s)*1.1)
# f1.text(x = .48, y = 0.01, s= "exploration probability ($p_e$)", ha = "center")
# f1_ax7.set_ylabel("mean survival time")
f1_ax7.legend(frameon = False)

f1_ax8 = f1.add_subplot(gs[1, 2:4])
f1_ax8.plot(npe, out_e, label = "energy")
f1_ax8.fill_between(npe, [x[0] for x in out_e_conf], [x[1] for x in out_e_conf],
                    alpha = .5)
f1_ax8.set_xlabel("$p_e$")
# f1_ax8.set_ylabel("mean energy production  ")
f1_ax8.legend(frameon = False)

f1_ax9 = f1.add_subplot(gs[1,4:6])
f1_ax9.plot(npe,magn_badarea, label = "magnitude of redarea")
f1_ax9.fill_between(npe,
                    [x[0] for x in out_ed_conf],
                    [x[1] for x in out_ed_conf],
                    alpha = .5)
f1_ax9.set_xlabel("$p_e$")
f1_ax9.legend(frameon = False)

# remove ticklabels from y axis in gridplots
plt.setp(f1_ax2.get_yticklabels(), visible=False)
plt.setp(f1_ax4.get_yticklabels(), visible=False)
plt.setp(f1_ax6.get_yticklabels(), visible=False)
# plt.yscale('log')
# plt.xscale('log')
plt.savefig("../../results/model/"+folder+"/pub_figure4.jpg")
plt.show()
