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

# data input and processing for upper subplots ---------------------------------
print("Hello! Starting import...")
data = np.load("./cluster/parameter_analysis_20200713/output_interpol.npy", allow_pickle=True)
# data = np.load("output_interpol.npy", allow_pickle=True)
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

# PLOT #########################################################################

# parameters
plot_pe = npe.searchsorted([0, 1e-4, 5e-4, 5e-3])  # indices of tested pe values
labels = {"st": ("$B_1$", "$B_2$", "$B_3$", "$B_4$")}

# color ranges
c_st = darr[plot_pe, :, :, colnames == "st"].flatten()

cmap = cm.get_cmap('plasma')
cmap.set_over("grey")

textwidth = 12.12537
fig = plt.figure(figsize=(textwidth, textwidth / 2))
plt.rcParams.update({'font.size': 14})

h1 = darr[npe.searchsorted(8e-4), :, nphi.searchsorted(1.19), 4].flatten()
l1 = "median = " + str(np.median(h1)) + "; $p_e = 8e^{-4}$" + "; $\\phi = 1.19$"

h2 = darr[npe.searchsorted(1e-3), :, nphi.searchsorted(1.19), 4].flatten()
l2 = "median = " + str(np.median(h2)) + "; $p_e = 1e^{-3}$" + "; $\\phi = 1.19$"

fig = plt.figure(figsize=(textwidth, textwidth / 2))
plt.cla()
plt.hist(h1, bins=20, label=l1)
plt.hist(h2, bins=20, label=l2)
plt.legend()

myphi = 1.19
med = np.median(darr[:, :, nphi.searchsorted(myphi), 4], axis=1)
avg = np.mean(darr[:, :, nphi.searchsorted(myphi), 4], axis=1)

fig = plt.figure(figsize=(textwidth, textwidth / 2))
for i in range(1000):
    bif = darr[:, i, nphi.searchsorted(myphi), 4]
    plt.plot(npe, bif, color="black")
    plt.plot(npe, med, color="red")
    plt.plot(npe, avg, color="blue")

plt.xscale('log')
plt.xlim(0, 3e-3)

# rho

fig = plt.figure(figsize=(textwidth, textwidth / 2))
fig.add_subplot(311)

sub = darr[:, :, nphi.searchsorted(myphi), 4]
med = np.median(sub, axis=1)

for i in range(1000):
    bif = sub[:, i]
    plt.plot(npe, bif, color="black", alpha=.3)
    plt.plot(npe, med, color="red")

plt.xscale('log')
plt.xlim(0, 3e-3)
# phi
sub = darr[:, nrho.searchsorted(0.05), :, 4]
med = np.median(sub, axis=1)

fig.add_subplot(312)
for i in range(60):
    bif = sub[:, i]
    plt.plot(npe, bif, color="blue", alpha=.3)
    plt.plot(npe, med, color="red")

plt.xscale('log')
plt.xlim(0, 3e-3)

fig.add_subplot(313)
plt.cla()
for pe in range(200):
    sub = darr[pe, :, nphi.searchsorted(myphi), 4]
    subm = np.median(sub)
    pearr = np.repeat(npe[pe], 1000)
    plt.plot(pearr, sub, 'o', alpha=.1, color="black")
    plt.plot(npe[pe], subm, 'o', color="red")

plt.xscale('log')
plt.xlim(0, 3e-3)

plt.plot()

x1d = np.array([1, 2, 3])
np.median(x1d)

x2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np.median(x2d, axis=1)

x3d = np.array(
    [[[1, 2, 3],
      [4, 5, 0],
      [7, 8, 9]],
     [[11, 12, 13],
      [14, 15, 16],
      [17, 18, 19]],
     [[21, 22, 23],
      [24, 25, 26],
      [27, 28, 29]]]
)

np.median(x3d, axis=0)

d3d = darr[:, :, :, :]
d3da = darr[:3, :3, :3, np.where(colnames == "phi")[0][0]]

d3d
d3da
x =np.median(d3d, axis=1)

fig=plt.figure()

plt.cla()
levels = [1, 1.05, 1.075, 1.1, 1.15, 1.18, 1.19, 1.195, 1.2, 1.25]
myphi=nphi.searchsorted(levels)
# myphi=nphi.searchsorted(1.185)
# plt.contour(x[:, :, 0], x[:, :, 4], x[:, :, 2], levels=levels, linestyles="-", colors="black")
plt.plot(x[:,myphi,0], x[:,myphi,4])
plt.xscale('log')
