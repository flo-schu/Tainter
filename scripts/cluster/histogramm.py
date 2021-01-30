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
import seaborn as sns

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

d_interesting = data[(data[:, 4] > 9000) & (data[:, 4] < 10000), :]

darr = data.reshape((len(npe), len(nrho), len(nphi), 5))
d = data[data[:, 4] < 10000, 4]
ax = plt.subplot()
ax2 = ax.twinx()
ax.cla(); ax2.cla()
ax.hist(d, bins=50)
ax2 = sns.kdeplot(d, clip=(0, 10000), color="orange")
ax.set_xlabel("survival time")
ax.set_ylabel("frequency")
ax2.set_ylabel("density")
