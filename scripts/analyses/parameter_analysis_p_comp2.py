import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import sys
sys.path.append('../helpers/')
from shifted_cmap import shiftedColorMap


folder = "20191112_0937"

data   = pd.read_csv("../../results/model/"+folder+"/parscan.csv")

### FILTER DATA SET ###################################################
data   = data.query("rho < 1")
#

p_e = data.p_e
rho = data.rho
phi = data.phi
npe = p_e.unique()
nrho = rho.unique()
nphi = phi.unique()

dat = np.array(data)[:,1:]
names = np.array(data.columns[1:])

out_e = list()
out_s = list()
out

for i in npe:
    sube = dat[p_e == i, names == "te"]
    subs = dat[p_e == i, names == "s"]
    out_e.append(np.mean(sube))
    out_s.append(np.mean(subs))

fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)

ax1.plot(npe, out_e, label = "energy")
ax2.plot(npe, out_s, label = "survival")
ax2.set_xlabel("exploration probability (p_e)")
ax1.set_ylabel("mean energy production  ")
ax2.set_ylabel("mean survival time")
ax1.legend()
ax2.legend()

# plt.yscale('log')
# plt.xscale('log')
plt.savefig("../../results/model/"+folder+"/total_energy_pe.jpg")
plt.show()
