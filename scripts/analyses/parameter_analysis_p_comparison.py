import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import sys
sys.path.append('../helpers/')
from shifted_cmap import shiftedColorMap


folder = "20191027_0101"

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

te_0 = dat[p_e == 0, names == "te"]
st_0 = dat[p_e == 0, names == "s"]
# print(te_0)
# replicate te of pe = 0 npe times and reshape to array of arrays with 1 element each
te_0_arr = np.reshape(np.array(te_0.tolist() * len(npe)),(len(data),1))
st_0_arr = np.reshape(np.array(st_0.tolist() * len(npe)),(len(data),1))
# add new arrays to existing array
dat = np.append(dat, te_0_arr, axis = 1)
dat = np.append(dat, st_0_arr, axis = 1)
names = np.append(names,["te0","st0"])

te_diff = dat[:,names == "te"] - dat[:, names == "te0"]
st_diff = (dat[:,names == "s"] - dat[:, names == "st0"]) / dat[:, names == "s"]
te_diff_arr = np.reshape(te_diff, (len(data),1))
st_diff_arr = np.reshape(st_diff, (len(data),1))
dat = np.append(dat, te_diff_arr, axis = 1)
dat = np.append(dat, st_diff_arr, axis = 1)
names = np.append(names,["tediff","stdiff"])

comp = dat[te_diff[:,0] < 0, : ]
print(names)

input("abort? Press ctrl+C or enter to continue")
size_badarea = list()
magn_badarea = list()
avg_st = list()
for i in npe:
    te = comp[comp[:,names == "p_e"][:,0] == i, names == "tediff"]
    # calculate mean difference of survival time.
    # if positive,
    st = np.mean(dat[dat[:,names == "p_e"][:,0] == i,names=="stdiff"])
    size_badarea.append(len(te))
    magn_badarea.append(sum(te)/-10000)
    avg_st.append(st*-1)
# plt.plot(npe, size_badarea)
e_surp = np.array(magn_badarea)/np.array(size_badarea)
# esurp_max =
e_surp[0] = 0 # because division through 0 but in reality it is 0
plt.plot(npe, e_surp)
plt.plot(npe, avg_st, "--")
# plt.plot(npe, magn_badarea, linestyle = "--")
# plt.plot(npe, size_badarea, linestyle = "-.")
plt.xlabel("p_e")
plt.ylabel("average energy surplus per unit time")
plt.savefig("../../results/model/"+folder+"/comparison_size_badarea_over_pe.jpg")
plt.show()

# for this plot, the size of the parameterspace is calculated
# where t_diff is negative. This means the region of the parameter space
# in which the classic model outperforms the social mobility model
# additionally, the magnitude of the outperformance is calculated,
# by summing up all negative t_diff values. They are then divided by 10000,
# to get the average per unit time (because the model runs for 10000 timesteps)
# to get a sense what the average gain of the classic model is, this figure
# (magnitude) is divided by the size of the parameter space.
# Thus the average gain per unit time over the parameter ranges is calculated.
# The average is comprehensive since, the whole range over the phi parameter
# is included and also rho is computed between [0,1) with only exclusion of 1,
# because it represents an extreme case, where all nodes of the network
# are connected.
