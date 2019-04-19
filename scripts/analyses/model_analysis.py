import numpy as np
import scipy.stats as sci
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
import pandas as pd
import os as os
#os.chdir("./Tainter/Models/tf5_cluster")

# Parameters ###################################################################
N   = 400       # Network size
p_e = 0.01      # exploration probability

epsilon = 1     # threshold
rho     = 0.05  # link density in erdos renyi network
phi     = 1.1   # efficiency of coordinated Workers
beta    = 15    # scale parameter of beta distribution
alpha   = 1     # location parameter of beta distribution
################################################################################


# functions ####################################################################
def f_a(x, N, p_e, epsilon, rho, phi, beta, alpha):
    return(
    p_e * (N - 2 * x) +
    sci.beta.cdf(
        (((epsilon * N) / (((N - x) * (1 - rho) ** x) +
        ((N - x) * (1 - (1 - rho) ** x)) ** phi))),
        a = beta, b = alpha)
    )

def f_e(x0, N, rho, phi):
    return(
    ((N - x0) * (1 - rho) ** x0 + ((N - x0) * (1 - (1 - rho) ** x0)) ** phi) / N
    )

# roots ########################################################################
# fsolve returns the roots (nullstellen) of the functions. These points are
# fix points of the function

x0 = fsolve(
    f_a, 200,
    args = (N, p_e, epsilon, rho, phi, beta, alpha))

x = np.linspace(0,399,100)

plt.plot(x,f_a(x,N, p_e, epsilon, rho, phi, beta, alpha))
plt.axhline(0, c= 'black', lw = .5)
plt.axvline(x0, c = "red", lw = 1, ls = "--")
plt.show()


# calculate Ecap for roots #####################################################

e0 = f_e(x0, N, rho, phi)

plt.plot(x, f_e(x, N, rho, phi))
plt.axvline(x0, c = "red", lw = 1, ls = "--")
plt.axhline(e0, c = "g", lw = 1, ls = "-")


# add fixpoints to data found in analysis ######################################
# folder = input("folder name")
folder = "20190419_0930"
data_explore = pd.read_csv("../../results/model/"+folder+"/t5_explore.csv")
data_base    = pd.read_csv("../../results/model/"+folder+"/t5_base.csv")

fig, (ax1,ax3) = plt.subplots(2,1,sharex=True)
ax3.plot('x','Admin',ls='-',fillstyle='none', data = data_explore,
    label = "A total")
ax3.fill_between(data_explore['x'],data_explore['A_pool_shk'],
    label = "A_shock",alpha = .5)
ax3.fill_between(data_explore['x'],data_explore['A_pool_exp']+data_explore['A_pool_shk'],
    data_explore['A_pool_shk'],
    label = "A_explore",alpha = .5)
ax3.axhline(x0/N, c = "b", ls = "--", lw = 2)
ax4 = ax3.twinx()
ax4.axhline(e0, c = "g", ls = "--", lw = 2)

ax4.plot(data_explore['x'], data_explore['Ecap'],
    label = "Energy per capita", c = "g", ls = "-", lw = .5)

ax3.set_ylim(0,1)
ax2= ax1.twinx()
ax1.plot(data_base['x'],data_base.Admin,ls='-',fillstyle='none')

ax1.fill_between(data_base['x'],data_base['A_pool_shk'],
    alpha = .5)
ax1.fill_between(data_base['x'],data_base['A_pool_exp']+data_base['A_pool_shk'],
    data_base['A_pool_shk'],
    alpha = .5)
ax2.plot(data_base['x'], data_base['Ecap'], c = "g", ls = "-", lw = .5)

ax1.set_ylim(0,1)
plt.setp(ax1.get_xticklabels(), visible=False)
ax4.set_ylim(0,max(np.append(data_base["Ecap"],data_explore['Ecap']))+.1)
ax2.set_ylim(0,max(np.append(data_base["Ecap"],data_explore['Ecap']))+.1)
fig.text(0.03, 0.5, 'Administrator share', ha='center',
    va='center', rotation='vertical')
fig.text(0.97, 0.5, 'Energy per capita', ha='center',
    va='center', rotation='vertical')
fig.text(0.5, 0.04, 'Time', ha='center', va='center')
fig.legend(ncol = 4)
fig.savefig("../../results/model/"+folder+"/Admin_Ecap_twocases_fixpoints.png")
plt.show()

# Analyse e for many phi #######################################################

# phi_it = np.linspace(1,1.5,100)
# e0_phi =f_e(fsolve(f_a, 390, args = (N, p_e, epsilon, rho, phi, beta, alpha) ), N, rho, phi_it)
#
# data = pd.read_csv("./results/model/20190327_1730/filtered_data.csv")
#
# plt.plot(phi_it, e0_phi)
