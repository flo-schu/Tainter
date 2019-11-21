import numpy as np
import scipy.stats as sci
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
import pandas as pd
import os as os
from scipy.integrate import ode
from scipy.integrate import trapz
# os.chdir("./Tainter/Models/tf5_cluster")

def f_a(t, x, N, p_e, epsilon, rho, phi, beta, alpha):
    return(
    p_e * (N - 2 * x)  +
    sci.beta.cdf(
        (((epsilon * N) / (((N - x) * (1 - rho) ** x) +
        ((N - x) * (1 - (1 - rho) ** x)) ** phi))),
        a = beta, b = alpha)
    )

def f_e(x, N, rho, phi):
    return(
    ((N - x) * (1 - rho) ** x + ((N - x) * (1 - (1 - rho) ** x)) ** phi) / N
    )

def get_timeseries(timestep, tmax, initial_value):
    r = ode(f_a)
    r.set_initial_value(initial_value)
    r.set_f_params(N, p_e, epsilon, rho, phi, beta, alpha)
    r.set_integrator("vode")

    t = [0]
    results = [initial_value]

    while r.t < tmax and r.successful():
        r.integrate(r.t+timestep)
        t.append(r.t)
        results.append(r.y)

    return t, results

# with exp
# Parameters ###################################################################
N   = 400       # Network size
p_e = 0.01      # exploration probability

epsilon = 1     # threshold
rho     = 0.05  # link density in erdos renyi network
phi     = 1.05   # efficiency of coordinated Workers
beta    = 15    # scale parameter of beta distribution
alpha   = 1     # location parameter of beta distribution

t_e, x_e = get_timeseries(1, 1500, 0)


# with exp
# Parameters ###################################################################
N   = 400       # Network size
p_e = 0.00      # exploration probability

epsilon = 1     # threshold
rho     = 0.05  # link density in erdos renyi network
phi     = 1.05   # efficiency of coordinated Workers
beta    = 15    # scale parameter of beta distribution
alpha   = 1     # location parameter of beta distribution

t_b, x_b = get_timeseries(1, 1500, 0)


# folder = "20190327_1730"
folder = "20191121_1152"
data_explore = pd.read_csv("../../results/model/"+folder+"/t5_explore.csv")
data_base    = pd.read_csv("../../results/model/"+folder+"/t5_base.csv")


fig, (ax1,ax3) = plt.subplots(2,1,sharex=True)
cmap = plt.get_cmap("tab10")
ax3.plot('x','Admin',ls='-',fillstyle='none', data = data_explore,
    label = "A total", color = cmap(0))
ax3.plot(t_e, np.array(x_e)/N, ls = "--", c = cmap(0))
ax3.fill_between(data_explore['x'],data_explore['A_pool_shk'],
    label = "A_shock",alpha = .5, color = cmap(0))
ax3.fill_between(data_explore['x'],data_explore['A_pool_exp']+data_explore['A_pool_shk'],
    data_explore['A_pool_shk'],
    label = "A_explore",alpha = .5, color = cmap(1))
ax4 = ax3.twinx()
ax4.plot(data_explore['x'], data_explore['Ecap'],
    label = "Energy per capita", ls = "-", lw = .5, c = cmap(2))
ax4.plot(t_e, f_e(np.array(x_e), N, rho, phi), ls = "--", c= cmap(2))


ax3.set_ylim(0,1)
ax2= ax1.twinx()
ax1.plot(np.array(data_base['x']),np.array(data_base['Admin']),
    ls='-',fillstyle='none',c=cmap(0))
ax1.plot(t_b, np.array(x_b)/N, ls = "--",c=cmap(0))
ax1.fill_between(np.array(data_base['x']),np.array(data_base['A_pool_shk']),
    alpha = .5, color=cmap(0))
ax1.fill_between(np.array(data_base['x']),np.array(data_base['A_pool_exp'])+np.array(data_base['A_pool_shk']),
    data_base['A_pool_shk'],
    alpha = .5, color=cmap(1))
ax2.plot(np.array(data_base.x), np.array(data_base.Ecap), ls = "-", lw = .5, c=cmap(2))
ax2.plot(t_b, f_e(np.array(x_b), N, rho, phi), ls = "--", c=cmap(2))

ax1.set_ylim(0,1)
plt.setp(ax1.get_xticklabels(), visible=False)
ax4.set_ylim(0,max(np.append(data_base["Ecap"],data_explore['Ecap']))+.1)
ax2.set_ylim(0,max(np.append(data_base["Ecap"],data_explore['Ecap']))+.1)
fig.legend(ncol = 4)
fig.text(0.03, 0.5, 'Administrator share', ha='center',
    va='center', rotation='vertical')
fig.text(0.97, 0.5, 'Energy per capita', ha='center',
    va='center', rotation='vertical')
fig.text(0.5, 0.04, 'Time', ha='center', va='center')
plt.show()
fig.savefig("./results/model/"+folder+"/Admin_Ecap_twocases_analytic.png")
