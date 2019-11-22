import numpy as np
import scipy.stats as sci
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd
import os as os
from scipy.integrate import ode
from scipy.integrate import trapz
from collections import OrderedDict
import seaborn as sns

# os.chdir("./Tainter/Models/tf5_cluster")
# case = input("plot explore or base?")


# Parameters ###################################################################
N   = 400       # Network size

# if case == "explore":
#     p_e = 0.02      # exploration probability
# elif case == "base":
#     p_e = 0.0
# else:
#     case = input("plot explore or base?")

epsilon = 1     # threshold
rho     = 0.02  # link density in erdos renyi network
phi     = 1.05  # efficiency of coordinated Workers
beta    = 15    # scale parameter of beta distribution
alpha   = 1     # location parameter of beta distribution


# x_interval = np.linspace(0,1,100)
# plt.plot(x_interval, sci.beta.cdf(x_interval, beta, alpha))
# plt.title("beta cdf")
#plt.show()

def f_a(t, x, N, p_e, epsilon, rho, phi, beta, alpha):
    #TODO: Marc kannst du das mal checken?!
    # print(x )
    if (x >= N):
        factor = 0
    else:
        factor = sci.beta.cdf(
        (((epsilon * N) / (((N - x) * (1 - rho) ** x) +
        ((N - x) * (1 - (1 - rho) ** x)) ** phi))),
        a = beta, b = alpha)

    return(p_e * (N - 2 * x) + factor)

    # return(
    #     p_e * (N - 2 * x)  +
    # sci.beta.cdf(
    #     (((epsilon * N) / (((N - x) * (1 - rho) ** x) +
    #     ((N - x) * (1 - (1 - rho) ** x)) ** phi))),
    #     a = beta, b = alpha)
    # )

def f_e(x, N, rho, phi):
    x = np.array([N if i >= N else i for i in x ])

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
##################################################
p_e = 0.0 # base
tb, xb = get_timeseries(1, 5000, 0)
# xb2 = np.mean(np.array(xb[0:150000]).reshape(-1,100),axis = 1)
# tb2 = np.arange(1500)
# print(xb[len(xb)-10:len(xb)])
# plt.plot(tb,np.array(xb)/N)
# plt.plot(tb,f_e(np.array(xb), N, rho, phi))
# plt.show()
# print(f_e(np.array(xb), N, rho, phi))
p_e = 0.02 # explore
te, xe = get_timeseries(1, 5000, 0)

p_e = 0.00275 # explore
ti, xi = get_timeseries(1, 5000, 0)

# len(te)

# folder = input("folder name:")
# folder = "20190419_1134"
# folder = "20191121_1616" # pe(i)=0.003
folder = "20191121_1732" # pe(i)=0.00275
admin_b = np.load("../../results/model/"+folder+"/base_admin.npy")
ecap_b = np.load("../../results/model/"+folder+"/base_ecap.npy")

admin_e = np.load("../../results/model/"+folder+"/expl_admin.npy")
ecap_e = np.load("../../results/model/"+folder+"/expl_ecap.npy")

admin_i = np.load("../../results/model/"+folder+"/inter_admin.npy")
ecap_i = np.load("../../results/model/"+folder+"/inter_ecap.npy")

st_b = list()
st_i = list()
st_e = list()

for i in np.arange(100):
    st_b.append(np.sum(ecap_b[:,i]!=0))
    st_i.append(np.sum(ecap_i[:,i]!=0))
    st_e.append(np.sum(ecap_e[:,i]!=0))


textwidth = 12.12537
plt.rcParams.update({'font.size': 14})

cmap = cm.get_cmap("tab20",20)
fig,(ax1,ax3,ax5) = plt.subplots(3,1,sharex = True,
                                 figsize = (textwidth, textwidth/2))
ax2= ax1.twinx()
ax4= ax3.twinx()
ax6= ax5.twinx()
ax7= ax1.twinx()
ax8= ax3.twinx()
ax9= ax5.twinx()

alpha_sim = .05
# simulation
for i in np.arange(100):
    ax1.plot(admin_b[:,i],
             label = "$A_{simulation}$" if i == 0 else "",
             color = cmap(1), alpha = alpha_sim)
for i in np.arange(100):
    ax2.plot(ecap_b[:,i], label = "$E_{simulation}$" if i == 9 else "",
             alpha = alpha_sim, color = cmap(3))

ax3.plot(admin_i, color = cmap(1), alpha = alpha_sim)
ax4.plot(ecap_i,  color = cmap(3), alpha = alpha_sim)

ax5.plot(admin_e,  color = cmap(1), alpha = alpha_sim)
ax6.plot(ecap_e, alpha = alpha_sim, color = cmap(3))

# analytic
ax1.plot(tb, np.array(xb)/N, "--",
         label = "$A_{analytic}$", color = cmap(0))
ax2.plot(tb, f_e(np.array(xb), N, rho, phi), "--",
         label = "$E_{analytic}$", color = cmap(2))

ax3.plot(ti, np.array(xi)/N, "--",color = cmap(0))
ax4.plot(ti, f_e(np.array(xi), N, rho, phi),"--",
         color = cmap(2))

ax5.plot(te, np.array(xe)/N, "--",color = cmap(0))
ax6.plot(te, f_e(np.array(xe), N, rho, phi),"--",
         color = cmap(2))

sns.kdeplot(st_b, ax=ax7)
sns.kdeplot(st_i, ax=ax8)
sns.kdeplot(st_e, ax=ax9)

# ax7.set_ylim(0,.1)
# ax8.set_ylim(0,.1)
# ax9.set_ylim(0,.1)



ax1.set_ylim(0,1.05)
ax3.set_ylim(0,1.05)
ax5.set_ylim(0,1.05)
# ax2.set_ylim(-.05,1.5)
# ax4.set_ylim(-.05,1.5)
# ax6.set_ylim(-.05,1.5)

fig.subplots_adjust(bottom=0.08, left = 0.06, right = 0.94,top = .95)
fig.text(0.01, 0.5, 'Administrator share', ha='center',
    va='center', rotation='vertical')
fig.text(0.99, 0.5, 'Energy per capita', ha='center',
    va='center', rotation='vertical')
fig.text(0.5, 0.01, 'Time', ha='center', va='center')
ax1.annotate("A", xy=(0.01, 0.83), xycoords="axes fraction")
ax3.annotate("B", xy=(0.01, 0.83), xycoords="axes fraction")
ax5.annotate("C", xy=(0.01, 0.83), xycoords="axes fraction")
ax1.annotate("$p_{e}$ = 0",       xy=(0.85, 0.75), xycoords="axes fraction")
ax3.annotate("$p_{e}$ = 0.00275", xy=(0.85, 0.75), xycoords="axes fraction")
ax5.annotate("$p_{e}$ = 0.02",    xy=(0.85, 0.75), xycoords="axes fraction")
# handles, labels = plt.gca().get_legend_handles_labels()
# print(labels)
# by_label = OrderedDict(zip(labels, handles))
# print(by_label)
# fig.legend(by_label.values(), by_label.keys(),
#     loc = "center", bbox_to_anchor = (.75,.7),ncol = 1)
# fig.legend(loc = "center", bbox_to_anchor = (.55,.8),ncol = 4, frameon = False)
plt.savefig("../../results/model/"+folder+"/comp_integration-model_exploration.png")
plt.show()
