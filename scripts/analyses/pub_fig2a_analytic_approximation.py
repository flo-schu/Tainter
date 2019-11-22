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

st_e.append(5000.1)
# print(st_e)
# sns.kdeplot(st_e)
# plt.show()
# input("enter")

textwidth = 12.12537
plt.rcParams.update({'font.size': 14})

cmap = cm.get_cmap("tab20",20)
fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex = True,
                                 figsize = (textwidth, textwidth/2))

cv = dict(p0 = 14, p1 = 0, p2 = 2)


alpha_sim = .05
# simulation
ax1.plot(admin_b, color = cmap(cv["p0"]+1), alpha = alpha_sim)
ax1.plot(admin_i, color = cmap(cv["p1"]+1), alpha = alpha_sim)
ax1.plot(admin_e, color = cmap(cv["p2"]+1), alpha = alpha_sim)
ax1.yaxis.set_label_position("right")
ax1.set_ylabel("admin")

ax2.plot(ecap_b,  color = cmap(cv["p0"]+1), alpha = alpha_sim)
ax2.plot(ecap_i,  color = cmap(cv["p1"]+1), alpha = alpha_sim)
ax2.plot(ecap_e,  color = cmap(cv["p2"]+1), alpha = alpha_sim)
ax2.yaxis.set_label_position("right")
ax2.set_ylabel("energy $cap^{-1}$")

# density plots
sns.kdeplot(st_b, ax=ax3, color = cmap(cv["p0"]))
sns.kdeplot(st_i, ax=ax3, color = cmap(cv["p1"]))
sns.kdeplot(st_e, ax=ax3, color = cmap(cv["p2"]))
ax3.set_ylim(0,0.001)
ax3.set_xlim(-250,5250)
ax3.yaxis.set_label_position("right")
ax3.set_ylabel("collapse")

# analytic
ax1.plot(tb, np.array(xb)/N, "--", color = cmap(cv["p0"]), label = "$p_{e}$ = 0.0")
ax1.plot(ti, np.array(xi)/N, "--", color = cmap(cv["p1"]), label = "$p_{e}$ = 0.00275")
ax1.plot(te, np.array(xe)/N, "--", color = cmap(cv["p2"]), label = "$p_{e}$ = 0.02")

ax2.plot(tb, f_e(np.array(xb), N, rho, phi),"--", color = cmap(cv["p0"]))
ax2.plot(ti, f_e(np.array(xi), N, rho, phi),"--", color = cmap(cv["p1"]))
ax2.plot(te, f_e(np.array(xe), N, rho, phi),"--", color = cmap(cv["p2"]))

fig.subplots_adjust(bottom=0.08, left = 0.1, right = 0.94,top = .95)

fig.legend(loc = "center", bbox_to_anchor = (.75,.22),ncol = 1, frameon = False)
fig.text(0.5, 0.01, 'Time', ha='center', va='center')
ax1.annotate("A", xy=(0.01, 0.83), xycoords="axes fraction")
ax2.annotate("B", xy=(0.01, 0.83), xycoords="axes fraction")
ax3.annotate("C", xy=(0.01, 0.83), xycoords="axes fraction")


plt.savefig("../../results/model/"+folder+"/pub_figure3.png")
plt.show()
