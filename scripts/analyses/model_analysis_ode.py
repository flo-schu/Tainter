import numpy as np
import scipy.stats as sci
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd
import os as os
from scipy.integrate import ode
from scipy.integrate import trapz

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

p_e = 0.0 # base
tb, xb = get_timeseries(1, 1500, 0)
# plt.plot(tb,np.array(xb)/N)
# plt.plot(tb,f_e(np.array(xb), N, rho, phi))
# plt.show()
p_e = 0.02 # explore
te, xe = get_timeseries(1, 1500, 0)



# folder = input("folder name:")
folder = "20190419_1134"
data_b = pd.read_csv("../../results/model/"+folder+"/t5_base.csv")
data_e = pd.read_csv("../../results/model/"+folder+"/t5_explore.csv")

cmap = cm.get_cmap("tab20",20)
fig,(ax1,ax3) = plt.subplots(2,1,sharex = True)
ax1.plot(data_b.x, data_b.Admin, label = "$A_{simulation}$",
    marker = "o", ls = "",  color = cmap(1))
ax1.plot(tb, np.array(xb)/N, label = "$A_{analytic}$", color = cmap(0))
ax2= ax1.twinx()
ax2.plot(data_b.x, data_b.Ecap,
    marker = "o", ls = "", label = "$E_{simulation}$", alpha = .1, color = cmap(3))
ax2.plot(tb, f_e(np.array(xb), N, rho, phi),
    label = "$E_{analytic}$", color = cmap(2))

ax3.plot([data_e.x], [data_e.Admin],
    marker = "o", ls = "",  color = cmap(1))
ax3.plot(tb, np.array(xe)/N, color = cmap(0))
ax4= ax3.twinx()
ax4.plot([data_e.x], [data_e.Ecap],
    marker = "o", ls = "", alpha = .1, color = cmap(3))
ax4.plot(tb, f_e(np.array(xe), N, rho, phi),
    color = cmap(2))


fig.text(0.05, 0.5, 'Administrator share', ha='center',
    va='center', rotation='vertical')
fig.text(0.98, 0.5, 'Energy per capita', ha='center',
    va='center', rotation='vertical')
fig.text(0.5, 0.04, 'Time', ha='center', va='center')
fig.legend(loc = "center", bbox_to_anchor = (.75,.7),ncol = 1)
plt.savefig("../../results/model/"+folder+"/comp_integration-model_exploration.png")
plt.show()
