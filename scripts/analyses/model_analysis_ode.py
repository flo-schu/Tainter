import numpy as np
import scipy.stats as sci
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
import pandas as pd
import os as os
from scipy.integrate import ode
from scipy.integrate import trapz

# os.chdir("./Tainter/Models/tf5_cluster")


# Parameters ###################################################################
N   = 400       # Network size
p_e = 0.02      # exploration probability

epsilon = 1     # threshold
rho     = 0.02  # link density in erdos renyi network
phi     = 1.0   # efficiency of coordinated Workers
beta    = 15    # scale parameter of beta distribution
alpha   = 1     # location parameter of beta distribution


# x_interval = np.linspace(0,1,100)
# plt.plot(x_interval, sci.beta.cdf(x_interval, beta, alpha))
# plt.title("beta cdf")
#plt.show()

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


t, x = get_timeseries(1, 1500, 0)

folder = input("folder name:")
data = pd.read_csv("./results/model/"+folder+"/t5_explore.csv")
# data = pd.read_csv("./results/model/"+folder+"/t5_base.csv")

plt.plot(data.x, data.Admin, label = "Admin model",
    ls = "--", color = "b")
plt.plot(t, np.array(x)/N, label = "Admin analytic")
plt.plot(data.x, data.Ecap,
    marker = "o", ls = "", label = "Ecap model", alpha = .1, color = "orange")
plt.plot(t, f_e(np.array(x), N, rho, phi),
    label = "Ecap analytic", color = "orange")
plt.legend()
plt.ylabel("Administrator share")
plt.xlabel("time")
# plt.savefig("./results/model/"+folder+"/comp_integration-model_exploration.png")
plt.show()
