import numpy as np
import scipy.stats as sci
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
import pandas as pd
import os as os
from scipy.integrate import ode
from scipy.integrate import trapz
import itertools as it
# os.chdir("./Tainter/Models/tf5_cluster")


# Parameters ###################################################################
N   = 400       # Network size

epsilon = 1     # threshold
beta    = 15    # scale parameter of beta distribution
alpha   = 1     # location parameter of beta distribution

# Definition Equations #########################################################
def f_a(t, x, N, p_e, epsilon, rho, phi, beta, alpha):
    return(
    p_e * (N - 2 * x)  +
    sci.beta.cdf(
        (((epsilon * N) / (((N - x) * (1 - rho) ** x) +
        ((N - x) * (1 - (1 - rho) ** x)) ** phi))),
        a = beta, b = alpha)
    )

def f_s(x, N, p_e, epsilon, rho, phi, beta, alpha):
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

# Solver #######################################################################
def get_timeseries(timestep, tmax, initial_value, rho, phi):
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

# folder = "20190419_1134"
folder = "20191025_1217"


rho     = np.linspace(0,0.1,101)  # link density in erdos renyi network
phi     = np.linspace(1,1.5,101)   # efficiency of coordinated Workers
pargrid = it.product(rho, phi)

t_data = list()
x_data = list()
te_data = list()
rho_par = list()
phi_par = list()
s_data = list()

p_e = 0.0      # exploration probability

for i in pargrid:
    rho_i, phi_i = i[0], i[1]
    t, x = get_timeseries(1, 10000, 0, rho_i, phi_i)

    e = f_e(np.array(x),N,rho_i, phi_i)
    te = trapz(e,t)
    rho_par.append(rho_i)
    phi_par.append(phi_i)

    te_data.append(te[0])
    s_data.append(t[-1])
    print(i, "of", np.max(rho), np.max(phi))


data = pd.DataFrame(np.array([phi_par,rho_par,te_data,s_data]).transpose())
data.columns = ["phi", "rho","te","s"]
data.to_csv("../../results/model/"+folder+"/parscan_base.csv")

pargrid = it.product(rho, phi)

t_data = list()
x_data = list()
te_data = list()
rho_par = list()
phi_par = list()
s_data = list()

p_e = 0.02      # exploration probability

for i in pargrid:
    rho_i, phi_i = i[0], i[1]
    t, x = get_timeseries(1, 10000, 0, rho_i, phi_i)

    e = f_e(np.array(x),N,rho_i, phi_i)
    te = trapz(e,t)
    rho_par.append(rho_i)
    phi_par.append(phi_i)

    te_data.append(te[0])
    s_data.append(t[-1])
    print(i, "of", np.max(rho), np.max(phi))


data = pd.DataFrame(np.array([phi_par,rho_par,te_data,s_data]).transpose())
data.columns = ["phi", "rho","te","s"]
data.to_csv("../../results/model/"+folder+"/parscan_expl"+str(p_e)+".csv")
