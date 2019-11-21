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
folder = "20191112_0937"
# os.makedirs("../../results/model/"+folder)
rho     = np.linspace(0,1,11)  # link density in erdos renyi network
phi     = np.linspace(1,1.5,11)   # efficiency of coordinated Workers
pe_range= np.logspace(np.log10(0.0001),np.log10(.02),21)
pargrid = it.product(rho, phi)

print(pe_range)
input("press enter")

t_data = list()
x_data = list()
te_data = list()
rho_par = list()
phi_par = list()
pe_par  = list()
s_data = list()

for p_e in pe_range:
    print(p_e)
    pargrid = it.product(rho, phi)
    for i in pargrid:
        rho_i, phi_i = i[0], i[1]
        t, x = get_timeseries(1, 10000, 0, rho_i, phi_i)

        e = f_e(np.array(x),N,rho_i, phi_i)
        te = trapz(e,t)
        rho_par.append(rho_i)
        phi_par.append(phi_i)
        pe_par.append(p_e)

        te_data.append(te[0])
        s_data.append(t[-1])
        print(p_e, i, "of",np.max(pe_range), np.max(rho), np.max(phi))


data = pd.DataFrame(np.array([pe_par,phi_par,rho_par,te_data,s_data]).transpose())
data.columns = ["p_e","phi", "rho","te","s"]
data.to_csv("../../results/model/"+folder+"/parscan.csv")
