# created by Florian Schunck on 26.06.2020
# Project: tainter
# Short description of the feature:
# 1. parameter analysis of the exploration, link density and output elasticity
#    by approximated analytic equations.
# 2. can take arguments from call on EVE cluster from bash script and outputs
#    results in chunks which can then be collected afterwards.
# ------------------------------------------------------------------------------
# Open tasks:
# TODO:
#
# ------------------------------------------------------------------------------

import sys
import numpy as np
import scipy.stats as sci
from scipy.integrate import ode
from scipy.integrate import trapz

# environmental variables -----------------------------------------------------
paramfile = sys.argv[1]
output_dir = sys.argv[2]
njob = int(sys.argv[3])

print(paramfile, output_dir, njob)

# Parameters ------------------------------------------------------------------
N = 400  # Network size
epsilon = 1  # threshold
beta = 15  # scale parameter of beta distribution
alpha = 1  # location parameter of beta distribution
# p_e, rho, phi
params = np.loadtxt(paramfile, delimiter=",")

print(params)
# Definition Equations --------------------------------------------------------
def f_a(t, x, N, p_e, epsilon, rho, phi, beta, alpha):
    return (
            p_e * (N - 2 * x) +
            sci.beta.cdf(
                (((epsilon * N) / (((N - x) * (1 - rho) ** x) +
                                   ((N - x) * (1 - (1 - rho) ** x)) ** phi))),
                a=beta, b=alpha)
    )


def f_s(x, N, p_e, epsilon, rho, phi, beta, alpha):
    return (
            p_e * (N - 2 * x) +
            sci.beta.cdf(
                (((epsilon * N) / (((N - x) * (1 - rho) ** x) +
                                   ((N - x) * (1 - (1 - rho) ** x)) ** phi))),
                a=beta, b=alpha)
    )


def f_e(x, N, rho, phi):
    return (
            ((N - x) * (1 - rho) ** x + ((N - x) * (1 - (1 - rho) ** x)) ** phi) / N
    )


# Solver ----------------------------------------------------------------------
def get_timeseries(timestep, tmax, initial_value, p_e, rho, phi):
    r = ode(f_a)
    r.set_initial_value(initial_value)
    r.set_f_params(N, p_e, epsilon, rho, phi, beta, alpha)
    r.set_integrator("vode")

    t = [0]
    results = np.array([initial_value])
    while r.t < tmax and r.successful():
        print(r.t)

        r.integrate(r.t + timestep)
        t.append(r.t)
        results = np.append(results, r.y)

    return t, results


data = []

for i in range(len(params)):
    print(i)
    par = params[i]
    print(par)
    p_e, rho, phi = par[0], par[1], par[2]
    print("...")
    t, x = get_timeseries(1, 10000, 0, p_e, rho, phi)
    print("...")
    e = f_e(np.array(x), N, rho, phi)
    te = trapz(e, t)
    st = t[-1]
    data.append(np.array([p_e, rho, phi, te, st]))
    print("pe, rho, phi: ", par, " -- st: ", st, " -- te: ", te)

data = np.array(data)
np.savetxt(output_dir + "/chunk_" + str(njob).zfill(4) + ".txt", data,
           delimiter=",", newline="\n")

print("python script executed correctly.")
