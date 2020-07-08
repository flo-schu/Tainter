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
from scipy.integrate import odeint
from scipy.integrate import trapz
import matplotlib.pyplot as plt

# environmental variables -----------------------------------------------------
paramfile = sys.argv[1]
output_dir = sys.argv[2]
njob = int(sys.argv[3])

# print(paramfile, output_dir, njob)

# Parameters ------------------------------------------------------------------
N = 400  # Network size
epsilon = 1  # threshold
beta = 15  # scale parameter of beta distribution
alpha = 1  # location parameter of beta distribution
# p_e, rho, phi
params = np.loadtxt(paramfile, delimiter=",")


# print(params)


# Definition Equations --------------------------------------------------------
def f_a(t, x, N, p_e, epsilon, rho, phi, beta, alpha):
    return (
        p_e * (N - 2 * x) + sci.beta.cdf(
        (((epsilon * N) / (((N - x) * (1 - rho) ** x) +
        ((N - x) * (1 - (1 - rho) ** x)) ** phi))),
        a=beta, b=alpha)
    )


def f_e(x, N, rho, phi):
    return (
            ((N - x) * (1 - rho) ** x + ((N - x) * (1 - (1 - rho) ** x)) ** phi) / N
    )


def get_st(t, e):
    return t[e <= 0]


t = np.linspace(0, 10000, 10000)
data = []

for i in range(len(params)):
    print(i)
    par = params[i]
    p_e, rho, phi = par[0], par[1], par[2]
    x = odeint(f_a, y0=0, t=t, args=(N, p_e, epsilon, rho, phi, beta, alpha))
    e = f_e(np.array(x[:, 0]), N, rho, phi)
    print(e)
    plt.plot(t, x / N, label="admin")
    plt.plot(t, e, label="energy")
    plt.legend()
    plt.show()
    te = trapz(e, t)
    st = get_st(t, e)[0]

    data.append(np.array([p_e, rho, phi, te, st]))
    print("pe, rho, phi: ", par, " -- st: ", st, " -- te: ", te)

data = np.array(data)
np.savetxt(output_dir + "/chunk_" + str(njob).zfill(4) + ".txt", data,
           delimiter=",", newline="\n")

print("python script executed correctly.")
