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


# Definition Equations --------------------------------------------------------
def f_a(x, t, N, p_e, epsilon, rho, phi, beta, alpha):
    if x >= N:
        return 0
    else:
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


def get_st(t, e):
    if all(np.round(e, 6) > 0):
        return int(np.max(t))
    else:
        return int(np.where(np.round(e, 6) == 0)[0][0]+1)


# Debugging: -------------------------------------------------------------------
# t = np.linspace(0, 10000, 10001)
# par = [0,0.002,1.333]
# p_e, rho, phi = par[0], par[1], par[2]
# result = odeint(f_a, y0=0, t=t, args=(N, p_e, epsilon, rho, phi, beta, alpha),
#                 full_output=False)
# result[result > N] = N  # turn all x > N to N (fix numerical issue)
# e = f_e(result[:, 0], N, rho, phi)
# te = trapz(e, t)
# st = get_st(t, e)
# print("| pe, rho, phi:", par, "-- st:", st,
#       "-- te:", np.round(te, 18), "-- min_e:", np.round(np.min(e), 2), flush=True)
#
#
# import matplotlib.pyplot as plt
# plt.cla()
# plt.plot(t, result/N, label="a")
# plt.plot(t, e, label="e")
# plt.legend()
# plt.xscale('log')
# plt.show()

# input("stop.")
# ------------------------------------------------------------------------------

t = np.linspace(0, 10000, 10001)
data = []

for i in range(len(params)):
    par = params[i]
    p_e, rho, phi = par[0], par[1], par[2]
    result = odeint(f_a, y0=0, t=t, args=(N, p_e, epsilon, rho, phi, beta, alpha),
                    full_output=False)
    result[result > N] = N  # turn all x > N to N (fix numerical issue)
    e = f_e(result[:, 0], N, rho, phi)
    st = get_st(t, e)
    te = trapz(e[:st], t[:st])

    data.append(np.array([p_e, rho, phi, te, st]))
    print("#", str(i).zfill(5), "| pe, rho, phi:", np.round(par, 3), "-- st:", st,
          "-- te:", np.round(te,0), "-- min_e:", np.round(np.min(e),2), flush=True)

data = np.array(data)
np.savetxt(output_dir + "/chunk_" + str(njob).zfill(4) + ".txt", data,
           delimiter=",", newline="\n")

print("python script executed correctly.")
