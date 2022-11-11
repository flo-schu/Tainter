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
import os
import json
import glob
import numpy as np
from tqdm import tqdm
from tainter.model.approximation import integrate_fa

# environmental variables -----------------------------------------------------

def parameter_scan(
    output_dir, 
    njob, 
    parameters,
):
    print(parameters, output_dir, njob)

    # Parameters ------------------------------------------------------------------

    # load approximation params, order is important here
    with open(os.path.join(output_dir, "approx_params.json"), "r") as f:
        fixed_params = json.load(f)

    print(fixed_params)

    # load variable parameter file
    params = np.loadtxt(
        os.path.join(output_dir, f"params_{str(njob).zfill(4)}.txt"), 
        delimiter=","
    )



    # # Debugging: -------------------------------------------------------------------
    # import matplotlib.pyplot as plt
    # t = np.linspace(0, 10000, 10001)
    # par = [0.0027,0.002,1.333]
    # p_e, rho, phi = par[0], par[1], par[2]
    # result = odeint(f_a, y0=0, t=t, args=(N, p_e, epsilon, rho, phi, beta, alpha))
    # result[result > N] = N  # turn all x > N to N (fix numerical issue)
    # e = f_e(result[:, 0], N, rho, phi)
    # st = get_st(t, e)
    # te = trapz(e[:st], t[:st])

    # print("| pe, rho, phi:", par, "-- st:", st, "-- te:", np.round(te, 18), "-- min_e:", np.round(np.min(e), 2), flush=True)

    # plt.cla()
    # plt.plot(t, result/N, label="a")
    # plt.plot(t, e, label="e")
    # plt.legend()
    # plt.xscale('log')
    # plt.show()

    # input("stop.")
    # # ------------------------------------------------------------------------------

    t = np.linspace(0, 10000, 10001)
    data = []

    with tqdm(total=len(params)) as pbar:

        for i in range(len(params)):
            par = list(params[i])

            # set parameters 
            for pname, pval in zip(parameters, par):
                fixed_params[pname] = pval

            st, te, result, e = integrate_fa(t, fixed_params)
            data.append(np.array(par + [te, st]))

            pbar.update(1)

    data = np.array(data)
    np.savetxt(
        os.path.join(output_dir, "result_" + str(njob).zfill(4) + ".txt"),
        data,
        delimiter=",", newline="\n"
    )

def process_output(directory):
    result_files = glob.glob(os.path.join(directory, "result_*.txt"))

    data = []

    for f in result_files:
        data.append(np.loadtxt(f, delimiter=","))

    return np.concatenate(data)
    


if __name__ == "__main__":
    output_dir = sys.argv[1]
    njob = int(sys.argv[2])

    parameter_scan(output_dir=output_dir, njob=njob)