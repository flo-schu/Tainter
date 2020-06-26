import csv
import numpy as np
import itertools as it

rho = np.linspace(0, 0.3, 11)  # link density in erdos renyi network
phi = np.linspace(1, 1.5, 11)  # efficiency of coordinated Workers

pe_null = np.array([0])
pe_explore = np.logspace(-4, -1.6, num=11)

pe_range = np.concatenate((pe_null, pe_explore), axis=None)
pargrid = it.product(pe_range, rho, phi)

with open("params.csv", "w", newline="") as file:
    f = csv.writer(file)
    f.writerow(["pe", "rho", "phi"])
    f.writerows(pargrid)
