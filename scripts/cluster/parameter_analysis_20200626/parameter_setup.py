# created by Florian Schunck on 26.06.2020
# Project: tainter
# Short description of the feature:
# 1. create a parameter grid and save it to chunks of size n. The last chunk
#    contains the tailing elements and can be smaller than n.
# 2. allows to create one chunk for each parameter combination or one chunk
#    with all combination or any number in between
#
# ------------------------------------------------------------------------------
# Open tasks:
# TODO:
#
# ------------------------------------------------------------------------------

import numpy as np
import itertools as it
import math

# params
rho = np.linspace(0, 0.3, 51)  # link density in erdos renyi network
phi = np.linspace(1, 1.5, 51)  # efficiency of coordinated Workers
pe_null = np.array([0])
pe_explore = np.logspace(-4, -1.6, num=51)
p_e = np.concatenate((pe_null, pe_explore), axis=None)

# chunks
n_pargrid = len(rho) * len(phi) * len(p_e)
# choose n_pargrid if only one chunk should be computed,
# choose 1 if n chunks should be created
n = 1000
n_chunks = math.ceil(n_pargrid / n)

# parameter grid
pargrid = np.array(list(it.product(p_e, rho, phi)))

# save paramter chunks
for chunk in range(n_chunks):
    lower_slice = chunk * n
    upper_slice = min((chunk + 1) * n, n_pargrid)
    np.savetxt(
        "params/chunk_"+str(chunk+1)+".txt",
        pargrid[lower_slice:upper_slice],
        delimiter=",", newline="\n"
    )


