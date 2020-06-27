# created by Florian Schunck on 27.06.2020
# Project:
# Short description of the feature:
# 1. output processing of date from parameter analysis on EVE cluster
#
# ------------------------------------------------------------------------------
# Open tasks:
# TODO:
#
# ------------------------------------------------------------------------------

import numpy as np
import os

# in this script I use type hints to declare the type of my variables used
# for type checking for better debugging and later code analysis.

output_files = os.listdir("output/")  # returns list of files

output = np.ndarray(shape=(0, 5))
for i in range(len(output_files)):
    chunk = np.loadtxt("./output/"+output_files[i], delimiter=",", ndmin=2)
    output = np.concatenate((output, chunk), axis=0)

print(output)

colnames = "p_e, rho, phi, te, st"
np.savetxt("./output.txt", output, header=colnames, delimiter=",", newline="\n")
