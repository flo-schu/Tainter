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
import re
import os
import pickle

# in this script I use type hints to declare the type of my variables used
# for type checking for better debugging and later code analysis.

params_files = os.listdir("../parameter_analysis_20200630/params/")
output_files = os.listdir("output/")  # returns list of files

par = sorted([int(re.findall('[0-9]+', i)[0]) for i in params_files])
out = sorted([int(re.findall('[0-9]+', i)[0]) for i in output_files])

print(par)
print(out)

print(len(par), len(out))
missing = [p for p in par if p not in out]

print(missing)

if len(params_files) == len(output_files):
    pass
else:
    raise ValueError("not as many output as params files exist")

output = np.ndarray(shape=(0, 5))

with open("output.pkl", "ab") as f:
    for i in range(len(output_files)):
        print(output_files[i])
        chunk = np.loadtxt("./output/"+output_files[i], delimiter=",", ndmin=2)
        chunk = chunk[:, 0:5]
        pickle.dump(chunk, f)
        # output = np.concatenate((output, chunk), axis=0)

# colnames = "p_e, rho, phi, te, st"
# np.savetxt("./output.txt", output, header=colnames, delimiter=",", newline="\n")
# np.save("output", output, allow_pickle=True)