# created by Florian Schunck in December 2019
# Project: tainter
# Short description of the feature:
# 1. create Plot 4 for publication. The plot displays the paramter grids in
#    terms of survival time and energy production and contrasts p_e=0 und p_e>0
#    to illustrate divergent model outcomes dependent on the degree of
#    exploration
# ------------------------------------------------------------------------------
# Open tasks:
# TODO: add lines for p_e values displayed in grids
#
# ------------------------------------------------------------------------------

import sys

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats as st

# sys.path.append('../../scripts/helpers/')
# from shifted_cmap import shiftedColorMap

# data input and processing for upper subplots ---------------------------------
print("Hello! Starting import...")
data = np.loadtxt("./cluster/parameter_analysis_20200630/output_corrected.txt", delimiter=",", skiprows=1)
colnames = np.array(["p_e", "rho", "phi", "te", "st"])
print("Import complete")

p_e = data[:, 0]
rho = data[:, 1]
phi = data[:, 2]
npe = np.unique(p_e)
nrho = np.unique(rho)
nphi = np.unique(phi)

print("import complete")


print(len(data), len(npe), len(nrho), len(nphi))