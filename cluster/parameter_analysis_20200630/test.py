# created by Florian Schunck on 09.07.2020
# Project: Tainter
# Short description of the feature:
# 
# 
# ------------------------------------------------------------------------------
# Open tasks:
# TODO:
# 
# ------------------------------------------------------------------------------
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
from matplotlib import pyplot as plt
from scipy import stats as st

sys.path.append('../../scripts/helpers/')
from shifted_cmap import shiftedColorMap

# data input and processing for upper subplots ---------------------------------
print("Hello! Starting import...")
data = np.loadtxt("output.txt", delimiter=",", skiprows=1)
colnames = np.array(["p_e", "rho", "phi", "te", "st"])
print("import complete")
