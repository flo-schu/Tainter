import os
import pickle
import random as random
import shutil
import sys
import time as time

import matplotlib as mpl
import matplotlib.pyplot as plt
import model.tainter_function_blocks as tblocks
import model.tainter_plots as t_plot
import numpy as np
import pandas as pd
from matplotlib import cm
from model.tainter_function_5_0 import tf5 as tainter_explore
from helpers.manage import icreate_folder_date

# model parameters
N = 400
scenarios = [[0, "b"], [0.02, "e"]]

# script parameters
path = "../data/model/"
plot_time = 1500
random.seed(4)

path = icreate_folder_date(path)

for i in range(len(scenarios)):
    s = scenarios[i][0]
    history, t, args, fct, merun, wb, G = tainter_explore(
        # with erdos reny networks it is possible that some nodes are not 
        # linked and the model runs for ever
        network = "erdos" , 
        N = N,
        k = 0,
        p = 0.02,
        layout = "fixed",
        first_admin = "highest degree" ,
        choice = "topcoordinated",
        exploration = s,
        a = 1.0 ,
        stress = ["off"] ,
        shock = ["on","beta",[1,15]],
        tmax = 10000,
        threshold =1.0 ,
        eff = 1.05 ,
        death_energy_level = 0.0,
        print_every = None
    )

    print(t+1, wb, merun)

    # shorten history
    history_b = history.copy()
    history = {key: value[0:plot_time] for key, value in history_b.items()}

    data = tblocks.disentangle_admins(history, N)

    runmode = scenarios[i][1]
    if runmode == "e":
        data_explore = data.copy()
        dat = pd.DataFrame(data_explore)
        dat.to_csv(os.path.join(path, "tainter_explore.csv"))
        with open(os.path.join(path, "explore_parameters.txt"), "w") as f:
            f.write(str(args))

    elif runmode == "b":
        data_base = data.copy()
        dat = pd.DataFrame(data_base)
        dat.to_csv(os.path.join(path, "tainter_base.csv"))
        with open(os.path.join(path, "base_parameters.txt"), "w") as f:
            f.write(str(args))

