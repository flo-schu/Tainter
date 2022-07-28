import os
import pickle
import random as random
import shutil
import sys
import time as time

import matplotlib as mpl
import matplotlib.pyplot as plt
import model.methods as tm
import numpy as np
import pandas as pd
from matplotlib import cm
from model.main import tainter
from helpers.manage import icreate_folder_date

# model parameters
N = 400
scenarios = [[0, "b"], [0.2, "e"]]

# script parameters
path = "data/model/"
plot_time = 5000
# random.seed(0)

path = icreate_folder_date(path)

history, t, args, fct, merun, wb, G = tainter(
    # with erdos reny networks it is possible that some nodes are not 
    # linked and the model runs for ever
    network = "erdos" , 
    N = N,
    k = 0,
    p = 0.02,
    layout = "fixed",
    first_admin = "highest degree" ,
    choice = "topcoordinated",
    exploration = 0.2,
    a = 1.0 ,
    stress = ["off"] ,
    shock = ["on","beta",[1,15]],
    tmax = 10000,
    threshold =1.0 ,
    elast_l = 0.95 ,
    elast_lc = 0.95 ,
    eff_lc = 1.2 ,
    death_energy_level = 0.0,
    print_every = None
)


# shorten history
history_b = history.copy()
history = {key: value[0:plot_time] for key, value in history_b.items()}

data = tm.disentangle_admins(history, N)

pd.DataFrame(data).to_csv(os.path.join(path, "data.csv"))
with open(os.path.join(path, "parameters.txt"), "w") as f:
    f.write(str(args))
