import os
import sys
sys.path.append('../model/')

# os.chdir("./Tainter/Models/tf5_cluster")
import numpy as np
import shutil
import random as random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import tainter_plots as t_plot
import tainter_function_blocks as tblocks
from tainter_function_5_0 import tf5 as tainter_explore
import pickle
import time as time

specs = [[0,1500,"b","y"],[0.02,1500,"e","n"]]
iter = 0
folder = time.strftime("%Y%m%d_%H%M")
if os.path.isdir("../../results/model/"+folder):
    if input("dir exists. overwrite? (y/n) ") == "y":
        shutil.rmtree("../../results/model/"+folder)
        os.mkdir("../../results/model/"+folder)
else:
    os.mkdir("../../results/model/"+folder)


print(folder)
input("press enter to continue")

#foldername = input("Foldername")
again = "y"
while again == "y":
    random.seed(4)
    np.random.seed(4)
    # expl = float(input("set exploration [0,1]"))
    expl = specs[iter][0]
    N = 400
    history, t, args, fct, merun, wb, G = tainter_explore(
                              network = "erdos" , # with erdos reny networks it is possible that some nodes are not linked and the model runs for ever
                              N = N,
                              k = 0,
                              p = 0.02,
                              layout = "fixed",
                              first_admin = "highest degree" ,
                              choice = "topcoordinated",
                              # popmode = "conditional",
                              # linkmode = "off",
                              # death = 0.0,
                              exploration = expl,
                              a = 1.0 ,
                              stress = ["off"] ,
                              shock = ["on","beta",[1,15]],
                              tmax = 10000,
                              threshold =1.0 ,
                              eff = 1.05 ,
                              death_energy_level = 0.0,
                              print_every = None)

    print(t+1, wb, merun)

    # plot_time = int(input("plot time (steps):"))
    plot_time = specs[iter][1]
    history_b = history.copy()
    if plot_time > 0:
        history = {key: value[0:plot_time] for key, value in history_b.items()}

    A_tot = [list(i[0]) for i in history["Administrators"]]
    A_exp = history['Aexpl']
    A_shk = list()
    A_exp_add = list()
    A_exp_rem = list()
    A_null = list()
    A_shk2 = history['Ashk']
    A_shk2 = [[] if i[0] is None else i for i in A_shk2 ]
    A_shk3 = list()
    A_shk4 = set()
    A_shk4_size = list()
    A_exp_temp = set()
    A_exp_size = list()

    for i in np.arange(0, len(A_tot)):
        A_exp_add.append([j for j in A_exp[i][0]])
        A_exp_rem.append([j for j in A_exp[i][1]])

    # calculate the administrators created by the mechanism
    for i in np.arange(0, len(A_tot)):
        A_shk.append(list(set(A_tot[i]).difference(set(A_tot[i-1])).difference(A_exp[i][0])))
        A_shk4 = A_shk4.difference(A_exp_rem[i])
        A_shk4 = A_shk4.union(A_shk2[i])
        A_shk4_size.append(len(A_shk4))
        A_exp_temp = A_exp_temp.union(set(A_exp_add[i]))
        A_exp_temp = A_exp_temp.difference(set(A_exp_rem[i]))
        A_exp_size.append(len(A_exp_temp))



    data = {'x': np.arange(len(A_shk)),
            'Admin': np.array([len(i[0]) for i in history['Administrators']])/N,
            'A_shk_c': np.array([len(i) for i in A_shk])/N,
            'A_shk': np.array([len(i) for i in A_shk2])/N,
            'A_pool_shk': np.array(A_shk4_size)/N,
            'A_pool_exp': np.array(A_exp_size)/N,
            'A_exp_add': np.array([len(i) for i in A_exp_add])/N,
            'A_exp_rem': np.array([len(i) for i in A_exp_rem])/N,
            'A_exp': np.array([len(i) for i in A_exp_add]) -
            np.array([len(i) for i in A_exp_rem])/N,
            'Ecap': np.array(history['Energy per capita']),
            'Access': np.array(history['access'])}


    # runmode = input("save data as explore (e) or basecase (b)?")
    runmode = specs[iter][2]
    if runmode == "e":
        data_explore = data.copy()
        dat = pd.DataFrame(data_explore)
        dat.to_csv("../../results/model/"+folder+"/t5_explore.csv")
        print("data saved to exploration")
    elif runmode == "b":
        data_base = data.copy()
        dat = pd.DataFrame(data_base)
        dat.to_csv("../../results/model/"+folder+"/t5_base.csv")
        print("data saved to bascase")
    # again = input("again? (y/n): ")
    again = specs[iter][3]
    iter += 1



# save additional data
hist = pd.DataFrame(history)
hist.to_csv("../../results/model/"+folder+"/history.csv")
with open("../../results/model/"+folder+"/parameters.txt", "w") as text_file:
    text_file.write(str(args))
with open("../../results/model/"+folder+"/network.pickle","wb") as handle:
    pickle.dump(G, handle)

print("saved all data!")
# input("plot?")
