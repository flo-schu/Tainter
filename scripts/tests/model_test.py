import os
#os.chdir("./Tainter/Models/tf5_cluster")
import numpy as np
import random as random
import pandas as pd
import matplotlib.pyplot as plt
import tainter_plots as t_plot
import tainter_function_blocks as tblocks
from tainter_function_5_0 import tf5 as tainter_explore
import pickle
import time as time


folder = time.strftime("%Y%m%d_%H%M")
os.mkdir("./results/model/"+folder)
#foldername = input("Foldername")
again = "y"
while again == "y":
    random.seed(1)
    np.random.seed(1)
    expl = float(input("set exploration [0,1]"))
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
                              eff = 1.0 ,
                              death_energy_level = 0.25,
                              print_every = None)

    print(t+1, wb, merun)

    plot_time = int(input("plot time (steps):"))
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


    runmode = input("save data as explore (e) or basecase (b)?")
    if runmode == "e":
        data_explore = data.copy()
        dat = pd.DataFrame(data_explore)
        dat.to_csv("./results/model/"+folder+"/t5_explore.csv")
        print("data saved to exploration")
    elif runmode == "b":
        data_base = data.copy()
        dat = pd.DataFrame(data_base)
        dat.to_csv("./results/model/"+folder+"/t5_base.csv")
        print("data saved to bascase")
    again = input("again? (y/n): ")


# save additional data
hist = pd.DataFrame(history)
hist.to_csv("./results/model/"+folder+"/history.csv")
with open("./results/model/"+folder+"/parameters.txt", "w") as text_file:
    text_file.write(str(args))
with open("./results/model/"+folder+"/network.pickle","wb") as handle:
    pickle.dump(G, handle)

print("saved all data!")
input("plot?")


fig, (ax1,ax3) = plt.subplots(2,1,sharex=True)
ax3.plot('x','Admin',ls='-',fillstyle='none', data = data_explore,
    label = "A total")
ax3.fill_between(data_explore['x'],data_explore['A_pool_shk'],
    label = "A_shock",alpha = .5)
ax3.fill_between(data_explore['x'],data_explore['A_pool_exp']+data_explore['A_pool_shk'],
    data_explore['A_pool_shk'],
    label = "A_explore",alpha = .5)
ax4 = ax3.twinx()
ax4.plot(data_explore['x'], data_explore['Ecap'],
    label = "Energy per capita", c = "g", ls = "-", lw = .5)

ax3.set_ylim(0,1)
ax2= ax1.twinx()
ax1.plot(data_base['x'],data_base['Admin'],ls='-',fillstyle='none')

ax1.fill_between(data_base['x'],data_base['A_pool_shk'],
    alpha = .5)
ax1.fill_between(data_base['x'],data_base['A_pool_exp']+data_base['A_pool_shk'],
    data_base['A_pool_shk'],
    alpha = .5)
ax2.plot(data_base['x'], data_base['Ecap'], c = "g", ls = "-", lw = .5)

ax1.set_ylim(0,1)
plt.setp(ax1.get_xticklabels(), visible=False)
ax4.set_ylim(0,max(np.append(data_base["Ecap"],data_explore['Ecap']))+.1)
ax2.set_ylim(0,max(np.append(data_base["Ecap"],data_explore['Ecap']))+.1)
fig.legend(ncol = 4)
fig.text(0.03, 0.5, 'Administrator share', ha='center',
    va='center', rotation='vertical')
fig.text(0.97, 0.5, 'Energy per capita', ha='center',
    va='center', rotation='vertical')
fig.text(0.5, 0.04, 'Time', ha='center', va='center')
plt.show()
fig.savefig("./results/model/"+folder+"/Admin_Ecap_twocases.png")
