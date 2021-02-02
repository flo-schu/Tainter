import os
import pandas as pd
import model.tainter_function_blocks as tblocks
from model.tainter_function_5_0 import tf5 as tainter_explore
from helpers.manage import icreate_folder_date

N = 400
scenarios = [[0, "b"], [0.00275, "i"], [0.02, "e"]]
iterations = 3

plot_time = 5000
path = icreate_folder_date("../data/model")

# loop for repeting the same thing over and over again
for i in range(iterations):
    print(i)

    # loop through exploration settings
    for s in range(len(scenarios)):
        expl = scenarios[s][0]
        history, t, args, fct, merun, wb, G = tainter_explore(
            network = "erdos" , 
            N = N,
            k = 0,
            p = 0.02,
            layout = "fixed",
            first_admin = "highest degree" ,
            choice = "topcoordinated",
            exploration = expl,
            a = 1.0 ,
            stress = ["off"] ,
            shock = ["on","beta",[1,15]],
            tmax = 5000,
            threshold =1.0 ,
            eff = 1.05 ,
            death_energy_level = 0.0,
            print_every = None
        )

        print(expl, t+1, wb, merun)

        history_b = history.copy()
        history = {key: value[0:plot_time] for key, value in history_b.items()}

        data = tblocks.disentangle_admins(history, N)

        runmode = scenarios[s][1]

        if runmode == "e":
            data_explore = data.copy()
            dat = pd.DataFrame(data_explore)
            dat.to_csv(os.path.join(path, "t5_explore"+str(i)+".csv"))
        elif runmode == "i":
            data_intermediate = data.copy()
            dat = pd.DataFrame(data_intermediate)
            dat.to_csv(os.path.join(path, "t5_intermediate"+str(i)+".csv"))
        elif runmode == "b":
            data_base = data.copy()
            dat = pd.DataFrame(data_base)
            dat.to_csv(os.path.join(path, "t5_base"+str(i)+".csv"))


with open(os.path.join(path, "parameters.txt"), "w") as f:
    f.write(str(args))

print("saved all data!")
