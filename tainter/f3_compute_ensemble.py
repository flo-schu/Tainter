import os
import pandas as pd
import numpy as np
import model.methods as tm
from tainter.model.simulation import tainter
from tainter.helpers.manage import icreate_folder_date

N = 400
mri_low = (
    "decreasing_lower_coordinated", 
    dict(elast_l=0.95, elast_lc=0.9, eff_lc=1.2)
)
mri_equal = (
    "decreasing_equal", 
    dict(elast_l=0.95, elast_lc=0.95, eff_lc=1.5)
)

scenarios = [[0, "b"], [0.00275, "i"], [0.02, "e"]]
iterations = 50

plot_time = 5000
path = icreate_folder_date("data/model")

np.random.seed(65421)
# loop for repeting the same thing over and over again
for i in range(iterations):
    print(i)

    # loop through exploration settings
    for s in range(len(scenarios)):
        expl = scenarios[s][0]
        history, t, args, fct, merun, wb, G = tainter(
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
            death_energy_level = 0.0,
            print_every = None,
            **mri_equal[1]
        )

        print(expl, t+1, wb, merun)

        history_b = history.copy()
        history = {key: value[0:plot_time] for key, value in history_b.items()}

        data = tm.disentangle_admins(history, N)

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
