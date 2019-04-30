import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


folder  = "../../results/model/20190424_0932/"
ddir = os.listdir(folder)
dbase = [j for j in ddir if "base" in j]
dexpl = [j for j in ddir if "expl" in j]

def extract_val(flist, timesteps, index):
    out = np.ndarray(shape = (timesteps,len(flist)))
    for i in np.arange(len(flist)):
        data = pd.read_csv(folder+flist[i])
        temp = np.array(data[index])
        temp = np.append(temp,np.repeat(temp[-1],timesteps-len(temp)))
        out[:,i] = temp

    return(out)

admin_base = extract_val(dbase, 1500, "Admin")
ecap_base  = extract_val(dbase, 1500, "Ecap")
admin_expl = extract_val(dexpl, 1500, "Admin")
ecap_expl  = extract_val(dexpl, 1500, "Ecap")

np.save(folder+"base_admin", admin_base)
np.save(folder+"base_ecap",  ecap_base)
np.save(folder+"expl_admin", admin_expl)
np.save(folder+"expl_ecap",  ecap_expl)
