import numpy as np
import pandas as pd
import os


folder  = "../data/model/20210202_2331/" 
ddir = os.listdir(folder)
dbase  = [j for j in ddir if "base" in j]
dexpl  = [j for j in ddir if "expl" in j]
dinter = [j for j in ddir if "intermediate" in j]

def extract_val(flist, timesteps, index):
    out = np.ndarray(shape = (timesteps,len(flist)))
    for i in np.arange(len(flist)):
        data = pd.read_csv(folder+flist[i])
        temp = np.array(data[index])
        temp = np.append(temp,np.repeat(temp[-1],timesteps-len(temp)))
        out[:,i] = temp

    return(out)

extract_time = 5000

admin_base  = extract_val(dbase, 5000, "Admin")
ecap_base   = extract_val(dbase, 5000, "Ecap")
admin_inter = extract_val(dinter, 5000, "Admin")
ecap_inter  = extract_val(dinter, 5000, "Ecap")
admin_expl  = extract_val(dexpl, 5000, "Admin")
ecap_expl   = extract_val(dexpl, 5000, "Ecap")

np.save(folder+"base_admin",  admin_base)
np.save(folder+"base_ecap",   ecap_base)
np.save(folder+"inter_admin", admin_inter)
np.save(folder+"inter_ecap",  ecap_inter)
np.save(folder+"expl_admin",  admin_expl)
np.save(folder+"expl_ecap",   ecap_expl)
