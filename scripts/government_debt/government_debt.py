import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("../data/government_debt/data.csv",sep=",")
data = data.drop('Country Code',1)
debts = np.array(data, dtype=float)

mindebt = []
for i in range(len(debts)):
    sub = debts[i,:]
    subnn = sub[~np.isnan(sub)]

    if len(subnn) == 0:
        res = np.nan
    else:
        res = subnn.max()

    mindebt.extend([res])

normed = debts / np.array(mindebt)[: ,None]

np.median(normed, axis =1)
plt.plot(normed.T)