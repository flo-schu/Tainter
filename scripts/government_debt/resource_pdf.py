import numpy as np
import matplotlib.pyplot as plt

a = 15
b = 1
r = np.linspace(0,1,20000)
rpdf = a*b*r**(a-1)

plt.plot(r,rpdf)
plt.show()

rpdf.sum()