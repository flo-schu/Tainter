import numpy as np
import scipy.stats as sci
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
import pandas as pd
import os as os
from scipy.integrate import ode
from scipy.integrate import trapz
import itertools as it
# os.chdir("./Tainter/Models/tf5_cluster")


# Parameters ###################################################################
N   = 400       # Network size
p_e = 0.0      # exploration probability

epsilon = 1     # threshold
beta    = 15    # scale parameter of beta distribution
alpha   = 1     # location parameter of beta distribution

# Definition Equations #########################################################
def f_a(t, x, N, p_e, epsilon, rho, phi, beta, alpha):
    return(
    p_e * (N - 2 * x)  +
    sci.beta.cdf(
        (((epsilon * N) / (((N - x) * (1 - rho) ** x) +
        ((N - x) * (1 - (1 - rho) ** x)) ** phi))),
        a = beta, b = alpha)
    )

def f_s(x, N, p_e, epsilon, rho, phi, beta, alpha):
    return(
    p_e * (N - 2 * x)  +
    sci.beta.cdf(
        (((epsilon * N) / (((N - x) * (1 - rho) ** x) +
        ((N - x) * (1 - (1 - rho) ** x)) ** phi))),
        a = beta, b = alpha)
    )

def f_e(x, N, rho, phi):
    return(
    ((N - x) * (1 - rho) ** x + ((N - x) * (1 - (1 - rho) ** x)) ** phi) / N
    )

# Solver #######################################################################
def get_timeseries(timestep, tmax, initial_value, rho, phi):
    r = ode(f_a)
    r.set_initial_value(initial_value)
    r.set_f_params(N, p_e, epsilon, rho, phi, beta, alpha)
    r.set_integrator("vode")

    t = [0]
    results = [initial_value]

    while r.t < tmax and r.successful():
        r.integrate(r.t+timestep)
        t.append(r.t)
        results.append(r.y)

    return t, results

rho     = np.linspace(0,0.1,11)  # link density in erdos renyi network
phi     = np.linspace(1,1.5,11)   # efficiency of coordinated Workers
pargrid = it.product(rho, phi)

t_data = list()
x_data = list()
te_data = list()
rho_par = list()
phi_par = list()
s_data = list()

for i in pargrid:
    rho_i, phi_i = i[0], i[1]
    t, x = get_timeseries(1, 10000, 0, rho_i, phi_i)

    e = f_e(np.array(x),N,rho_i, phi_i)
    te = trapz(e,t)
    rho_par.append(rho_i)
    phi_par.append(phi_i)

    te_data.append(te)
    s_data.append(t[-1])
    print(i, te)


folder = "20190327_1730"
data = pd.DataFrame(np.array([phi_par,rho_par,te_data,s_data]).transpose())
data.columns = ["phi", "rho","te","s"]

fname = input("set filename for data without file extension)(no filename -> data not saved): ")
if len(fname) > 0:
    data.to_csv("./results/model/"+folder+"/"+fname + ".csv")

data = pd.read_csv("./results/model/"+folder+"data11x11_x1-0.csv")

# produced energy
grid = np.array(data.te).reshape((len(rho), len(phi)))
grid = np.flipud(grid.T)
plt.imshow(grid, extent= (data.rho.min(), data.rho.max(), data.phi.min(), data.phi.max()),
    interpolation = "nearest", cmap = "viridis", aspect = "auto")
plt.colorbar()
plt.xlabel("link density (rho)")
plt.ylabel("efficiency (phi)")
plt.show()
data.sort_values(["phi","rho"])
# survivaltime
grid = np.array(data.s).reshape((len(rho), len(phi)))
grid = np.flipud(grid.T)
plt.imshow(grid, extent= (data.rho.min(), data.rho.max(), data.phi.min(), data.phi.max()),
    interpolation = "nearest", cmap = "viridis", aspect = "auto")
plt.colorbar()
plt.xlabel("link density (rho)")
plt.ylabel("efficiency (phi)")
plt.show()


fig, (ax1,ax3) = plt.subplots(1,2)
for i in np.arange(0,len(data),int(np.sqrt(len(data)))):
    d = data.sort_values(["phi","rho"])[i:i+11]
    d.index = np.arange(11)
    ax1.plot(d.rho, d.te, label = "phi = "+str(d.phi[1]))

for i in np.arange(0,121,11):
    d = data[i:i+11]
    ax3.plot(d.phi, d.te, label = "rho = "+str(d.rho[i]))
ax1.legend()
ax3.legend()
plt.subplots_adjust(wspace = .5)
#plt.ylabel("produced energy per capita")
ax3.set_xlabel("efficiency (phi)")
ax1.set_xlabel("link density (rho)")
