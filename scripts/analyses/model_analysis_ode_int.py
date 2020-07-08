import numpy as np
import scipy.stats as sci
from matplotlib import pyplot as plt
from matplotlib import cm

from scipy.integrate import ode, odeint

# Parameters -------------------------------------------------------------------
N = 400  # Network size
epsilon = 1  # threshold
rho = 0.02  # link density in erdos renyi network
phi = 1.05  # efficiency of coordinated Workers
beta = 15  # scale parameter of beta distribution
alpha = 1  # location parameter of beta distribution


# ODEs -------------------------------------------------------------------------
def f_a(x, t, N, p_e, epsilon, rho, phi, beta, alpha):
    if x >= N:
        factor = 0
    else:
        factor = sci.beta.cdf(
            (((epsilon * N) / (((N - x) * (1 - rho) ** x) +
                               ((N - x) * (1 - (1 - rho) ** x)) ** phi))),
            a=beta, b=alpha)

    return p_e * (N - 2 * x) + factor


def f_e(x, N, rho, phi):
    x = np.array([N if i >= N else i for i in x])

    return (
            ((N - x) * (1 - rho) ** x + ((N - x) * (1 - (1 - rho) ** x)) ** phi) / N
    )


# def get_timeseries(timestep, tmax, initial_value):
#     r = ode(f_a)
#     r.set_initial_value(initial_value)
#     r.set_f_params(N, p_e, epsilon, rho, phi, beta, alpha)
#     r.set_integrator("vode")
#
#     t = [0]
#     results = [initial_value]
#
#     while r.t < tmax and r.successful():
#         r.integrate(r.t + timestep)
#         t.append(r.t)
#         results.append(r.y)
#     return t, results


##################################################
p_e = 0.0  # base

results = odeint(f_a, y0=0, t=np.linspace(0, 1500, 1500),
                 args=(N, p_e, epsilon, rho, phi, beta, alpha))

print(results)

tb, xb = get_timeseries(1, 1500, 0)
# xb2 = np.mean(np.array(xb[0:150000]).reshape(-1,100),axis = 1)
# tb2 = np.arange(1500)
# print(xb[len(xb)-10:len(xb)])
# plt.plot(tb,np.array(xb)/N)
# plt.plot(tb,f_e(np.array(xb), N, rho, phi))
# plt.show()
# print(f_e(np.array(xb), N, rho, phi))
p_e = 0.02  # explore
te, xe = get_timeseries(1, 1500, 0)

# len(te)

# folder = input("folder name:")
# folder = "20190419_1134"
folder = "20190424_0932"
admin_b = np.load("../../results/model/" + folder + "/base_admin.npy")
ecap_b = np.load("../../results/model/" + folder + "/base_ecap.npy")

admin_e = np.load("../../results/model/" + folder + "/expl_admin.npy")
ecap_e = np.load("../../results/model/" + folder + "/expl_ecap.npy")

cmap = cm.get_cmap("tab20", 20)
fig, (ax1, ax3) = plt.subplots(2, 1, sharex=True)
ax2 = ax1.twinx()
ax4 = ax3.twinx()

# simulation
for i in np.arange(100):
    ax1.plot(admin_b[:, i], label="$A_{simulation}$" if i == 0 else "",
             color=cmap(1), alpha=.1)
for i in np.arange(100):
    ax2.plot(ecap_b[:, i], label="$E_{simulation}$" if i == 9 else "",
             alpha=.1, color=cmap(3))
ax3.plot(admin_e, color=cmap(1), alpha=.1)
ax4.plot(ecap_e, alpha=.1, color=cmap(3))

# analytic
ax1.plot(tb, np.array(xb) / N, label="$A_{analytic}$", color=cmap(0))
ax2.plot(tb, f_e(np.array(xb), N, rho, phi),
         label="$E_{analytic}$", color=cmap(2))

ax3.plot(te, np.array(xe) / N, color=cmap(0))
ax4.plot(te, f_e(np.array(xe), N, rho, phi),
         color=cmap(2))

fig.text(0.05, 0.5, 'Administrator share', ha='center',
         va='center', rotation='vertical')
fig.text(0.98, 0.5, 'Energy per capita', ha='center',
         va='center', rotation='vertical')
fig.text(0.5, 0.04, 'Time', ha='center', va='center')
ax1.annotate("A", xy=(0.01, 0.9), xycoords="axes fraction")
ax3.annotate("B", xy=(0.01, 0.9), xycoords="axes fraction")
# handles, labels = plt.gca().get_legend_handles_labels()
# print(labels)
# by_label = OrderedDict(zip(labels, handles))
# print(by_label)
# fig.legend(by_label.values(), by_label.keys(),
#     loc = "center", bbox_to_anchor = (.75,.7),ncol = 1)
fig.legend(loc="center", bbox_to_anchor=(.75, .7), ncol=1)
plt.savefig("../../results/model/" + folder + "/comp_integration-model_exploration.png")
plt.show()
