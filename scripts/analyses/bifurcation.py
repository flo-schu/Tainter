import numpy as np
import scipy.stats as sci
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.integrate import odeint, trapz


def f_a(x, t, N, p_e, epsilon, rho, phi, beta, alpha):
    if x >= N:
        return 0
    else:
        return (
                p_e * (N - 2 * x) +
                sci.beta.cdf(
                    (((epsilon * N) / (((N - x) * (1 - rho) ** x) +
                                       ((N - x) * (1 - (1 - rho) ** x)) ** phi))),
                    a=beta, b=alpha)
        )


def f_e(x, N, rho, phi):
    return (
            ((N - x) * (1 - rho) ** x + ((N - x) * (1 - (1 - rho) ** x)) ** phi) / N
    )


def get_st(t, e):
    if all(np.round(e, 6) > 0):
        return int(np.max(t))
    else:
        return int(np.where(np.round(e, 6) == 0)[0][0] + 1)


def integrate_fa(t, p_e):
    result = odeint(f_a, y0=0, t=t, args=(N, p_e, epsilon, rho, phi, beta, alpha)).flatten()
    result[result > N] = N  # turn all x > N to N (fix numerical issue)
    e = f_e(result, N, rho, phi)
    st = get_st(t, e)
    te = trapz(e[:st], t[:st])
    return st, te, result, e



N = 400  # Network size
epsilon = 1  # threshold
rho = 0.2  # link density in erdos renyi network
phi_arr = np.linspace(1.18257,1.1826,100)
beta = 15  # scale parameter of beta distribution
alpha = 1  # location parameter of beta distribution
t = np.linspace(0, 10000, 100001)

max_a = np.zeros(100)

for i in range(len(phi_arr)):
# phi = 1.15  # efficiency of coordinated Workers
    phi = phi_arr[i]
    st, te, admin, ecap = integrate_fa(t, p_e=0.002)
    max_a[i] = admin.max() / N

xbif = phi_arr[np.where(max_a < 1)[0][0]-1]


textwidth = 12.12537
plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(textwidth, textwidth / 3))
plt.plot(phi_arr, max_a)
plt.hlines(0.5, 1,2, linewidth=.5, linestyle="--")
plt.vlines(xbif, ymin=.5,ymax=1, linestyle="--")
plt.xlim(1.18257,1.1826)
plt.ylim(0.5,1.025)
plt.xlabel("output elasticity $\phi$")
plt.ylabel("max. administrator share")
# plt.plot(xbif,1, 'o', color="white")
plt.plot(xbif,1, 'o', alpha = 0.25, color="black")
plt.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.98)
plt.savefig("./plots/bifurcation.png")
plt.show()