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


# Data params
N = 400  # Network size
epsilon = 1  # threshold
rho = 0.2  # link density in erdos renyi network
phi = 1.18  # efficiency of coordinated Workers
beta = 15  # scale parameter of beta distribution
alpha = 1  # location parameter of beta distribution

base_folder = "./results/model/"
folder = "20191121_1732"
sim_admin = ['base_admin.npy', 'inter_admin.npy', 'expl_admin.npy']
sim_encap = ['base_ecap.npy', 'inter_ecap.npy', 'expl_ecap.npy']
lines = ["solid", "dashed", "dashdot"]
colors = [1, 2, 3]
pe_plot = [0.002, 0.0002]
labels = ["$p_{e}$ = " + str(i) for i in pe_plot]
t = np.linspace(0, 10000, 10001)

# plot params
textwidth = 12.12537
plt.rcParams.update({'font.size': 14})
cmap_a = cm.get_cmap("Blues", 3)
cmap_b = cm.get_cmap("Oranges", 4)
cmap_c = cm.get_cmap("Greys", 4)
cv = dict(p0=14, p1=0, p2=2)
alpha_sim = .05
fig, ax1 = plt.subplots(1, 1, figsize=(textwidth, textwidth / 4))
ax1.set_ylabel("admin share")
ax1.set_ylim(0, 1)
ax1.set_xlim(0, 10000)
ax1.set_xlabel("time")

# ax1:
for c, lab, line, pe_it in zip(colors, labels, lines, pe_plot):
    st, te, admin, ecap = integrate_fa(t, p_e=pe_it)
    ax1.plot(t, admin / N, "-", color="white", linewidth=3, alpha=.5)
    ax1.plot(t, admin / N, linestyle=line, color=cmap_a(c), linewidth=1.5, label=lab)

ax1.legend(loc="center", bbox_to_anchor=(.5, .25), ncol=3, frameon=False)
plt.subplots_adjust(left=0.07, right=0.97, bottom=0.17, top=0.98)
plt.savefig(base_folder + folder + "/pub_figure3_b.pdf")
plt.savefig(base_folder + folder + "/pub_figure3_b.png", dpi=600)
plt.show()
