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
rho = 0.02  # link density in erdos renyi network
phi = 1.05  # efficiency of coordinated Workers
beta = 15  # scale parameter of beta distribution
alpha = 1  # location parameter of beta distribution

folder = "20191121_1732"
sim_admin = ['base_admin.npy', 'inter_admin.npy', 'expl_admin.npy']
sim_encap = ['base_ecap.npy', 'inter_ecap.npy', 'expl_ecap.npy']
labels = ["$p_{e}$ = 0.0", "$p_{e}$ = 0.00275", "$p_{e}$ = 0.2"]
lines = ["dashed", "dotted", "dashdot"]
colors = [1, 2, 3]
pe_plot = [0, 0.00275, 0.02]
t = np.linspace(0, 5000, 5001)

# plot params
textwidth = 12.12537
plt.rcParams.update({'font.size': 14})
cmap_a = cm.get_cmap("Blues", 4)
cmap_b = cm.get_cmap("Oranges", 4)
cmap_c = cm.get_cmap("Greys", 4)
cv = dict(p0=14, p1=0, p2=2)
alpha_sim = .05
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(textwidth, textwidth / 2))
ax1.set_ylabel("admin share")
ax1.set_ylim(0, 1.25)
ax1.set_xlim(0, 5250)
ax1.set_xticklabels(['', '', '', '', '', ''])

ax2.set_ylabel("energy $cap^{-1}$")
ax2.set_xlabel("time")
ax2.set_ylim(0, 1.25)
ax2.set_xlim(0, 5250)

ax3.set_ylabel("frequency")
ax3.set_xlabel("time of collapse")
ax3.set_xlim(0, 5250)
ax3.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
ax3.set_xticklabels(['0', '1000', '2000', '3000', '4000', '$\\geq 5000$'])

ax1.annotate("A", xy=(0.01, 0.83), xycoords="axes fraction")
ax2.annotate("B", xy=(0.01, 0.83), xycoords="axes fraction")
ax3.annotate("C", xy=(0.01, 0.83), xycoords="axes fraction")

fig.subplots_adjust(bottom=0.1, left=0.07, right=0.97, top=.98, hspace=.1)
bbox = list(ax3.get_position().bounds)
bbox[3] = 0.9*bbox[3] # Reduce the height of the axis a bit.
ax3.set_position(bbox)
bbox = list(ax2.get_position().bounds)
bbox[1] = 1.1*bbox[1] # shift bbox upwards
bbox[3] = 0.9*bbox[3] # Reduce the height of the axis a bit.
ax2.set_position(bbox)
bbox = list(ax2.get_position().bounds)
bbox = list(ax1.get_position().bounds)
bbox[1] = 1.01*bbox[1] # shift bbox upwards
bbox[3] = 0.9*bbox[3] # Reduce the height of the axis a bit.
ax1.set_position(bbox)

# ax1:
for sa, c in zip(sim_admin, colors):
    sim_a = np.load("../../results/model/" + folder + "/" + sa)
    ax1.plot(sim_a, color=cmap_a(c), alpha=alpha_sim)

for c, lab, line, pe_it in zip(colors, labels, lines, pe_plot):
    st, te, admin, ecap = integrate_fa(t, p_e=pe_it)
    ax1.plot(t, admin / N, "-", color="white", linewidth=3, alpha=.5)
    ax1.plot(t, admin / N, linestyle=line, color=cmap_a(c), linewidth=1.5, label=lab)

# ax2:
for se, c in zip(sim_encap, colors):
    sim_e = np.load("../../results/model/" + folder + "/" + se)
    # print(sim_e[:,20])
    ax2.plot(sim_e, color=cmap_b(c), alpha=alpha_sim)

for c, lab, line, pe_it in zip(colors, labels, lines, pe_plot):
    st, te, admin, ecap = integrate_fa(t, p_e=pe_it)
    # print(ecap)
    ax2.plot(t, ecap, "-", color="white", linewidth=3, alpha=.5)
    ax2.plot(t, ecap, linestyle=line, color=cmap_b(c), linewidth=1.5, label=lab)

# ax3:
for se, c, lab in zip(sim_encap.__reversed__(), colors.__reversed__(), labels.__reversed__()):
    sim_e = np.load("../../results/model/" + folder + "/" + se)
    st_sim = np.zeros(100)
    # count timepoints where energy was produced for each of the 100 model runs
    for i in np.arange(100):
        st_sim[i] = np.sum(sim_e[:, i] != 0)
    ax3.hist(st_sim, bins=21, range=[0, 5250], color=cmap_c(c), alpha=1, label=lab)

ax1.legend(loc="center", bbox_to_anchor=(.5, .25), ncol=3, frameon=False)
ax2.legend(loc="center", bbox_to_anchor=(.5, .75), ncol=3, frameon=False)
handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles[::-1], labels[::-1], loc="center", bbox_to_anchor=(.5, .75), ncol=3, frameon=False)

plt.savefig("../../results/model/" + folder + "/pub_figure3.pdf")
plt.savefig("../../results/model/" + folder + "/pub_figure3.png", dpi=600)
plt.show()
