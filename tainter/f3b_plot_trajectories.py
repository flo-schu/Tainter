import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from tainter.model.approximation import integrate_fa


def fig3b_administrator_trajectories(
    exploration_setups,
    network_size=400, 
    link_probability_between_nodes=0.02,
    mri_of_laborers=0.8,
    mri_of_coordinated_laborers=0.8,
    efficiency_of_coordinated_laborers=1.2,
    shock_alpha=1,
    shock_beta=15,

):

    fixed_params = dict(
        N = network_size,  # Network size
        p_e = 0.0,  # default value
        rho = link_probability_between_nodes,  # link density in erdos renyi network
        phi = mri_of_coordinated_laborers,  # efficiency of coordinated Workers
        psi = mri_of_laborers,
        c = efficiency_of_coordinated_laborers,
        beta = shock_alpha,  # scale parameter of beta distribution
        alpha = shock_beta  # location parameter of beta distribution
    )

    # Data params
    lines = ["solid", "dashed", "dashdot"]
    colors = [1, 2, 3]
    labels = ["$p_{e}$ = " + str(i) for i in exploration_setups]
    t = np.linspace(0, 10000, 10001)

    # plot params
    textwidth = 12.12537
    plt.rcParams.update({'font.size': 14})
    cmap_a = cm.get_cmap("Blues", 3)

    fig, ax1 = plt.subplots(1, 1, figsize=(textwidth, textwidth / 4))
    ax1.set_ylabel("admin share")
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, 10000)
    ax1.set_xlabel("time")

    # ax1:
    for c, lab, line, pe_it in zip(colors, labels, lines, exploration_setups):
        fixed_params["p_e"] = pe_it
        st, te, admin, ecap = integrate_fa(t, params=fixed_params)
        ax1.plot(t, admin / network_size, "-", color="white", linewidth=3, alpha=.5)
        ax1.plot(t, admin / network_size, linestyle=line, color=cmap_a(c), linewidth=1.5, label=lab)

    ax1.legend(loc="center", bbox_to_anchor=(.5, .25), ncol=3, frameon=False)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.17, top=0.98)
    
    return fig


if __name__ == "__main__":
    fig3b_administrator_trajectories([0.002, 0.0002])