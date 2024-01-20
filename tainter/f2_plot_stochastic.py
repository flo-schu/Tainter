import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from tainter.model.simulation import tainter
import tainter.model.methods as tm
from scipy.optimize import minimize, basinhopping
# choose which runs to plot ----------------------------------------------------
# directories where simulation results are stored which are to be compared

def fig2_stochastic_simulations(
    seed=None,
    exploration_setups=[],
    plot_time=5000,
    network_size=400, 
    rho=0.02,
    elasticity_l=0.75,
    elasticity_c=0.75,
    productivity_c=1.05,
    shock_alpha=15,
    shock_beta=1,
    **simulation_kwargs
):
    np.random.seed(seed)
    random.seed(seed)

    results = []
    for explo in exploration_setups:
        history, t, args, fct, merun, wb, G = tainter(
            # with erdos reny networks it is possible that some nodes are not 
            # linked and the model runs for ever
            network = "erdos" , 
            N = network_size,
            k = 0,
            p = rho,
            layout = "fixed",
            first_admin = "highest degree" ,
            choice = "topcoordinated",
            exploration = explo,
            shock_alpha=shock_alpha,
            shock_beta=shock_beta,
            tmax = 10000,
            death_energy_level = 0.0,
            print_every = None,
            elasticity_l=elasticity_l,
            elasticity_c=elasticity_c,
            productivity_c=productivity_c,
            **simulation_kwargs
        )


        # shorten history
        history_b = history.copy()
        history = {key: value[0:plot_time] for key, value in history_b.items()}

        data = tm.disentangle_admins(history, network_size)

        results.append(pd.DataFrame(data))
        print(f"executed scenario with exploration = {explo}")


    # set up figure ----------------------------------------------------------------
    cmap_a = cm.get_cmap("Blues", 4)
    cmap_e = cm.get_cmap("Oranges", 4)
    textwidth = 12.12537
    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(textwidth, textwidth / 2))
    gs =  fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(gs[0,0:2])
    ax2 = ax1.twinx()
    ax1b= fig.add_subplot(gs[0,2])
    ax3 = fig.add_subplot(gs[1,0:3])
    ax4 = ax3.twinx()
    ax5 = fig.add_subplot(gs[2,0:3])
    ax6 = ax5.twinx()

    # plot figure ------------------------------------------------------------------
    for i, (ax, d, explo_) in enumerate(
        zip([(ax1, ax2), (ax3, ax4), (ax5, ax6)], results, exploration_setups)):
        ax[0].plot(np.array(d['x']), np.array(d['Admin']), ls='-', fillstyle='none',
                color=cmap_a(3), label="$A_{total}$")
        ax[0].fill_between(np.array(d['x']), np.array(d['A_pool_shk']), alpha=.5,
                        color=cmap_a(3), label="$A_{shock}$")
        ax[0].fill_between(np.array(d['x']),
                        np.array(d['A_pool_exp']) + np.array(d['A_pool_shk']),
                        np.array(d['A_pool_shk']),
                        alpha=.25, color=cmap_a(3), label="$A_{exploration}$")
        ax[1].plot(np.array(d['x']), np.array(d['Ecap']), ls="-", lw=.5,
                color=cmap_e(2), label="$E_{prodcution}$")
        ax[0].set_ylim(0, 1.25)
        ax[1].set_ylim(0, 1.25)
        ax[0].annotate(f"$p_{{e}}$ = {explo_}", xy=(0.995, 0.88), xycoords="axes fraction",horizontalalignment="right")

        if i == 0:
            ax[0].set_xlim(0, plot_time * 0.65)
        else:
            ax[0].set_xlim(0, plot_time)

    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1b.set_xticks([])
    ax1b.set_yticks([])
    ax1b.set_xlabel('administration')
    ax1b.set_ylabel('energy')
    plt.setp(ax3.get_xticklabels(), visible=False)

    fig.text(0.01, 0.5, 'Administrator share', ha='center',
            va='center', rotation='vertical')
    fig.text(0.99, 0.5, 'Energy per capita', ha='center',
            va='center', rotation='vertical')
    fig.text(0.5, 0.01, 'Time', ha='center', va='center')
    fig.subplots_adjust(bottom=0.08, left=0.06, right=0.94, top=.95)

    ahandles, alabels = ax1.get_legend_handles_labels()
    ehandles, elabels = ax2.get_legend_handles_labels()

    handles = [ahandles[0], ehandles[0], ahandles[1], ahandles[2]]
    labels = [alabels[0], elabels[0], alabels[1], alabels[2]]
    ax3.legend(handles, labels, loc="right", ncol=2, frameon=True, fancybox=False)

    ax2.annotate("A", xy=(0.005, 0.88), xycoords="axes fraction", zorder=4)
    ax1b.annotate("B", xy=(0.94, 0.88), xycoords="axes fraction", zorder=4)
    ax3.annotate("C", xy=(0.005, 0.88), xycoords="axes fraction", zorder=4)
    ax5.annotate("D", xy=(0.005, 0.88), xycoords="axes fraction", zorder=4)


    # ------------------------------------------------------------------------------
    # plot stylized diminishing marginal returns

    # compute moving average of empirical ecap
    window = 11
    ecap = np.convolve(np.array(results[0]['Ecap']), np.ones(window), 'valid') / window
    # trim because of convolution
    cutoff = int(np.floor(window/2))
    admin = np.array(results[0]['Admin'])[cutoff:len(results[0])-cutoff]  

    # only select the initial phase:
    idx_init = np.where(ecap-1 > -0.1)[0]
    ecap_init = ecap[idx_init]
    admin_init = admin[idx_init]

    # plot observed data
    ax1b.plot(admin_init, ecap_init, 'o', lw=1,
            color=cmap_e(2), label="$E_{prodcution}$", alpha=.5)
    ax1b.set_ylim(ecap_init.min(), ecap_init.max())
    ax1b.set_xlim(admin_init.min(), admin_init.max())

    # theoretical complexity (ranges between 0 and 1 like administrator share)
    complexity = np.linspace(0,admin_init.max(), num=len(idx_init))

    # compute theoretical diminishing returns
    def returns_on_complexity(complexity, shift, gains, losses, offset):
        return (
            gains * (complexity + shift) * 1 +    # gains through complexity inc.
            losses * (complexity + shift) ** 2 +  # losses through compelxity inc.
            offset
        )


    def loss(X):
        shift, gains, losses, offset = X
        roc = returns_on_complexity(complexity, shift, gains, losses, offset)

        return np.sum((roc - ecap_init) ** 2)
        

    res = basinhopping(loss, (0.1, 1.1,0.1,1))
    
    roc = returns_on_complexity(complexity, *res.x)
    roc = returns_on_complexity(complexity, 0.1, 3.6, -11.5, 0.8)
    ax1b.plot(complexity, roc, color="black", lw=1.5)


    return fig
    # save -------------------------------------------------------------------------

if __name__ == "__main__":
    exploration_scenarios = [0, 0.00275,  0.02]  # must always be a list of 3
    fig = fig2_stochastic_simulations(exploration_setups=exploration_scenarios)