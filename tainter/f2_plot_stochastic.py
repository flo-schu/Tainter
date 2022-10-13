import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from model.main import tainter
import model.methods as tm

# choose which runs to plot ----------------------------------------------------
# directories where simulation results are stored which are to be compared
mri_setups = [
    ("increasing_greater_coordinated", dict(elast_l=1.0, elast_lc=1.05, eff_lc=1.0)),
    ("decreasing_equal", dict(elast_l=.8, elast_lc=.8, eff_lc=2)),
    ("decreasing_higher_coordinated", dict(elast_l=.8, elast_lc=.85, eff_lc=1.5)),
    ("decreasing_lower_coordinated", dict(elast_l=0.95, elast_lc=0.9, eff_lc=1.2))
]
scenarios = [
    [0.0, "base"], [0.00275, "exploration low"],  [0.02, "exploration high"]
]
N = 400
plot_time = 5000
np.random.seed(8975)

for mri_name, mri_params in mri_setups:
    results = []
    for explo, sname in scenarios:
        history, t, args, fct, merun, wb, G = tainter(
            # with erdos reny networks it is possible that some nodes are not 
            # linked and the model runs for ever
            network = "erdos" , 
            N = N,
            k = 0,
            p = 0.02,
            layout = "fixed",
            first_admin = "highest degree" ,
            choice = "topcoordinated",
            exploration = explo,
            a = 1.0 ,
            stress = ["off"] ,
            shock = ["on","beta",[1,15]],
            tmax = 10000,
            threshold = 0.5 ,
            death_energy_level = 0.0,
            print_every = None,
            **mri_params
        )


        # shorten history
        history_b = history.copy()
        history = {key: value[0:plot_time] for key, value in history_b.items()}

        data = tm.disentangle_admins(history, N)

        results.append(pd.DataFrame(data))
        print(f"executed scenario {mri_name} with exploration = {explo}")


    # set up figure ----------------------------------------------------------------
    cmap_a = cm.get_cmap("Blues", 4)
    cmap_e = cm.get_cmap("Oranges", 4)
    textwidth = 12.12537
    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(constrained_layout=True, figsize=(textwidth, textwidth / 2))
    gs =  fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(gs[0,0:2])
    ax2 = ax1.twinx()
    ax1b= fig.add_subplot(gs[0,2])
    ax3 = fig.add_subplot(gs[1,0:3])
    ax4 = ax3.twinx()
    ax5 = fig.add_subplot(gs[2,0:3])
    ax6 = ax5.twinx()

    # plot figure ------------------------------------------------------------------
    for i, (ax, d, scenario) in enumerate(
        zip([(ax1, ax2), (ax3, ax4), (ax5, ax6)], results, scenarios)):
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
        ax[0].annotate(f"$p_{{e}}$ = {scenario[0]}", xy=(0.995, 0.88), xycoords="axes fraction",horizontalalignment="right")

        if i == 0:
            ax[0].set_xlim(0, 3250)
        else:
            ax[0].set_xlim(0, 5000)

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

    ax1.annotate("A", xy=(0.005, 0.88), xycoords="axes fraction")
    ax1b.annotate("B", xy=(0.94, 0.88), xycoords="axes fraction")
    ax3.annotate("C", xy=(0.005, 0.88), xycoords="axes fraction")
    ax5.annotate("D", xy=(0.005, 0.88), xycoords="axes fraction")


    # ------------------------------------------------------------------------------
    # plot stylized diminishing marginal returns
    # compute theoretical diminishing returns
    def output(complexity, eff, loss):
        return complexity * eff - (complexity) ** loss + 0.93

    def marginal(arr):
        return np.diff(arr)

    complexity = np.linspace(0,10, num =100)
    returns = output(complexity, 1.1, 1.11)

    # compute moving average of empirical ecap
    ecap = np.convolve(np.array(results[0]['Ecap']), np.ones(5), 'valid') / 5
    x = np.array(results[0]['Admin'])

    ax1b.plot(x[2:len(x)-2], ecap, 'o', lw=1,
            color=cmap_e(2), label="$E_{prodcution}$")
    ax1b.plot(complexity/10, returns, color="black", lw=1.5)

    ax1b.set_ylim(0.85,1.1)
    ax1b.set_xlim(0,0.3)


    # save -------------------------------------------------------------------------
    fig.savefig(os.path.join("results/plots/", f"pub_figure2_{mri_name}.png"), dpi = 65)
