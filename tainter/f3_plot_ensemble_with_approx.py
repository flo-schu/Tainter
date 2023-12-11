import os
import textwrap
import json
import random
import warnings
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm

import tainter.model.methods as tm
from tainter.model.simulation import tainter
from tainter.model.approximation import integrate_fa, get_st


def fig3_stochastic_ensemble_and_macroscopic_approximation(
    seed,
    exploration_setups,
    ensemble_data="./data/ensemble/",
    use_pre_simulated_data=False,
    iterations=2,
    plot_time=5000,
    network_size=400, 
    rho=0.02,
    elasticity_l=0.8,
    elasticity_c=0.8,
    productivity_c=1.2,
    shock_alpha=1,
    shock_beta=15,
    **simulation_kwargs

):
    if not use_pre_simulated_data:
        np.random.seed(seed)
        random.seed(seed)
        if os.path.exists(ensemble_data):
            pre_simulated_files = os.listdir(ensemble_data)
            if len(pre_simulated_files) > 0:
                warnings.warn(textwrap.dedent("""
                    you are about to overwrite pre-simulated data. If you continue,
                    all pre-simulated data will be deleted. Press ('y' to do so).
                    If you press 'n' (or 'enter'), the function will stop,
                    please restart with a new argument 'ensemble_data'.
                """))
                while True:
                    answer = input("\noverwrite? (y/n)")
                    if answer == "y":
                        [os.remove(os.path.join(ensemble_data, f)) 
                            for f in pre_simulated_files]
                        break
                    elif answer == "n" or answer == "":
                        raise FileExistsError("specify new argument 'ensemble_data'")

        else:
            os.makedirs(ensemble_data)


        # loop for repeting the same thing over and over again
        with tqdm(total=iterations * len(exploration_setups)) as pbar:
            for i in range(iterations):

                # loop through exploration settings
                for exploration_rate in exploration_setups:

                    # with erdos reny networks it is possible that some nodes 
                    # are not linked and the model runs for ever
                    history, t, args, fct, merun, wb, G = tainter(
                        network = "erdos" , 
                        N = network_size,
                        k = 0,
                        p = rho,
                        layout = "fixed",
                        first_admin = "highest degree" ,
                        choice = "topcoordinated",
                        exploration = exploration_rate,
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

                    # print(textwrap.dedent(f"""
                    #     exploration_rate: {exploration_rate}
                    #     survival time: {t+1} 
                    #     wellbeing: {wb} 
                    #     energy maximum: {merun}
                    #     """
                    # ), end="\r")

                    history_b = history.copy()
                    history = {key: value[0:plot_time] for key, value in 
                        history_b.items()}

                    data = tm.disentangle_admins(history, network_size)

                    data_base = data.copy()
                    dat = pd.DataFrame(data_base)
                    dat.to_csv(os.path.join(
                        ensemble_data, f"exploration_{exploration_rate}_{i}.csv"
                    ))
                    pbar.update(1)

        with open(os.path.join(ensemble_data, "parameters.json"), "w") as f:
            json.dump(args, f, indent=4)

    # Data params
    fixed_params = dict(
        N = network_size,  # Network size
        p_e = 0.0,  # default value
        rho = rho,  # link density in erdos renyi network
        phi = elasticity_c,  # efficiency of coordinated Workers
        psi = elasticity_l,
        c = productivity_c,
        beta = shock_beta,  # scale parameter of beta distribution
        alpha = shock_alpha  # location parameter of beta distribution
    )


    # create the plot

    lines = ["dashed", "dotted", "dashdot"]
    colors = [1, 2, 3]
    labels = ["$p_{e}$ = " + str(i) for i in exploration_setups]
    t = np.linspace(0, plot_time, plot_time + 1)

    # plot params
    textwidth = 12.12537
    plt.rcParams.update({'font.size': 14})
    cmap_a = cm.get_cmap("Blues", 4)
    cmap_b = cm.get_cmap("Oranges", 4)
    cmap_c = cm.get_cmap("Greys", 4)

    alpha_sim = .05
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(textwidth, textwidth / 2))
    ax1.set_ylabel("admin share")
    ax1.set_ylim(0, 1.25)
    ax1.set_xlim(0, plot_time * 1.05)
    ax1.set_xticklabels(['', '', '', '', '', ''])

    ax2.set_ylabel("energy $cap^{-1}$")
    ax2.set_xlabel("time")
    ax2.set_ylim(0, 1.25)
    ax2.set_xlim(0, plot_time * 1.05)

    ax3.set_ylabel("frequency")
    ax3.set_xlabel("time of collapse")
    ax3.set_xlim(0, plot_time * 1.05)
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

    ddir = os.listdir(ensemble_data)
    ddir = [f for f in ddir if ".csv" in f]
    surv_time = []
    for exploration_rate, c, lab, line in zip(exploration_setups, colors, labels, lines):
        datafiles  = [j for j in ddir if j.split("_")[1] == str(exploration_rate)]

        def extract_val(flist, timesteps, index):
            out = np.ndarray(shape = (timesteps,len(flist)))
            for i in np.arange(len(flist)):
                data = pd.read_csv(os.path.join(ensemble_data, flist[i]))
                temp = np.array(data[index])
                temp = np.append(temp,np.repeat(temp[-1],timesteps-len(temp)))
                out[:,i] = temp

            return(out)

        sim_admin = extract_val(datafiles, plot_time, "Admin")
        sim_ecap = extract_val(datafiles, plot_time, "Ecap")

        
        ax1.plot(sim_admin, color=cmap_a(c), alpha=alpha_sim)
        ax2.plot(sim_ecap, color=cmap_b(c), alpha=alpha_sim)

        fixed_params["p_e"] = exploration_rate
        st, te, admin, ecap = integrate_fa(t=t, params=fixed_params)

        ax1.plot(t, admin / fixed_params["N"], "-", color="white", 
            linewidth=3, alpha=.5)
        ax1.plot(t, admin / fixed_params["N"], linestyle=line, 
            color=cmap_a(c), linewidth=1.5, label=lab)
        ax2.plot(t, ecap, "-", color="white", linewidth=3, alpha=.5)
        ax2.plot(t, ecap, linestyle=line, color=cmap_b(c), 
            linewidth=1.5, label=lab)

        st_sim = np.zeros(sim_ecap.shape[1])
        # count timepoints where energy was produced for each of the 100 model runs
        for i in np.arange(sim_ecap.shape[1]):
            st_sim[i] = np.sum(sim_ecap[:, i] != 0)

        surv_time.append(st_sim)

    ax3_explo_order = [2, 0, 1]
    for explo_i in ax3_explo_order:
        ax3.hist(surv_time[explo_i], bins=21, range=[0, plot_time * 1.05], 
            color=cmap_c(colors[explo_i]), alpha=.75, label=labels[explo_i])

    ax1.legend(loc="center", bbox_to_anchor=(.5, .25), ncol=3, frameon=False)
    ax2.legend(loc="center", bbox_to_anchor=(.5, .75), ncol=3, frameon=False)
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(
        [h for _, h in sorted(zip(ax3_explo_order, handles))], 
        [l for _, l in sorted(zip(ax3_explo_order, labels))], 
        loc="center", bbox_to_anchor=(.5, .75), ncol=3, frameon=False
    )

    # plt.savefig(base_folder + folder + "/pub_figure3.pdf")
    return fig

