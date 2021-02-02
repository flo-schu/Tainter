#seed = 1
import inspect
import numbers as numbers

import tainter_function_blocks as tainter

def tf5(  network = "watts" ,
          N = 100,
          k = 5,
          p = 0.05,
          layout = "spring",
          first_admin = "random" ,
          choice = "topworker",
          # popmode = "random",
          # linkmode = "random",
          exploration = 0.0,
          mepercent = 0.75,
          # death = 0.00,
          a = 1.0 ,
          stress = ["off"] ,
          shock = ["off"],
          tmax = None,
          threshold = 1.0 ,
          eff = 1.2 ,
          death_energy_level = 0,
          print_every = None):
    """
    This function implements a dynamic population to the network

    network = "watts"
        assumes watts-strogatz network from package networkx as standard network.
        It is possible to choose also from "barabasi" and "erdos". A network can
        be passed to the function as a variable. The variable must be a of the
        type networkx.classes.graph.Graph

    N = 100
        Network size

    k = 5
        mean degree of network (See the definitions of watts, barabasi, erdos to
        see what the variables do there)

    p = 0.05
        probability of rewiring (needed in watts and erdos networks). 1 means
        completely random, 0 produces a network where all members have the
        same degree

    layout = "spring"
        options for the design of the network. 'Spring' produces a network wich
        assembles nodes with higher degree among each other closer together.
        'Random' draws the position of the nodes at the beginning and then
        remains in this setup.

    first_admin = "random"
        assumes that the first admin is drawn at random. If "highest degree" is
        passed, the first node of all nodes with the highes degree is chosen.
        Could be optimized

    popmode = "random"
        sets the probability to be born as administrator or worker
        'random' produces equal chances 0.5, 0.5
        'conditional' produces chances according to the state of the network
        meaning, that it calculates weights depending on the percentage of workers
        and admins in the network. If 80% of nodes in the networks are 'A', the
        chances of being born as 'A' are equally 80%

    linkmode = "random"
        if 'random', all links of a dead node are removed and norm(k,1) are drawn
        if 'conditional', half (rounded down) of the links are retained and half of
        a randomly drawn number of a normal distribution centered around k, is
        added.

    death = 0.05
        sets the percentage of nodes dying each loop

    a = 1.0
        access to resource (between 0 and 1) TODO: create fn. Default 1.0:
        inital access to resource.

    stress = 0.0
        Between 0 and 1: reduction of productivity, induced by´stress. can be
        many concepts (population increase, declining ease of extraction, etc.)
        default = 0.0 no stress.

    tmax
        integer optional. Overwrites stress variable. If set, the model breaks
        after tmax timesteps

    threshold = 1.0
        this is the threshold at which a first or further admin is selected

    eff = 1.2
       efficiency increase for the effect of an administrator (in basic model:
       increase productivity of workers connected to admins

    death_energy_level = 0
        energy production per capita, at which certain death occurs (in basic
        model: all nodes are admins)

    print_every
        integer. If defined, graph is printed every n time steps
    """
    args = locals() # save arguments passed to the function
    args = dict(zip(list(args.keys()),[round(i,3) if isinstance(i, numbers.Number) else i for i in args.values()])) # round arguments
    fct = inspect.stack()[0][3] # save function_name which is executed

    # Initialize function
    A, L, Lc, positions, E_cap, tmax, ainit, A_exp, L_exp, Lc_exp, Admin = tainter.init(N, stress, a, eff, tmax)
    G = tainter.construct_network(network, N, k, p)
    history = tainter.init_history(A, L, Lc, E_cap, a, A_exp, L_exp, Lc_exp)

    for t in range(tmax):
        A_2 = A.copy()
        L_2 = L.copy()
        Lc_2 = Lc.copy()

        A, L, Lc = tainter.exploration(G, A, L, Lc, N, exploration )
        A_exp, L_exp, Lc_exp = tainter.node_origin(A,L,Lc,A_2,L_2,Lc_2)

        # Environmental Stress or Shocks
        a, bumm =tainter.access(a,ainit, stress, shock)

        # Calculate the Energy per Capita
        E_cap = tainter.energy_out_capita(a-bumm, L, Lc, eff, N)

        # If E_cap below a threshold -> Admin selection mechanism
        if E_cap < threshold:

            Admin = tainter.select_Admin(G, A, L, Lc, first_admin, choice)

            # if no Admin is selected, there is no need to update the network
            if Admin != None:
                L, Lc, A = tainter.update_network(A, L, Lc, Admin, G)

        E_cap = tainter.energy_out_capita(a-bumm, L, Lc, eff, N)

        if print_every != None:
            tainter.print_graph(print_every, t, A, Lc, L, G, positions, N)

        tainter.update_history(history, A, L, Lc, E_cap, a-bumm, A_exp,L_exp, Lc_exp,Admin)
        Admin = None

        if E_cap <= death_energy_level:
            merun = tainter.maximum_energy(history,mepercent)
            wb = tainter.wellbeing(history,threshold)
            tot_energy = tainter.total_energy(history)
            break

        if t == tmax-1:
            merun = tainter.maximum_energy(history,mepercent)
            wb = tainter.wellbeing(history,threshold)
            tot_energy = tainter.total_energy(history)

    return history, t, args, fct, tot_energy, wb, G
