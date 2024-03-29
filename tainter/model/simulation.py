import inspect
import numbers as numbers
import tainter.model.methods as tm

def tainter(  
    network = "watts" ,
    N = 100,
    k = None,
    p = None,
    layout = "spring",
    first_admin = "random" ,
    choice = "topworker",
    exploration = 0.0,
    shock_alpha=1,
    shock_beta=15,
    tmax = None,
    elasticity_l = 0.95 ,
    elasticity_c = 0.95 ,
    productivity_c = 1.2,
    death_energy_level = 0,
    print_every = None
):
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

    tmax
        integer optional. If set, the model breaks
        after tmax timesteps

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
    # calculate the initial per capita resource access of the network
    # ensures that the per capita energy production equals 1 initially
    resource_access = N / N ** elasticity_l
    A, L, C, positions, E_cap, tmax, ainit, A_exp, L_exp, Lc_exp, Admin = tm.init(N, resource_access, elasticity_l, elasticity_c, productivity_c, tmax)
    G = tm.construct_network(network, N, k, p)
    history = tm.init_history(A, L, C, E_cap, resource_access, A_exp, L_exp, Lc_exp)

    for t in range(tmax):
        A_2 = A.copy()
        L_2 = L.copy()
        Lc_2 = C.copy()

        A, L, C = tm.exploration(G, A, L, C, N, exploration )
        A_exp, L_exp, Lc_exp = tm.node_origin(A,L,C,A_2,L_2,Lc_2)

        # Environmental Shocks
        shock = tm.shock(shock_alpha=shock_alpha, shock_beta=shock_beta)

        # this is a more general form of resource reduction which scales
        # with resource access. 
        access_after_shock = resource_access * (1 - shock)

        # Calculate the Energy per Capita
        E_cap = tm.energy_out_capita(access_after_shock, L, C, elasticity_l, elasticity_c, productivity_c, N)

        # If E_cap below a threshold -> Admin selection mechanism
        if E_cap < 1:

            Admin = tm.select_Admin(G, A, L, C, first_admin, choice)

            # if no Admin is selected, there is no need to update the network
            if Admin != None:
                L, C, A = tm.update_network(A, L, C, Admin, G)

        E_cap = tm.energy_out_capita(access_after_shock, L, C, elasticity_l, elasticity_c, productivity_c, N)

        if print_every != None:
            tm.print_graph(t, A, C, L, G, positions, N, layout=layout, print_every=print_every)

        tm.update_history(history, A, L, C, E_cap, access_after_shock, A_exp,L_exp, Lc_exp,Admin)
        Admin = None

        if E_cap <= death_energy_level:
            merun = tm.maximum_energy(history, 0.75)
            wb = tm.wellbeing(history,1)
            tot_energy = tm.total_energy(history)
            break

        if t == tmax-1:
            merun = tm.maximum_energy(history, 0.75)
            wb = tm.wellbeing(history,1)
            tot_energy = tm.total_energy(history)

    return history, t, args, fct, tot_energy, wb, G
