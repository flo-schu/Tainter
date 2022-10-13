import importlib.util
import os
import pickle as pickle
import random
import tkinter as tk
from collections import Counter
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy, norm
import networkx as nx

# import networkx package from tf5 source
# spec = importlib.util.spec_from_file_location("networkx", "../packages/networkx/__init__.py")
# nx = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(nx)

def init_history(A, L, Lc, E_cap, a, A_exp, L_exp, Lc_exp):
    """initialize dictionary which records the history of the network"""
    history = {
        'Administration':[len(A)],
        'Administrators':[[A.copy()]],
        'Labourers':[len(L)],
        'Workers':[[L.copy()]],
        'coordinated Labourers':[len(Lc)],
        'cWorkers':[[Lc.copy()]],
        'Energy per capita':[E_cap],
        'access':[a],
        'Aexpl':[A_exp],
        'Lexpl':[L_exp],
        'Lcexpl':[Lc_exp],
        'Ashk':[[None]]
    }
    return history

def update_history(history, A, L, Lc, E_cap, a, A_exp, L_exp, Lc_exp,Admin):
    """
    Append results to a dictionary.
    """
    history['Administration'].append(len(A))
    history['Administrators'].append([A.copy()])
    history['Labourers'].append(len(L))
    history['Workers'].append([L.copy()])
    history['coordinated Labourers'].append(len(Lc))
    history['cWorkers'].append([Lc.copy()])
    history['Energy per capita'].append(E_cap)
    history['access'].append(a)
    history['Aexpl'].append(A_exp)
    history['Lexpl'].append(L_exp)
    history['Lcexpl'].append(Lc_exp)
    history['Ashk'].append([Admin])
    return history


def node_origin(A,L,Lc,A2,L2,Lc2):
    A_add = [i for i in A.difference(A2)]
    A_rem = [i for i in A2.difference(A)]
    A_exp = [A_add,A_rem]

    L_add = [i for i in L.difference(L2)]
    L_rem = [i for i in L2.difference(L)]
    L_exp = [L_add,L_rem]

    Lc_add = [i for i in Lc.difference(Lc2)]
    Lc_rem = [i for i in Lc2.difference(Lc)]
    Lc_exp = [Lc_add,Lc_rem]

    return A_exp, L_exp, Lc_exp


def energy_out_capita(a, L, Lc, elast_l, elast_lc, eff_lc, N):
    """
    compute the energy output averaged per capita after efficiency increase
    by the administration.
    """
    E_cap = a * (len(L)**elast_l + eff_lc * len(Lc)**elast_lc) / N
    return E_cap

def access(a,ainit,stress,shock):
    """
    computes the availability based on the stress parameter passed in the 
    function call. If a is random. It is important that the initial access 
    is always 1
    """
    if stress[0] == "on":
        if stress[1] == "linear":
            a -= stress[2]
        elif stress[1] =="percent":
            a = a * (1-stress[2])
        else:
            pass
    else:
        pass

    if shock[0] == "on":
        if shock[1] =="beta":
            # ggf economic shocks installieren
            bumm = random.betavariate(shock[2][0],shock[2][1])
        elif shock[1] == "occasionalrecovery":
            if np.random.choice([1,0],p=shock[3]):
                bumm = shock[2]
            else:
                pass
    else:
        bumm = 0.0

    return a, bumm

def total_energy(history):
    tot_energy = np.sum(history['Energy per capita'])
    return tot_energy

def maximum_energy(history, mepercent):
    me = np.max(history['Energy per capita'])*mepercent
    energy = np.array(history['Energy per capita'])
    merun = (energy > me).sum()
    return merun

def wellbeing(history, threshold):
    energy = np.array(history['Energy per capita'])
    wb = (energy > threshold).sum()
    return wb

def complexity(G, N):
    """return offdiagonal complexity [Claussen 2007]"""
    c = np.zeros((N, N))
    ks = G.degree()  # degree sequence
    for i in range(N):
        for j in G.neighbors(i):
            if ks[i] <= ks[j]:
                c[ks[i], ks[j]] += 1
    atilde = np.zeros(N)
    for i in range(N):
        atilde[i] = sum(np.diag(c, i))
    a = atilde / sum(atilde)
    C = entropy(a)
    return C

def construct_network(network, N, k, p):
    """
    in this function, one of each networks "barabasi, watts, or erdos" is computed
    network defines the type of network (one of the above)
    N defines the number of nodes
    k defines the average degree of each node
    p defines the rewiring probability
    """
    if str(type(network)) == "<class 'networkx.classes.graph.Graph'>":
        G = network
    elif network == "watts":
        G = nx.watts_strogatz_graph(n=N, k=k, p=p)
    elif network == "barabasi":
        G = nx.barabasi_albert_graph(n = N, m = round(k/2))
    elif network == "erdos":
        G = nx.erdos_renyi_graph(n = N, p = p)
    else:
        print("no valid network. Please specify as networkx network or choose from 'watts', 'barabasi', 'erdos'")
    return G

def init(N, stress, a, elast_l, elast_lc, eff_lc, tmax):
    """
    creates sets for the classes of the network
    network is initialized with all nodes as L
    also positions of nodes in the network are calculated so they can remain static
    initial Ecap is calculated and if tmax is not provided it is determined
    """
    # set definitions admin, worker, coordinated workers at this point all start at 0
    A  = set([])  # initially only individual 0 is administrator set([]) erstellt eine menge von einer Liste
    L  = set(range(0, N))  # all others are labourers
    Lc = set([])

    A_exp = [[],[]]
    L_exp = [[],[]]
    Lc_exp = [[],[]]
    Admin = None

    # draw random positions of the network, which remain fixed for the remaining duration of the function
    positions = list(zip(np.random.choice(N, size = N, replace = False),np.random.choice(N, size = N, replace = False)))
    
    # calculate initial
    E_cap = energy_out_capita(a, L, Lc, elast_l, elast_lc, eff_lc, N)
    ainit = a

    return A, L, Lc, positions, E_cap, tmax, ainit, A_exp, L_exp, Lc_exp, Admin

def select_Admin(G, A, L, Lc, first_admin, choice):
    """
    function to select first and following Admins, based on certain rules
    A, L, Lc need to be sets of nodes of the networkx object G
    the parameters 'first_admin' and 'choice' specify by which method
    admins are selectedself.

    first_admin:    "random", "highest degree"
    choice:         "topworker": labourer of highest degree
                    "topcoordinated": Lc of highest degree
                    "toptop": L or Lc of highest degree

    """
    if len(A) == 0:
        if first_admin == "random":
            # chooses an admin at random
            Admin = np.random.choice(list(L))
        elif first_admin == "highest degree":
            # selects first out of the list of nodes with highest degree
            Admin = np.argmax(np.array(list(G.degree(L)))[:,1])
    else:
        if choice == "random":
            try:
                Admin = np.random.choice(list(L.union(Lc)))
            except ValueError:
                return None
        if choice == "topworker":
            # selects the labourer with the highest degree
            try:
                Admin = list(G.degree(L))[np.argmax(np.array(list(G.degree(L)))[:,1])][0]
            except IndexError:
                #print("Labourers: ", L)
                return None
        elif choice == "topcoordinated":
            # selects the corrdinated worker with the highest degree
            try:
                Admin = list(G.degree(Lc))[np.argmax(np.array(list(G.degree(Lc)))[:,1])][0]
            except IndexError:
                return None
        elif choice == "toptop":
            # selects the worker with the most connections irrespective of to whom
            try:
                Admin = list(G.degree(Lc.union(L)))[np.argmax(np.array(list(G.degree(Lc.union(L))))[:,1])][0]
            except IndexError:
                print("Labourers: ", L, "     coordinatedLabourers:  ", Lc)
                return None
    return Admin

def sample_network():
    """
    function for quickly generating a network with some nodes in different classes
    """
    N =100
    A = set(list(np.random.choice(range(100), size = 5)))
    L = set(range(100)).difference(A)
    Lc = set()
    G = nx.watts_strogatz_graph(n=100,k=5,p=0.05)
    L, Lc = refresh_network(A,L,Lc,G)
    return G, A, L, Lc, N

def exploration(G, A, L, Lc, N, exploration ):
    """
    Network exploration adds the function to the tainter model, to let nodes explore
    other 'occupations', with the parameter exploration [0,1], the probability to flip
    the node is set. Nodes can either change from L or Lc to A or from A to L. Lcs are
    afterwards recalculated.
    """
    for i in range(N):
        if exploration > random.random():
            if i in A:
                A.remove(i)
                L.add(i)
            elif i in Lc:
                Lc.remove(i)
                A.add(i)
            elif i in L:
                L.remove(i)
                A.add(i)

    L, Lc = refresh_network(A, L, Lc, G)

    if sum([len(A),len(L),len(Lc)]) != N:
        print("A:",len(A),"L:", len(L),"Lc:", len(Lc), "sum:", sum([len(A),len(L),len(Lc)]))
        print("Exploration does not work changed number of nodes!")

    return A, L, Lc

def refresh_network(A, L, Lc, G):
    """
    function which recalculates all nodes based on the set of admins. Necessary,
    if the network removes admins from the set.
    """
    Lcn = set()
    for i in A:
        # find all L neighbors of all A
        Lcn = Lcn.union(set(G.neighbors(i)).difference(A))

    # if some Lcs are not coordinated any longer, find them and assign them to L
    L = L.union(Lc.difference(Lcn))
    
    # remove all Lcs from L
    L = L.difference(Lcn)
    Lc = Lcn
    return L, Lc

def update_links(Admin, A, G):
    """create a tuple of link pairs between old Admins and new Admins"""
    newlinks = tuple(zip(list(np.repeat(Admin,len(A))),list(A)))
    
    # method add_edges_from updates the variable G as nonlocal
    G.add_edges_from(newlinks)

def update_network(A, L, Lc, Admin, G):
    """
    update network is an essential function, which is necessary for the operation
    of adding new Admins to the set of existing Admins (A). It does not involve
    a loop and therefore performs much better than the expensive rewiring
    """
    A.add(Admin)
    try:
        L.remove(Admin)
    except KeyError:
        Lc.remove(Admin)
    Lc = Lc.union(set(G.neighbors(Admin)).difference(A))
    L = L.difference(Lc)
    return L, Lc, A

def crosslinks(G, A, L, Lc):
    """
    crosslinks calculate the amount of links between Administration and Labour
    (L + Lc)
    """
    GA = G.subgraph(A)
    GnonA = G.subgraph(L.union(Lc))
    crosslinks = len(G.edges) - len(GA.edges) - len(GnonA.edges)
    return crosslinks

def print_graph(print_every, t, A, Lc, L, G, positions, N, layout = "spring"):
    """prints graph every n runs. Layout can be chosen from 'spring' and 
    'fixed' 
    """
    if print_every != None and t%print_every == 0:
        print(
            "#", t, "Number of nodes:",len(A) + len(Lc) + len(L),
            "mean degree:",np.mean(list(dict(G.degree).values()))
            )
        if layout == "spring":
            nx.draw(G, pos = None, node_color=[0 if i in L else 0.5 if i in Lc else 1 for i in range(N)])
        elif layout == "fixed":
            nx.draw(G, pos = positions, node_color=[0 if i in L else 0.5 if i in Lc else 1 for i in range(N)])
        plt.show()

def save_data(
    filename, d1 , folder_name = None , d2 = None, d3 = None, d4 = None, 
    d5 = None, d6 = None, d7 = None, d8 = None, d9 = None, d10 = None
    ):
    if folder_name is None:
        file = open( './data/'+ '/' + filename + '.pkl', 'wb')
    else:
        file = open( './data/'+ folder_name + '/' + filename + '.pkl', 'wb')
    pickle.dump((d1, d2, d3, d4, d5, d6, d7, d8, d9, d10), file)
    file.close()

def load_data(loaddata = True):
    if loaddata == True:
        root = tk.Tk(); root.withdraw()
        file_path = filedialog.askopenfilename()
        input_file = open(file_path, "rb")
        loaded_data = pickle.load(input_file)
        input_file.close()
    else:
        input_file = open(loaddata,"rb")
        loaded_data = pickle.load(input_file)
    return loaded_data

def history_matrix_init():
    history_record = pd.DataFrame(columns=[
        'Administration', 'Labourers', 'coordinated Labourers', 
        'Energy per capita', 'complexity', 'transitivity', 
        'average path length', 'access', 'crosslinks'
        ])
    run = pd.DataFrame()
    return history_record, run

def history_matrix_update(
    history, history_record, run,  loopdim = 1, 
    par1 = None,  par2 = None, par3 = None 
    ):
    """
    appends the history dictionary of each run to a data frame, so that it
    produces an easily accessible table. Furthermore it labels each run with the
    parameter combination the loops are iterating through. These are specified
    with par1, par2 and par3. The function hence can go up to three dimensions.
    """
    if loopdim == 1:
        run = run.append(
            pd.DataFrame(np.repeat(str(par1),len(history.get('Administration'))), 
            columns = ['Run']), ignore_index = True)
    elif loopdim == 2:
        run = run.append(
            pd.DataFrame(np.repeat(str(par1)+'-'+str(par2),len(history.get('Administration'))), 
            columns = ['Run']), ignore_index = True)
    elif loopdim == 3:
        run = run.append(
            pd.DataFrame(np.repeat(str(par1)+'-'+str(par2)+'-'+str(par3),len(history.get('Administration'))), 
            columns = ['Run']), ignore_index = True)
    history_record =  history_record.append(
        pd.DataFrame(history), ignore_index = True)

    return history_record, run

def history_finish(history_record, run, shortnames = True):
    """
    binds the columns history record and run, sort them and optionally
    shorten the names of the dictionary.
    """
    history_record = pd.concat([run,history_record],axis = 1).loc[:,[
        'Run','Administration', 'Labourers', 'coordinated Labourers', 
        'Energy per capita', 'complexity', 'transitivity', 
        'average path length', 'access', 'crosslinks'
        ]]
    if shortnames == True:
        short_names = {
            'Administration':'Admin', 
            'Labourers':'Labor', 
            'coordinated Labourers':'c.Labor', 
            'Energy per capita':'Ecap', 
            'complexity':'offdiag.complex', 
            'transitivity':'trans', 
            'average path length':'Path length', 
            'access':'access', 
            'crosslinks':'crosslinks'}
        history_record = history_record.rename(index = str, columns = short_names)
    if shortnames == False:
        pass
    return history_record


def disentangle_admins(history, N):
    """
    analyse by which mechanism admins were created 
    """
    A_tot = [list(i[0]) for i in history["Administrators"]]
    A_exp = history['Aexpl']
    A_shk = list()
    A_exp_add = list()
    A_exp_rem = list()
    A_null = list()
    A_shk2 = history['Ashk']
    A_shk2 = [[] if i[0] is None else i for i in A_shk2 ]
    A_shk3 = list()
    A_shk4 = set()
    A_shk4_size = list()
    A_exp_temp = set()
    A_exp_size = list()

    for i in np.arange(0, len(A_tot)):
        A_exp_add.append([j for j in A_exp[i][0]])
        A_exp_rem.append([j for j in A_exp[i][1]])

    # calculate the administrators created by the mechanism
    for i in np.arange(0, len(A_tot)):
        A_shk.append(list(set(A_tot[i]).difference(set(A_tot[i-1])).difference(A_exp[i][0])))
        A_shk4 = A_shk4.difference(A_exp_rem[i])
        A_shk4 = A_shk4.union(A_shk2[i])
        A_shk4_size.append(len(A_shk4))
        A_exp_temp = A_exp_temp.union(set(A_exp_add[i]))
        A_exp_temp = A_exp_temp.difference(set(A_exp_rem[i]))
        A_exp_size.append(len(A_exp_temp))

    data = {'x': np.arange(len(A_shk)),
            'Admin': np.array([len(i[0]) for i in history['Administrators']])/N,
            'A_shk_c': np.array([len(i) for i in A_shk])/N,
            'A_shk': np.array([len(i) for i in A_shk2])/N,
            'A_pool_shk': np.array(A_shk4_size)/N,
            'A_pool_exp': np.array(A_exp_size)/N,
            'A_exp_add': np.array([len(i) for i in A_exp_add])/N,
            'A_exp_rem': np.array([len(i) for i in A_exp_rem])/N,
            'A_exp': np.array([len(i) for i in A_exp_add]) -
                np.array([len(i) for i in A_exp_rem])/N,
            'Ecap': np.array(history['Energy per capita']),
            'Access': np.array(history['access'])}

    return data