#seed = 1
import random
from scipy.stats import norm
#random.seed(seed)  # normal python seed

import numpy as np
#random.seed(seed)  # numpy seed

import matplotlib.pyplot as plt
import os
import importlib.util

#os.chdir(os.getcwd()+'\\Tainter\\Models\\tf5')
#import networkx from tf5 source
spec = importlib.util.spec_from_file_location("networkx", "./packages/networkx/__init__.py")
nx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nx)


#import networkx as nx
import pandas as pd
from scipy.stats import entropy
import pickle as pickle
import tkinter as tk
from tkinter import filedialog
from collections import Counter

def init_history(A, L, Lc, E_cap, a, A_exp, L_exp, Lc_exp):
    """initialize dictionary which records the history of the network"""
    # A_tot1 = A.copy()
    # L_tot1 = L.copy()
    # Lc_tot1 = Lc.copy()
    history = {
               'Administration':[len(A)],
               'Administrators':[[A.copy()]],
               'Labourers':[len(L)],
               'Workers':[[L.copy()]],
               'coordinated Labourers':[len(Lc)],
               'cWorkers':[[Lc.copy()]],
               'Energy per capita':[E_cap],
               # 'complexity':[C],
               # 'transitivity':[nx.transitivity(G)],
               # 'average path length':[nx.average_shortest_path_length(G)],
               'access':[a],
               'Aexpl':[A_exp],
               'Lexpl':[L_exp],
               'Lcexpl':[Lc_exp],
               'Ashk':[[None]]
               # 'crosslinks':[crosslinks]
               }
    return history

def update_history(history, A, L, Lc, E_cap, a, A_exp, L_exp, Lc_exp,Admin):
    """
    Append results to a dictionary.
    """
    # A_tot = [i for i in A.copy()]
    # L_tot = [i for i in L.copy()]
    # Lc_tot =[i for i in L.copy()]
    history['Administration'].append(len(A))
    history['Administrators'].append([A.copy()])
    history['Labourers'].append(len(L))
    history['Workers'].append([L.copy()])
    history['coordinated Labourers'].append(len(Lc))
    history['cWorkers'].append([Lc.copy()])
    history['Energy per capita'].append(E_cap)
    # history['transitivity'].append(nx.transitivity(G))
    # history['average path length'].append(nx.average_shortest_path_length(G))
    # history['crosslinks'].append(crosslinks)
    history['access'].append(a)
    history['Aexpl'].append(A_exp)
    history['Lexpl'].append(L_exp)
    history['Lcexpl'].append(Lc_exp)
    history['Ashk'].append([Admin])
    # history['complexity'].append(C)
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


def energy_out_capita(a, L, Lc, eff, N):
    """
    compute the energy output averaged per capita after efficiency increase
    by the administration.
    """
    E_cap = a * (len(L) + len(Lc)**eff) / N
    return E_cap

def access(a,ainit,stress,shock):
    """
    computes the availability based on the stress parameter passed in the function call. If a is random. It is important that the initial access is always 1
    """
    #shock = ["on","beta",[1,10]]
    #stress = ["on","linear",0.01]
    #a = 1;ainit = 1
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
        # if np.random.random() > 0.1:
        #     bumm = 0
        #     return a, bumm

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

def init(N, stress, a, eff, tmax):
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
    E_cap = a * (len(L) + len(Lc)**eff) / N
    #if tmax == None:
    #    tmax  = int(1/stress)   # simulation time
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
                #print("coordinatedLabourers: ", Lc)
                return None
        elif choice == "toptop":
            # selects the worker with the most connections irrespective of to whom
            try:
                Admin = list(G.degree(Lc.union(L)))[np.argmax(np.array(list(G.degree(Lc.union(L))))[:,1])][0]
            except IndexError:
                print("Labourers: ", L, "     coordinatedLabourers:  ", Lc)
                return None
    return Admin

def popdev(L, Lc, A, G, N, k, popmode = "random", linkmode = "random", death = 0.05):
    """
    Population development (popdev) adds the possibility of dying and getting born
    to the network. It does not implement growth of the network. It also aims at
    keeping the initial mean degree of the network stable.
    The method works in a way, that an array of nodes is selected and their links are
    removed. The way this is done is defined via the argument [linkmode]
    a) "random": all links are removed and are newly assigned with the mean degree of
       the network to randomly chosen nodes
    b) "conditional": half of the nodes are retained and a random-normal chosen
       amount of Links centered around k with the standard ´deviation of 1 are
       newly assigned
    Additionally the class of the new node is deterimned. Again, either "random"
    or "conditional" via the argument "popmode". If popmode is random, the chance of
    being born as Administrator or Labourer is 0.5/0.5. If it is conditional, the chances
    are weighed, according to the distribution of Administrators and Workers in the network.
    i.e. If a network is full of Administrators it is likely that new Admins are born
    and vice versa.
    """
    # For some reason the mean degree implemented in the network is not the same
    # as the one which comes out. therefore K which defines which links are added
    # to new nodes, is calculated from the existing network
    #k = np.mean(list(dict(G.degree).values()))

    # calculate how many nodes die each round
    dying = int(round(N * death,0))

    # determining the birthchance at the beginning of the action of removing nodes
    # from the classes. Otherwise the probabilities won't sum to 1.
    if popmode == "conditional":
        # birthchance norms the amount of As and Ls in the network to values Between
        # 1 and 0. These are then the weights of the sampling
        birthchance = [len(A)/N, len(L.union(Lc))/N]
        #birthchance = [len(A)/len(A.union(L,Lc)), len(L.union(Lc))/len(A.union(L,Lc))]
    if popmode == "random":
        birthchance = [0.5,0.5]

    # select dead nodes from the set of all nodes (classes). Here a life expectancy
    # could also be implemented, if "higher" classes have a lower probability of
    # dying
    all = L.union(Lc,A)#; len(all)
    dead = np.random.choice(list(all), dying, replace = False);
    nbs = set()

    # delete old links
    if linkmode == "off":
        pass
    else:
        oldlinks = []
        for i in dead:
            if linkmode == "random":
                # all neighbors of node i are selected
                nbs = set(G.neighbors(i))
            elif isinstance(linkmode, float):
                nbs = set(G.neighbors(i))
                if len(nbs) == 0:
                    pass
                else:
                    # of all neighbors half (rounded down) are randomly chosen
                    nbs = np.random.choice(a = list(nbs), size = len(nbs)//2, replace = False)
            else:
                print("choose correct birth policy! ;-)")
            # tuples of (i, neighbor) are created to remove edges from network
            oldlinks = list(zip(np.repeat(i, len(nbs)),nbs))
            G.remove_edges_from(oldlinks)

    # removing dead from class
    L = L.difference(dead)
    A = A.difference(dead)
    for i in A:
        Lc = Lc.union(set(G.neighbors(i)).difference(A))
    Lc = Lc.difference(dead)
    L = L.difference(Lc)

    # get born
    born = dead

    # create new links
    if linkmode == "off":
        pass
    else:
        newlinks = []
        for i in born:
            # if k is smaller than 5, the likelihood of drawing a random number < 0
            # with a normal (gauss) distribution is too high. This would create a network
            # with growing links. Even with 5 the probability exists but it is probably
            # neglegible
            if k < 5:
                print("k (mean degree of network) is too low for rewiring of die/birth action. Either improve function or set k to minimum 5. K assumed for rewiring.")
                break
            else:
                if linkmode == "random":
                    # the number of links is drawn from a normal distribution centered
                    # around k with sigma = 1
                    links = int(round(random.gauss(k,1),0))
                    if links < 0: links = 0
                elif isinstance(linkmode, float):
                    links = int(round(random.gauss(k,1),0))
                    # new neighbors are randomly drawn
                    nbs = random.sample(all.difference([i]),links)
                    # Adding Links
                    newlinks = list(zip(np.repeat(i, len(nbs)),nbs))
                    # only half (rounded down) of the amount of links are generated
                    links = links//2
                    if links < 0: links = 0
            # new neighbors are randomly drawn
            nbs = random.sample(all.difference([i]),links)
            # Adding Links
            newlinks = list(zip(np.repeat(i, len(nbs)),nbs))
            G.add_edges_from(newlinks)


    # adding born to new class
    for i in born:
        # in this loop the individual class of each node is drawn
        newclass = np.random.choice( ["A","L"], p= birthchance)
        if newclass == "A":
            A = A.union([i])
        if newclass == "L":
            L = L.union([i])

    # refresh Lcs
    for i in A:
        Lc = Lc.union(set(G.neighbors(i)).difference(A))
    L = L.difference(Lc)

    return G, L, Lc, A

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
    #print("A:",len(A),"L:", len(L),"Lc:", len(Lc), "sum:", sum([len(A),len(L),len(Lc)]))
    return G, A, L, Lc, N

#G, A, L, Lc, N = sample_network()

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
                #Aold.append(i)
                #Lnew.append(i)
            elif i in Lc:
                Lc.remove(i)
                A.add(i)
                #Lcold.append(i)
                #Anew.append(i)
            elif i in L:
                L.remove(i)
                A.add(i)
                #Lold.append(i)
                #Anew.append(i)
            #print("A: ",len(A),"L: ", len(L),"Lc: ", len(Lc))

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
    """prints graph every n runs. Layout can be chosen from 'spring' and 'fixed' """
    if print_every != None and t%print_every == 0:
        print("#", t, "Number of nodes:",len(A) + len(Lc) + len(L),"mean degree:",np.mean(list(dict(G.degree).values())))
        if layout == "spring":
            nx.draw(G, pos = None, node_color=[0 if i in L else 0.5 if i in Lc else 1 for i in range(N)])
        elif layout == "fixed":
            nx.draw(G, pos = positions, node_color=[0 if i in L else 0.5 if i in Lc else 1 for i in range(N)])
        plt.show()

def save_data(filename, d1 , folder_name = None , d2 = None, d3 = None, d4 = None, d5 = None, d6 = None, d7 = None, d8 = None, d9 = None, d10 = None):
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
    history_record = pd.DataFrame(columns=['Administration', 'Labourers', 'coordinated Labourers', 'Energy per capita', 'complexity', 'transitivity', 'average path length', 'access', 'crosslinks'])
    run = pd.DataFrame()
    return history_record, run

def history_matrix_update(history, history_record, run,  loopdim = 1, par1 = None,  par2 = None, par3 = None ):
    """
    appends the history dictionary of each run to a data frame, so that it
    produces an easily accessible table. Furthermore it labels each run with the
    parameter combination the loops are iterating through. These are specified
    with par1, par2 and par3. The function hence can go up to three dimensions.
    """
    if loopdim == 1:
        run = run.append(pd.DataFrame(np.repeat(str(par1),len(history.get('Administration'))), columns = ['Run']), ignore_index = True)
    elif loopdim == 2:
        run = run.append(pd.DataFrame(np.repeat(str(par1)+'-'+str(par2),len(history.get('Administration'))), columns = ['Run']), ignore_index = True)
    elif loopdim == 3:
        run = run.append(pd.DataFrame(np.repeat(str(par1)+'-'+str(par2)+'-'+str(par3),len(history.get('Administration'))), columns = ['Run']), ignore_index = True)
    history_record =  history_record.append(pd.DataFrame(history), ignore_index = True)
    return history_record, run

def history_finish(history_record, run, shortnames = True):
    """
    binds the columns history record and run, sort them and optionally
    shorten the names of the dictionary.
    """
    history_record = pd.concat([run,history_record],axis = 1).loc[:,['Run','Administration', 'Labourers', 'coordinated Labourers', 'Energy per capita', 'complexity', 'transitivity', 'average path length', 'access', 'crosslinks']]
    if shortnames == True:
        short_names = {'Administration':'Admin', 'Labourers':'Labor', 'coordinated Labourers':'c.Labor', 'Energy per capita':'Ecap', 'complexity':'offdiag.complex', 'transitivity':'trans', 'average path length':'Path length', 'access':'access', 'crosslinks':'crosslinks'}
        history_record = history_record.rename(index = str, columns = short_names)
    if shortnames == False:
        pass
    return history_record



def testglob():
    global test1, test2
    test1 +20
    test2 = test2*10
    print(test1, test2)
