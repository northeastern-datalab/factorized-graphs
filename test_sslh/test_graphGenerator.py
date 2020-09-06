"""
Test class for 'sslh/graphGenerator'
Author: Wolfgang Gatterbauer
"""

import numpy as np
import sys
sys.path.append('./../sslh')
from graphGenerator import (create_distribution_vector,
                            local_randint,
                            calculate_nVec_from_Xd,
                            calculate_Ptot_from_graph,
                            calculate_outdegree_distribution_from_graph,
                            planted_distribution_model,
                            planted_distribution_model_H,
                            calculate_average_outdegree_from_graph,
                            create_blocked_matrix_from_graph)

from utils import (to_dictionary_beliefs,
                   to_centering_beliefs,
                   row_normalize_matrix,
                   eps_convergence_linbp)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from graphGenerator import graphGenerator
# from numba import jit




def test_create_distribution_vector():
    print("\n--- 'create_distribution_vector' ---")
    n = 20
    m = 41
    print("n:", n)
    print("m:", m)

    dist = create_distribution_vector(n, m, distribution='uniform')
    print("distribution for 'uniform':\n ", dist)
    dist = create_distribution_vector(n, m, distribution='triangle')
    print("distribution for 'triangle':\n ", dist)
    dist = create_distribution_vector(n, m, distribution='powerlaw', exponent=-0.5)
    print("distribution for 'powerlaw -0.5':\n ", dist)
    dist = create_distribution_vector(n, m, distribution='powerlaw', exponent=-1)
    print("distribution for 'powerlaw -1':\n ", dist)

    n = 20
    m = 81
    print("\nn:", n)
    print("m:", m)

    dist = create_distribution_vector(n, m, distribution='powerlaw', exponent=-0.5)
    print("distribution for 'powerlaw -0.5':\n ", dist)
    dist = create_distribution_vector(n, m, distribution='powerlaw', exponent=-1)
    print("distribution for 'powerlaw -1':\n ", dist)



def test_local_randint():
    print("\n--- 'local_randint(n):' ---")
    f = np.vectorize(local_randint)                     # !!! np.vectorize
    print("A few vectorized draws from same max n")
    print("0: {}".format(f([0 for _ in range(20)])))
    print("1: {}".format(f([1 for _ in range(20)])))
    print("2: {}".format(f([2 for _ in range(20)])))



def test_calculate_nVec_from_Xd():
    print("\n--- 'calculate_nVec_from_Xd(Xd):' ---")
    # Xd = {'n1': 1, 'n2' : 2, 3: 3, 4: 1, 5: 0, 6: 0, 7:0}     # Python 2 allowed comparing str and int, not anymore in Python 3
    Xd = {1: 1, 2: 2, 3: 3, 4: 1, 5: 0, 6: 0, 7: 0}
    print("Xd: {}".format(Xd))
    print("Result: {}".format(calculate_nVec_from_Xd(Xd)))



def test_smallMotivatingGraph_statistics():
    # 'create_blocked_matrix_from_graph()', 'test_calculate_Ptot_from_graph()' 'calculate_outdegree_distribution_from_graph()', 'calculate_average_outdegree_from_graph()'
    # uses motivation example with 15 nodes from VLDB introduction.
    # Weighs edges to see the affinities better in blocked matrix
    # create_blocked_matrix_from_graph,
    # test_calculate_Ptot_from_graph,
    # calculate_outdegree_distribution_from_graph,
    # calculate_average_outdegree_from_graph

    # VERSION = 'directed'        # 1: directed
    VERSION = 'undirected'      # 2: undirected

    print("\n--- Example graph from VLDB slides ---")
    Xd = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0,
          5: 1, 6: 1, 7: 1, 8: 1, 9: 1,
          10: 2, 11: 2, 12: 2, 13: 2, 14: 2,
          # 15: 2                                   # length of dictionary need to be = number of nodes in edges
          }
    # # Original VLDB drawing
    # row = [0, 2,  0, 1, 1, 2, 2, 2, 3, 4,   3,  4,  6, 7,   8,  9,  10, 10, 10, 11, 11, 11, 12, 13,]
    # col = [1, 3,  5, 8, 9, 6, 7, 8, 9, 5,  11, 12,  8, 9,  10, 14,  11, 12, 14, 12, 13, 14, 14, 14,]
    # weight = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, ]
    # # Corrected undirected graph with correct P_tot [2, 8, 2]
    # row = [0, 0, 1, 1, 2, 2, 2, 3, 4,   3,  4,  6,   8,  9,  10, 11, 12, 13,]
    # col = [1, 5, 8, 9, 6, 7, 8, 9, 5,  11, 12,  8,  10, 14,  11, 13, 14, 14,]
    # weight = [1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 5, 5, 6, 6, 6, 6, ]
    # # Corrected undirected graph with correct P_tot [2, 6, 2]
    row =    [0, 0, 1, 1, 2, 2, 3,  3,  4, 6, 8,  9, 10, 12, 13,]
    col =    [1, 5, 8, 9, 6, 7, 9, 11, 12, 8,10, 14, 11, 14, 14,]
    weight = [1, 2, 2, 2, 2, 2, 2,  3,  3, 4, 5,  5,  6,  6,  6, ]

    print("Xd:", Xd)
    if VERSION == 'undirected':
        weight = weight + weight
        row, col = row + col, col + row             # !!! assignment at same time: same line
    print("row:", row)
    print("col:", col)
    print("weight:", weight, "\n")

    print("- Random permutation of node ids:")
    ranks = np.random.permutation(len(Xd))          # ranks is the new mapping vector
    print("ranks:", ranks)
    row2 = ranks[row]                               # !!! mapping
    col2 = ranks[col]
    print("row2:", row2.tolist())          # list plots nicer than np.array
    print("col2:", col2.tolist())
    print("weight:", weight)
    W_rand = csr_matrix((weight, (row2, col2)), shape=(15, 15))
    nodes = np.array(list(Xd.keys()))
    nodes2 = ranks[nodes]
    print("nodes:  ", nodes)
    print("nodes2: ", nodes2)
    classes = np.array(list(Xd.values()))   # Python 3 requires list(dict.keys()), and also for values
    print("classes:  ", classes)
    Xd_rand = dict(zip(ranks[nodes], classes))
    print("Xd_rand: {}".format(Xd_rand))
    print("W_rand:\n{}".format(W_rand.todense()))

    print("\n- 'create_blocked_matrix_from_graph():' ")
    W_block, Xd_new = create_blocked_matrix_from_graph(W_rand, Xd_rand)
    W = W_block
    Xd = Xd_new

    print("W:\n{}".format(W.todense()))

    print("\n- 'test_calculate_Ptot_from_graph():' ")
    W2 = csr_matrix(W, copy=True)
    W2.data[:] = np.sign(W2.data)                   # W contains weighted edges -> unweighted before counting edges with Ptot
    Ptot = calculate_Ptot_from_graph(W2, Xd)
    print("Ptot:\n{}".format(Ptot))

    print("\n- 'test_calculate_nVec_from_Xd():' ")
    n_vec = calculate_nVec_from_Xd(Xd)
    print("n_vec: {}".format(n_vec))

    print("\n- 'calculate_outdegree_distribution_from_graph():' ")
    print("Outdegree distribution: {}".format( calculate_outdegree_distribution_from_graph(W, Xd=None) ))
    # print ("Outdegree distribution: {}".format( sorted(calculate_outdegree_distribution_from_graph(W, Xd=None).items()) ))
    print("Outdegree distribution per class: {}".format( calculate_outdegree_distribution_from_graph(W, Xd) ))
    print("Indegree distribution: {}".format( calculate_outdegree_distribution_from_graph(W.transpose(), Xd=None) ))
    print("Indegree distribution per class: {}".format(calculate_outdegree_distribution_from_graph(W.transpose(), Xd)))

    print("\n- 'calculate_average_outdegree_from_graph():' ")
    print("Average outdegree: {}".format(calculate_average_outdegree_from_graph(W, Xd=None)))
    print("Average outdegree per class: {}".format(calculate_average_outdegree_from_graph(W, Xd)))
    print("Average indegree: {}".format(calculate_average_outdegree_from_graph(W.transpose(), Xd=None)))
    print("Average indegree per class: {}".format(calculate_average_outdegree_from_graph(W.transpose(), Xd)))

    print("\n- Visualize adjacency matrix")
    plt.matshow(W.todense(), fignum=100, cmap=plt.cm.Greys)  # cmap=plt.cm.gray / Blues
    plt.xticks([4.5, 9.5])
    plt.yticks([4.5, 9.5])
    plt.grid(which='major')
    frame = plt.gca()
    frame.axes.xaxis.set_ticklabels([])
    frame.axes.yaxis.set_ticklabels([])
    plt.savefig('figs/Fig_test_calculate_Ptot_from_graph.png')
    os.system('open "figs/Fig_test_calculate_Ptot_from_graph.png"')



def test_planted_distribution_model():
    """ Tests the main graph generator with statistics and visualized degree distribution and edge adjacency matrix
    """
    print("\n--- 'planted_distribution_model_H', 'planted_distribution_model_P', 'number_of_connectedComponents', 'create_blocked_matrix_from_graph' --")
    CHOICE = 21
    print("CHOICE:", CHOICE)
    debug = 0

    # directed = True                     # !!! TODO: not yet clear what undirected means here, only P accepts directed
    backEdgesAllowed = True             # ??? should be enforced in code
    sameInAsOutDegreeRanking = False
    distribution = 'powerlaw'
    exponent = -0.3
    VERSION_P = True


    # --- AAAI figures ---
    if CHOICE in [1, 2, 3, 4, 5, 6]:
        n = 120
        alpha0 = [1/6, 1/3, 1/2]
        h = 8
        P = np.array([[1, h, 1],
                      [1, 1, h],
                      [h, 1, 1]])

    if CHOICE == 1:                     # P (equivalent to 2), AAAI 2
        m = 1080

    elif CHOICE == 2:                   # H (equivalent to 1)
        H0 = row_normalize_matrix(P)
        d_vec = [18, 9, 6]
        VERSION_P = False

    elif CHOICE == 3:                   # H (equivalent to 4), AAAI 3
        H0 = row_normalize_matrix(P)
        d_vec = 9
        VERSION_P = False

    elif CHOICE == 4:                   # P (equivalent to 3)
        P = np.array([[1, h, 1],
                      [2, 2, 2*h],
                      [3*h, 3, 3]])
        m = 1080

    elif CHOICE == 5:                   # H (equivalent to 2), but backedges=False
        H0 = row_normalize_matrix(P)
        d_vec = [18, 9, 6]
        VERSION_P = False
        backEdgesAllowed = False

    elif CHOICE == 6:                   # P undirected, AAAI 4
        P = np.array([[1, h, 1],
                      [h, 1, 1],
                      [1, 1, h]])
        directed = False
        backEdgesAllowed = False
        m = 540

    # --- AGAIN DIRECTED ---
    if CHOICE == 12:
        n = 1001
        alpha0 = [0.6, 0.2, 0.2]
        P = np.array([[0.1, 0.8, 0.1],
                      [0.8, 0.1, 0.1],
                      [0.1, 0.1, 0.8]])
        m = 3000
        distribution = 'uniform'    # uniform powerlaw
        exponent = None
        backEdgesAllowed = False    # ??? should be enforced in code

    if CHOICE == 13:
        # Nice for block matrix visualization
        n = 1000
        alpha0 = [0.334, 0.333, 0.333]
        h = 2
        P = np.array([[1, h, 1],
                      [h, 1, 1],
                      [1, 1, h]])
        m = 2000
        distribution = 'uniform'    # uniform powerlaw
        exponent = None
        backEdgesAllowed = False    # ??? should be enforced in code

    if CHOICE == 14:
        n = 1000
        alpha0 = [0.3334, 0.3333, 0.3333]
        h = 10
        P = np.array([[1, h, 1],
                      [h, 1, 1],
                      [1, 1, h]])
        m = 10000
        exponent = -0.55


    # --- UNDIRECTED ---
    if CHOICE == 20:
        n = 100
        alpha0 = [0.6, 0.2, 0.2]
        h = 1.4
        P = np.array([[1, h, 1],
                      [h, 1, 1],
                      [1, 1, h]])
        H0 = row_normalize_matrix(P)
        d_vec = 5
        directed = False
        exponent = -0.3
        VERSION_P = False

    elif CHOICE == 21:
        n = 1001
        alpha0 = [0.6, 0.2, 0.2]
        h = 4
        P = np.array([[1, h, 1],
                      [h, 1, 1],
                      [1, 1, h]])
        H0 = row_normalize_matrix(P)
        d_vec = 3.4                   # don't specify vector for undirected
        distribution = 'uniform'    # uniform powerlaw
        exponent = -0.5
        directed = False
        backEdgesAllowed = True             # ignored in code for undirected
        VERSION_P = False
        sameInAsOutDegreeRanking = True     # ignored in code for undirected

    elif CHOICE == 22:
        n = 1000
        m = 3000
        alpha0 = [0.6, 0.2, 0.2]
        h = 4
        P = np.array([[1, 3*h, 1],
                      [2*h, 1, 1],
                      [1, 1, h]])
        distribution = 'uniform'    # uniform powerlaw
        exponent = -0.5
        directed = False
        backEdgesAllowed = False             # ignored in code for undirected
        sameInAsOutDegreeRanking = True     # ignored in code for undirected
        debug=0

        VERSION_P = True
        H0 = row_normalize_matrix(P)


    # --- Create the graph
    start = time.time()
    if VERSION_P:
        W, Xd = planted_distribution_model(n, alpha=alpha0, P=P, m=m,
                                           distribution=distribution, exponent=exponent,
                                           directed=directed,
                                           backEdgesAllowed=backEdgesAllowed, sameInAsOutDegreeRanking=sameInAsOutDegreeRanking,
                                           debug=debug)
    else:
        W, Xd = planted_distribution_model_H(n, alpha=alpha0, H=H0, d_out=d_vec,
                                                  distribution=distribution, exponent=exponent,
                                                  directed=directed, backEdgesAllowed=backEdgesAllowed, sameInAsOutDegreeRanking=sameInAsOutDegreeRanking,
                                                  debug=debug)

    time_est = time.time()-start
    print("Time for graph generation: {}".format(time_est))

    # - Undirectd degrees: In + Out
    W_und = W.multiply(W.transpose())
    """if backEdgesAllowed then there can be edges in both directions."""
    # W_und.data[:] = np.sign(W_und.data)  # W contains weighted edges -> unweighted before counting edges with Ptot
    print("Fraction of edges that go in both directions: {}".format(np.sum(W_und.data) / np.sum(W.data)))

    # --- Statistics on created graph
    print("\n- 'calculate_Ptot_from_graph':")
    P_tot = calculate_Ptot_from_graph(W, Xd)
    print("P_tot:\n{}".format(P_tot))
    print("sum(P_tot): {}".format(np.sum(P_tot)))
    print("P (normalized to sum=1):\n{}".format(1. * P_tot / np.sum(P_tot)))           # Potential: normalized sum = 1
    H = row_normalize_matrix(P_tot)
    print("H (row-normalized):\n{}".format(H))

    print("\n- 'calculate_nVec_from_Xd':")
    n_vec = calculate_nVec_from_Xd(Xd)
    print("n_vec: {}".format(n_vec))
    print("alpha: {}".format(1.*n_vec / sum(n_vec)))

    print("\n- Average Out/Indegree 'calculate_average_outdegree_from_graph' (assumes directed for total; for undirected the totals are incorrect):")
    print("Average outdegree: {}".format(calculate_average_outdegree_from_graph(W)))
    print("Average indegree: {}".format(calculate_average_outdegree_from_graph(W.transpose())))
    print("Average total degree: {}".format(calculate_average_outdegree_from_graph(W + W.transpose())))
    print("Average outdegree per class: {}".format(calculate_average_outdegree_from_graph(W, Xd)))
    print("Average indegree per class: {}".format(calculate_average_outdegree_from_graph(W.transpose(), Xd)))
    print("Average total degree per class: {}".format(calculate_average_outdegree_from_graph(W + W.transpose(), Xd)))

    # - Overall degree distribution: In / out
    print("\n- Overall Out/In/Total degree distribution 'calculate_outdegree_distribution_from_graph':")
    print("Overall Out and Indegree distribution:")
    d_out_vec_tot = calculate_outdegree_distribution_from_graph(W, Xd=None)
    d_in_vec_tot = calculate_outdegree_distribution_from_graph(W.transpose(), Xd=None)
    print("Outdegree distribution (degree / number):\n{}".format(np.array([d_out_vec_tot.keys(), d_out_vec_tot.values()])))
    print("Indegree distribution (degree / number):\n{}".format(np.array([d_in_vec_tot.keys(), d_in_vec_tot.values()])))

    # - Overall degree distribution: In + Out
    d_tot_vec_tot = calculate_outdegree_distribution_from_graph(W + W.transpose(), Xd=None)
    print("Total degree distribution (degree / number):\n{}".format(np.array([d_tot_vec_tot.keys(), d_tot_vec_tot.values()])))

    # - Per-class degree distribution: In / out
    print("\n- Per-class Out/In/Total degree distribution 'calculate_outdegree_distribution_from_graph':")
    print("\nOutdegree distribution per class:")
    d_out_vec = calculate_outdegree_distribution_from_graph(W, Xd)
    for i in range(len(d_out_vec)):
        print("Class {}:".format(i))
        print(np.array([d_out_vec[i].keys(), d_out_vec[i].values()]))
    print("Indegree distribution per class:")
    d_in_vec = calculate_outdegree_distribution_from_graph(W.transpose(), Xd)
    for i in range(len(d_in_vec)):
        print("Class {}:".format(i))
        print(np.array([d_in_vec[i].keys(), d_in_vec[i].values()]))

    # - per-class degree distribution: In + out
    print("\nTotal degree distribution per class:")
    d_vec_und = calculate_outdegree_distribution_from_graph(W + W.transpose(), Xd)
    for i in range(len(d_vec_und)):
        print("Class {}:".format(i))
        print(np.array([d_vec_und[i].keys(), d_vec_und[i].values()]))

    print("\n- number of weakly connected components':")
    print("Number of weakly connected components: {}".format(connected_components(W, directed=True, connection='weak', return_labels=False)))


    # --- convergence boundary
    # print("\n- '_out_eps_convergence_directed_linbp', 'eps_convergence_linbp'")
    # if directed:
    #     eps_noEcho = _out_eps_convergence_directed_linbp(P, W, echo=False)
    #     eps_Echo = _out_eps_convergence_directed_linbp(P, W, echo=True)
    # else:
    Hc = to_centering_beliefs(H)
    eps_noEcho = eps_convergence_linbp(Hc, W, echo=False)
    eps_Echo = eps_convergence_linbp(Hc, W, echo=True)
    print("Eps (w/ echo): {}".format(eps_Echo))
    print("Eps (no echo): {}".format(eps_noEcho))


    # --- Fig1: Draw edge distributions
    print("\n- Fig1: Draw degree distributions")
    params = {'backend': 'pdf',
              'lines.linewidth': 4,
              'font.size': 10,
              'axes.labelsize': 24,  # fontsize for x and y labels (was 10)
              'axes.titlesize': 22,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'legend.fontsize': 8,
              'figure.figsize': [5, 4],
              'font.family': 'sans-serif'
    }
    mpl.rcdefaults()
    mpl.rcParams.update(params)
    fig = plt.figure(1)
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])  # main axes
    ax.xaxis.labelpad = -12
    ax.yaxis.labelpad = -12

    # A: Draw directed degree distribution
    y_vec = []
    for i in range(len(d_out_vec)):
        y = np.repeat(list(d_out_vec[i].keys()), list(d_out_vec[i].values()) )    # !!! np.repeat
        y = -np.sort(-y)
        y_vec.append(y)
        # print ("Class {}:\n{}".format(i,y))
    y_tot = np.repeat(list(d_out_vec_tot.keys()), list(d_out_vec_tot.values()))             # total outdegree
    y_tot = -np.sort(-y_tot)
    plt.loglog(range(1, len(y_vec[0])+1), y_vec[0], lw=4, color='orange', label=r"A out", linestyle='-')        # !!! plot default index starts from 0 otherwise
    plt.loglog(range(1, len(y_vec[1])+1), y_vec[1], lw=4, color='blue', label=r"B out", linestyle='--')
    plt.loglog(range(1, len(y_vec[2])+1), y_vec[2], lw=4, color='green', label=r"C out", linestyle=':')
    plt.loglog(range(1, len(y_tot)+1), y_tot, lw=1, color='black', label=r"tot out", linestyle='-')

    # B: Draw second edge distribution of undirected degree distribution
    y_vec = []
    for i in range(len(d_vec_und)):
        y = np.repeat(list(d_vec_und[i].keys()), list(d_vec_und[i].values()) )    # !!! np.repeat
        y = -np.sort(-y)
        y_vec.append(y)
        # print ("Class {}:\n{}".format(i,y))
    y_tot = np.repeat(list(d_tot_vec_tot.keys()), list(d_tot_vec_tot.values()))             # total outdegree
    y_tot = -np.sort(-y_tot)
    plt.loglog(range(1, len(y_vec[0])+1), y_vec[0], lw=4, color='orange', label=r"A", linestyle='-')
    plt.loglog(range(1, len(y_vec[1])+1), y_vec[1], lw=4, color='blue', label=r"B", linestyle='--')
    plt.loglog(range(1, len(y_vec[2])+1), y_vec[2], lw=4, color='green', label=r"C", linestyle=':')
    plt.loglog(range(1, len(y_tot)+1), y_tot, lw=1, color='black', label=r"tot", linestyle='-')

    plt.legend(loc='upper right', labelspacing=0)
    filename = 'figs/Fig_test_planted_distribution_model1_{}.pdf'.format(CHOICE)
    plt.savefig(filename, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype='letter', format='pdf',
                transparent=True, bbox_inches='tight', pad_inches=0.1,
                # frameon=None,                 # TODO: frameon deprecated
                )
    os.system("open " + filename)


    # --- Fig2: Draw block matrix
    print("\n- Fig2: 'create_blocked_matrix_from_graph'")
    W_new, Xd_new = create_blocked_matrix_from_graph(W, Xd)

    fig = plt.figure(2)
    row, col = W_new.nonzero()                      # transform the sparse W back to row col format
    plt.plot(col, row, 'o', color='r', markersize=2, markeredgewidth=2, lw=0, zorder=3)    # Notice (col, row) because first axis is vertical in matrices
    # plt.matshow(W_new.todense(), cmap=plt.cm.Greys)  # cmap=plt.cm.gray / Blues   # alternative that does not work as well
    plt.gca().invert_yaxis()    # invert the y-axis to start on top and go down

    # Show quadrants
    d1 = alpha0[0] * n
    d2 = (alpha0[0] + alpha0[1]) * n
    plt.grid(which='major', color='0.7', linestyle='-', linewidth=1)
    plt.xticks([0, d1, d2, n])
    plt.yticks([0, d1, d2, n])
    plt.xlabel('to', labelpad=-1)
    plt.ylabel('from', rotation=90, labelpad=0)

    frame = plt.gca()
    # frame.axes.xaxis.set_ticklabels([])       # would hide the labels
    # frame.axes.yaxis.set_ticklabels([])
    frame.tick_params(direction='inout', width=1, length=10)

    filename = 'figs/Fig_test_planted_distribution_model2_{}.pdf'.format(CHOICE)
    plt.savefig(filename, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype='letter', format='pdf',
            transparent=True, bbox_inches='tight', pad_inches=0.1)
    os.system("open " + filename)



def test_graph_statistics_forced_block_model():
    print("\n--- test_graph_statistics_forced_block_model() ---")
    H0 = np.array([[0.1, 0.8, 0.1],
                  [0.8, 0.1, 0.1],
                  [0.1, 0.1, 0.8]])
    alpha0 = np.array([0.4, 0.3, 0.3])
    print("alpha0: ", alpha0)
    print("H0:\n", H0)
    print("\n")

    n = 40
    b = 2
    start = time.time()
    Ws, X = graphGenerator(n, b, H=H0, alpha=alpha0, model='CBM', seed=None, directed=True)
    time_est = time.time()-start
    print("Time for graph generation: ", time_est)
    print("\n")

    Xd = to_dictionary_beliefs(X)
    n_vec = calculate_nVec_from_Xd(Xd)
    P_tot = calculate_Ptot_from_graph(Ws, Xd)
    H = row_normalize_matrix(P_tot)
    print("n_vec: ", n_vec)
    print("alpha: ", 1.*n_vec / sum(n_vec))
    print("P_tot:\n", P_tot)
    print("P:\n", 1. * P_tot / sum(P_tot.flatten()))           # Potential: normalized sum = 1
    print("H:\n", H)

    d_vec = calculate_outdegree_distribution_from_graph(Ws, Xd=None)
    print("Indegree distribution:\n", d_vec)
    d_vec_list = calculate_outdegree_distribution_from_graph(Ws, Xd)
    print("List of indegree distributions:")
    for dict in d_vec_list:
        print("  ", dict)









if __name__ == '__main__':
    test_create_distribution_vector()
    test_local_randint()
    test_calculate_nVec_from_Xd()
    test_smallMotivatingGraph_statistics()
    test_planted_distribution_model()
    test_graph_statistics_forced_block_model()

