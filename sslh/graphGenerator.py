"""
Synthetic Generators for labeled random graphs for SSL-H
Inspiration: http://networkx.github.io/documentation/latest/_modules/networkx/generators/random_graphs.html

Author: Wolfgang Gatterbauer
License: Apache Software License
"""


import random
import warnings
from random import randint
from numpy.random import random_sample, shuffle
import numpy as np
from scipy.sparse import csr_matrix
from scipy import optimize
from math import ceil, pi
import collections          # collections.Counter
from utils import (row_normalize_matrix,
                   check_normalized_beliefs,
                   from_dictionary_beliefs,
                   calculate_potential_from_row_normalized)
import networkx as nx
RANDOMSEED = None                 # TODO: remove after removing graph generator at the end. Better initialized in experimental loop outside this file with following two lines:
    # random.seed(RANDOMSEED)           # seeds some other python random generator
    # np.random.seed(seed=RANDOMSEED)   # seeds the actually used numpy random generator; both are used and thus needed




def planted_distribution_model_H(n, alpha, H, d_out,
                                 distribution='powerlaw', exponent=-0.5,
                                 directed=True,
                                 backEdgesAllowed=False,
                                 sameInAsOutDegreeRanking=False,
                                 debug=0):
    """Variation on planted_distribution_model_H that uses (P, m) instead of (H, d_out)
    Notice that for undirected graphs, the actual average degree distribution will be double of d_out
    """
    k = len(alpha)
    if not isinstance(d_out, (collections.Sequence, np.ndarray)):  # allow single number as input
        d_out = [d_out] * k

    P = calculate_potential_from_row_normalized(H, alpha, d_out)
    m = np.rint(n * np.array(alpha).dot(np.array(d_out)))

    # print("P:", P)

    return planted_distribution_model(n=n, alpha=alpha, P=P, m=m,
                                            distribution=distribution, exponent=exponent,
                                            directed=directed,
                                            backEdgesAllowed=backEdgesAllowed,
                                            sameInAsOutDegreeRanking=sameInAsOutDegreeRanking,
                                            debug=debug)



def planted_distribution_model(n, alpha, P, m,
                               distribution='powerlaw', exponent=-0.5,
                               directed=True,
                               backEdgesAllowed=False,                      # deprecated
                               sameInAsOutDegreeRanking=False,              # deprecated
                               debug=0):
    """Creates a directed random graph with planted compatibility matrix 'P'.
    Accepts (n, alpha_vec, P, m). The alternative version accepts: (n, alpha_vec, H, d_out_vec).
    If directed==False: creates an undirected graph. Requires:
        1) P to be symmetric, with identical row and column sum
        2) ignores: backEdgesAllowed, sameInAsOutDegreeRanking
    Notice: m = |W| for directed, but 2m = |W| for undirected.
    Notice: Average outdegree d_out = m/n for directed, but average total degree d = 2m/n for directed and undirected

    Parameters
    ----------
    n : int
        number of nodes
    alpha : k-dimensional ndarray or list
        node label distribution
    P : [k,k] ndarray
        Compatibility matrix (no need for column-normalized or symmetric)
    m : int
        total number of edges
    distribution : string, optional (Default = 'powerlaw')
        'uniform', 'triangle', 'powerlaw': used with "create_distribution_vector(n, m, distribution, exponent)"
    exponent : float, optional (Default = None)
        only for 'powerlaw', by default = -1
    directed : Boolean, optional (Default = True)
        False: then constructs an undirected graph. Requires symmetric doubly stochastic potential
    backEdgesAllowed : Boolean, optional (Default = False)
        False: then two nodes cannot be connected by two edges back and forth
        Overwritten for undirected to be False
    sameInAsOutDegreeRanking : Boolean, optional (Default = False)
        True: then node with highest indegree also has highest outdegree among its peers
        Overwritten for undirected to be False
    debug : int (Default = 0)
        0: print nothing
        1: prints some statistics
        2: prints even node degree distributions

    Returns
    -------
    W : sparse.csr_matrix
        sparse weighted adjacency matrix
    Xd : dictionary
        Explicit beliefs
    """

    # -- Jump out from inner loop if graph cannot be found
    # Define an exception that allows to jump out from inner loop to a certain outer loop
    class GraphNotFound(Exception):
        pass


    # -- Initialization
    alpha = np.asarray(alpha)
    k = len(alpha)
    k1, k2 = P.shape
    assert k == k1 and k == k2

    # # DEL
    # n_vec = np.array(alpha*n, int)  # number of nodes in each class
    # P_sum = P.sum()
    # P_tot = m * P / P_sum                   # !!!
    # P_row_sum = P.sum(1, keepdims=True)   # sums along horizontal axis
    # H = 1. * P / P_row_sum
    # m_out_vec = P_tot.sum(1, keepdims=False)     # sums along vertical axis
    # d_out_vec = m_out_vec / n_vec

    # if not directed:
    #     # for i in range(k):                  # symmetric matrix [not important either, P + P^T (back edges) becomes symmetric
    #     #     for j in range(k):
    #     #         assert P[i,j] == P[j,i]
    #     # s_vec = P.sum(1, keepdims=False)    # same col and row sum [actually not important, only symmetry]
    #     # for i in range(k):
    #     #     assert s_vec[0] == s_vec[i]
    #     # d_out_vec = 1. * d_out_vec / 2              # use half of the desired degree since symmetric back edges are created
    #     # d_out_vec = 1. * d_out_vec * np.power(alpha, -1) / k    # calculate the d_out vector (given doubly stoch constraint)
    #     if backEdgesAllowed:
    #         warnings.warn("'backEdgesAllowed' set to False")
    #         backEdgesAllowed = False            # Otherwise, same edge could be created twice, redundant
    #     if sameInAsOutDegreeRanking:
    #         warnings.warn("'sameInAsOutDegreeRanking' set to False")
    #         sameInAsOutDegreeRanking = False    # Otherwise in uniform distribution not correct
    # # d_out_vec = np.asarray(d_out_vec)


    # --- Big loop that attempts to create a graph for 20 times (sometimes the parameters don't allow a graph)
    attempt = 0
    finished = False
    while attempt < 20 and not finished:

        # -- (1) n_vec: np.array: number of nodes for each class
        n_vec = np.array(alpha*n, int)  # number of nodes in each class
        delta = np.sum(n_vec) - n
        n_vec[k-1] = n_vec[k-1] - delta     # make sure sum(N)=n, in case there are rounding errors, correct the last entry

        # -- Xd: dictionary: class of each node
        Xl = [ [i]*n_vec[i] for i in range(k) ]
        Xl = np.hstack(Xl)                  # flatten nested array
        shuffle(Xl)                         # random order of those classes. Array that maps i -> k
        Xd = {i : Xl[i] for i in range(n)}  # Xd: dictionary that maps i -> k

        # -- P / P_tot: nested np.array: total number of edges between each node type
        # P = calculate_potential_from_row_normalized(H, alpha, d_out_vec)
        P_tot = m * P / P.sum()                   # !!!
        P_tot = np.rint(P_tot).astype(int)
        # P_row_sum = P_tot.sum(1, keepdims=False)    # sums along horizontal axis
        delta = m - P_tot.sum()
        P_tot[0][0] = P_tot[0][0] + delta
        # for i in range(k):
        #     P_tot[i][i] = P_tot[i][i] + delta[i]
            # add any delta to diagonal: that guarantees a symmetric matrix for undirected case
        assert np.all(P_tot >= 0), "Negative values in P_tot due to rounding errors. Change graph parameters"   # Can happen for H with 0 entries due to necessary rounding to closest integers

        # -- (2) m_out_vec: np.array: number outgoing edges in each class / m: total number of edges
        # m_out_vec = np.rint(n_vec * d_out_vec).astype(int)      # round to nearest integer
        m_out_vec = P_tot.sum(1, keepdims=False)
        # delta = np.rint(np.sum(alpha * n * d_out_vec) - np.sum(m_out_vec)).astype(int)
        # m_out_vec[k-1] = m_out_vec[k-1] + delta     # make sure sum(m_out_vec)=expected(m), in case there are rounding errors, correct the last entry
        m = np.sum(m_out_vec)

        # -- m_in_vec: number of incoming edges per class / d_in_vec: np.array: average in-degree per class
        m_in_vec = P_tot.sum(0, keepdims=False)     # sums along vertical axis
        d_in_vec = 1. * m_in_vec / n_vec

        # -- (3) list_OutDegree_vecs, list_InDegree_vec: list of np.array: distribution of in/outdegrees for nodes in each class
        list_OutDegree_vec = []
        list_InDegree_vec = []
        for i in range(k):

            # if not directed:            # undirected case works differently: create double the degrees
            #     m_out_vec[i] *= 2       # but then deduce 2 outdegrees per edge (ignoring indegrees completely)

            out_distribution = create_distribution_vector(n_vec[i], m_out_vec[i], distribution=distribution, exponent=exponent)
            list_OutDegree_vec.append(out_distribution)

            # if directed:
            #     in_distribution = create_distribution_vector(n_vec[i], m_in_vec[i], distribution=distribution, exponent=exponent)
            #     list_InDegree_vec.append(in_distribution)
            in_distribution = create_distribution_vector(n_vec[i], m_in_vec[i], distribution=distribution, exponent=exponent)
            list_InDegree_vec.append(in_distribution)

        # -- listlistNodes: list of randomly shuffled node ids for each class
        listlistNodes = [[node for node in range(n) if Xd[node] == i] for i in range(k)]
        for innerList in listlistNodes:
            shuffle( innerList )

        # -- list_OutDegree_nodes: list of list of node ids:
        #   contains for each outgoing edge in each class the start node id, later used for sampling
        list_OutDegree_nodes = []
        for i in range(k):
            innerList = []
            for j, item in enumerate(listlistNodes[i]):
                innerList += [item] * list_OutDegree_vec[i][j]
            list_OutDegree_nodes.append(innerList)

        if not sameInAsOutDegreeRanking:  # shuffle the randomly ordered list again before assigning indegrees
            for innerList in listlistNodes:
                np.random.shuffle(innerList)
        list_InDegree_nodes = []  # list of each node times the outdegree
        for i in range(k):
            innerList = []
            for j, item in enumerate(listlistNodes[i]):
                innerList += [item] * list_InDegree_vec[i][j]
            list_InDegree_nodes.append(innerList)
        # if directed:
        #     if not sameInAsOutDegreeRanking:        # shuffle the randomly ordered list again before assigning indegrees
        #         for innerList in listlistNodes:
        #             np.random.shuffle( innerList )
        #     list_InDegree_nodes = []        # list of each node times the outdegree
        #     for i in range(k):
        #         innerList = []
        #         for j, item in enumerate(listlistNodes[i]):
        #             innerList += [item] * list_InDegree_vec[i][j]
        #         list_InDegree_nodes.append(innerList)


        if debug >= 1:
            print("\n-- Print generated graph statistics (debug >= 1):")
            # print("d_out_vec: ", d_out_vec)
            print("n_vec: ", n_vec)
            # print "Xd:\n ", Xd
            print("m_out_vec: ", m_out_vec)
            print("m: ", m)
            print("P:\n", P)
            print("P_tot:\n", P_tot)
            if not directed:
                print("Undirected P_tot+P_tot^T:\n", P_tot + P_tot.transpose())
            print("m_in_vec: ", m_in_vec)
            print("d_in_vec: ", d_in_vec)
            for i in range(k):
                # print "len(list_OutDegree_nodes[", i, "]): ", len(list_OutDegree_nodes[i])
                print("sum(list_OutDegree_vec[", i, "]): ", sum(list_OutDegree_vec[i]))
                print("len(list_OutDegree_vec[", i, "]): ", len(list_OutDegree_vec[i]))
                if debug == 2:
                    print("out_distribution class[", i, "]:\n", list_OutDegree_vec[i])
                if directed:
                    print("sum(list_InDegree_vec[", i, "]): ", sum(list_InDegree_vec[i]))
                    print("len(list_InDegree_vec[", i, "]): ", len(list_InDegree_vec[i]))
                    if debug == 2:
                        print("in_distribution class[", i, "]:\n", list_InDegree_vec[i])
                # print "len(list_InDegree_nodes[i]): ", len(list_InDegree_nodes[i])
            print("list_OutDegree_nodes:\n ", list_OutDegree_nodes)
            # print "list_InDegree_nodes:\n ", list_InDegree_nodes

        # -- (4) create actual edges: try 10 times
        row = []
        col = []
        edges = set()       # set of edges, used to verify if a given edge already exists
        try:

            for i in range(k):
                for j in range(k):
                    counterCollision = 0
                    while P_tot[i][j] > 0:

                        # print(P_tot)

                        # -- pick two compatible nodes for candidate edge (s, t)
                        i_index = local_randint(len(list_OutDegree_nodes[i]))
                        s = list_OutDegree_nodes[i][i_index]


                        j_index = local_randint(len(list_InDegree_nodes[j]))
                        t = list_InDegree_nodes[j][j_index]
                        # if directed:
                        #     j_index = local_randint(len(list_InDegree_nodes[j]))
                        #     t = list_InDegree_nodes[j][j_index]
                        # else:
                        #     j_index = local_randint(len(list_OutDegree_nodes[j]))   # Re-use OutDegree
                        #     t = list_OutDegree_nodes[j][j_index]


                        # -- check that this pair can be added as edge, then add it
                        if (not s == t and
                                not (s, t) in edges and
                                (backEdgesAllowed or not (t, s) in edges)):


                            def time_funct1():
                                row.append(s)
                                col.append(t)
                                edges.add((s, t))


                            def time_funct2():
                                del list_OutDegree_nodes[i][i_index]


                                # if directed:
                                #     del list_InDegree_nodes[j][j_index]
                                # else:
                                #     if i == j and i_index < j_index:            # careful with deletion if i == j
                                #         del list_OutDegree_nodes[j][j_index-1]  # prior deletion may have changed the indices
                                #     else:
                                #         del list_OutDegree_nodes[j][j_index]
                                del list_InDegree_nodes[j][j_index]


                                P_tot[i][j] -= 1
                                counterCollision = 0

                            # Some list manipulations are wrapped into functions so that profiler can take the time
                            # Turns out that above funct2 takes most time for large graphs: deletion creates a copy
                            # TODO: speed up
                            time_funct1()
                            time_funct2()


                        # -- throw exception if too many collisions (otherwise infinite loop)
                        else:
                            counterCollision += 1
                            if counterCollision > 2000:
                                # print "** Collision: incompatible parameters ***"
                                # print "** Most likely incompatible parameters with powerlaw function ***"
                                # print "** Edge class", i, "-> ", j
                                raise GraphNotFound("2000 collisions")

                            # # -- Idea of backtracking in case of conflicts
                            # if sameInAsOutDegreeRanking and counterEdges < P_tot[i, j]:
                            #     # print "already taken: ", (s,t)
                            #     s = row[-1]
                            #     # print "s: ", s
                            #     t = col[-1]
                            #     # print "row: ", row
                            #     del row[-1]
                            #     # print "row: ", row
                            #     del col[-1]
                            #     edges.remove((s,t))
                            #     # print "list_OutDegree_nodes[i]: ", list_OutDegree_nodes[i]
                            #     # np.append(list_OutDegree_nodes[i], s)
                            #     list_OutDegree_nodes[i].append(s)
                            #     # print "list_OutDegree_nodes[i]: ", list_OutDegree_nodes[i]
                            #     # np.append(list_InDegree_nodes[j], t)
                            #     list_InDegree_nodes[j].append(t)
                            #     counterEdges += 1
                            # else:
                            #     print i_id
                            #     print j_id
                            #     raise Exception("Back-edges already present")


        except GraphNotFound as e:
            print("Failed attempt #{}".format(attempt))
            attempt +=1
        else:
            finished = True

    if not finished:
        raise GraphNotFound("Graph generation failed")

    if not directed:
        row2 = list(row)    # need to make a temp copy
        row.extend(col)
        col.extend(row2)

    Ws = csr_matrix(([1]*len(row), (row, col)), shape=(n, n))
    # assert connected_components(Ws, directed=True, connection='weak', return_labels=False) < 2, "Created graph is not connected :("     % !!! TODO
    return Ws, Xd



def create_distribution_vector(n, m, distribution='uniform', exponent=-1):
    """Returns an integer distribution of length n, with total m items, and a chosen distribution

    Parameters
    ----------
    n : int
        The number of x values
    m : int
        The total sum of all entries to be created
    distribution : string, optional (Default = 'uniform')
        'uniform', 'triangle', 'powerlaw'
    exponent : float, optional (Default = None)
        only for 'powerlaw', by default = -1

    Returns
    -------
    distribution : np.array
        list of n values with sum = m, in decreasing value
    """
    if distribution == 'uniform':   # e.g., n=10, m=23: dist = [3, 3, 3, 2, 2, 2, 2, 2, 2, 2]
        d = m // n
        mod = m % n
        dist = [d+1] * mod, [d] * (n-mod)

    elif distribution == 'triangle':
        def triangle(x, slope):
            return int(ceil((x+pi/4)*slope-pi/pi/1e6))
            # Explanation 1: pi as irrational number for pivot of slope is chosen to make sure that there is
            #   always some slope from that point with a total number of points below (thus, only one point is added
            #   for an infinitesimal small increase of slope)
            # Explanation 2: pivot has vertical point to allow 0 edges for some nodes (small to only use if necessary)

        def sum_difference(k):  # given slope k, what is the difference to m currently
            s = m
            for i in range (n):
                s -= triangle(i,k)
            return s

        k0 = 2.*m/n**2
        slope = optimize.bisect(sum_difference, 0, 2*k0, xtol=1e-20, maxiter=500)
        # Explanation: find the correct slope so that exactly m points are below the separating line
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.bisect.html#scipy.optimize.bisect
        dist = []
        for i in range (n-1, -1, -1):        # counting backwards to start with biggest values first
            dist.append(triangle(i,slope))

    elif distribution == 'powerlaw':
        # if exponent is None:
        #     exponent = -1.0     # by default
        exponent -= pi/100000   # make sure that the exponent is irrational, allows to always find a solution

        # # DELETE: original slower implementation (factor 2): part 1/2
        # def powerlaw(x, top, exponent):
        #     return int(ceil(top*(x+1)**exponent-pi/1e6))   # top0 made
        #     # Explanation: -pi/100000 allows 0 if m<n (as for triangle)
        #
        # def sum_difference(top):
        #     s = m
        #     for i in range(n):
        #         s -= powerlaw(i, top, exponent)             # @np.vectorize did not help
        #     return s

        def sum_difference(top):
            return m-np.sum(np.ceil(top * (np.arange(1, n+1) ** exponent) - pi/1e6))
            # Explanation: -pi/100000 allows 0 if m<n (as for triangle)

        def powerlaw_vec(top):
            return np.ceil(top * (np.arange(1, n + 1) ** exponent) - pi / 1e6).astype(int)

        integral = 1. / (exponent + 1) * (n**(exponent+1)-1)
        a0 = 1.*m/integral              # upper bound
        top = optimize.bisect(sum_difference, 0, a0, xtol=1e-20, maxiter=500)

        dist = powerlaw_vec(top)
        # # DELETE: original slower implementation (factor 2): part 2/2
        # dist = []
        # for i in range (n):
        #     dist.append(powerlaw(i, top, exponent))

    else:
        raise Exception("You specified a non-existing method")

    dist = np.hstack(dist).astype(int)  # creates numpy array (plus flattens the array, necessary for triangular)
    assert dist.sum() == m
    return dist



def local_randint(n):
    """Chooses random integer between 0 and n-1. Just a wrapper around ranint that also allows n=0, in which case returns 0"""
    if n == 0:
        return 0
    elif n > 0:
        return randint(0, n-1)
    else:
        raise Exception("Value >= 0 required")



def calculate_nVec_from_Xd(Xd):
    """Calculates 'n_vec': the number of times each node class occurs in graph.
    Given graph with explicit beliefs in dictionary format 'Xd'.
    Assumes zeroindexing.
    """
    X0 = from_dictionary_beliefs(Xd)
    return X0.sum(axis=0)

    # # -- OLD DEL
    # k = max(Xd.values()) + 1
    # counterClasses = collections.Counter(Xd.values())   # count multiplicies of nodes classes
    # n_vec = np.zeros(k, dtype=int)
    # for j, w in counterClasses.items():
    #     n_vec[j] = w
    # return n_vec



def calculate_Ptot_from_graph(W, Xd, zeroindexing=True):
    """Calculates [k x k] array 'P_tot': the number of times each edge type occurs in graph.
    Uses a sparse directed (incl. undirected) [n x n] adjacency matrix 'W' and explicit beliefs in dictionary format 'Xd'.
    [Does not ignore weights of 'W'. Updated with simpler multiplication]
    Assumes zeroindexing.
    If normalizing is required later:
        m = sum(P_tot.flatten())       # sum of all entries = number of edges
        Pot = 1. * P_tot / m           # Potential: normalized sum = 1
        P_tot = Pot
    """
    X0 = from_dictionary_beliefs(Xd, zeroindexing=zeroindexing)
    return X0.transpose().dot(W.dot(X0))

    # # -- OLD DEL
    # k = max(Xd.values()) + 1
    # row, col = W.nonzero()                             # transform the sparse W back to row col format
    # edges = zip(row, col)
    # assert set(np.concatenate([row, col])) <= set(Xd.keys())     # !!! verify that Xd contains a type for each node in W
    # edgetypes = [ (Xd[s], Xd[t]) for (s, t) in edges]
    # counterEdges = collections.Counter(edgetypes)       # counts multiplicies of edge types
    # P_tot = np.zeros((k, k), dtype=int)
    # for (s, t), w in counterEdges.items():
    #     P_tot[s, t] = w
    # return P_tot
    # #     if normalized:
    # #         He_tot = sum(P_tot.flatten())       # sum of all entries
    # #         Pot = 1. * P_tot / He_tot           # Potential: normalized sum = 1
    # #         P_tot = Pot



def calculate_outdegree_distribution_from_graph(W, Xd=None):
    """Given a graph 'W', returns a dictionary {degree -> number of nodes with that degree}.
    If a dictionary 'Xd' of explicit beliefs is given, then returns a list of dictionaries, one for each node class.
    Takes weight into acount [OLD: Ignores weights of 'W'. Assumes zeroindexing.]
    Transpose W to get indegrees.
    """
    n, _ = W.shape
    countDegrees = W.dot(np.ones((n, 1))).flatten().astype(int)

    # # OLD version that ignored the weight
    # row, col = W.nonzero()                      # transform the sparse W back to row col format
    # countNodes = collections.Counter(row)       # count number of times every node appears in rows
    # # Add all nodes to the counter (thus nodes with 0 ocurrences have "node key -> 0", important for later statistics)
    # for key in range(n):
    #     countNodes.setdefault(key, 0)

    if Xd is None:
        countIndegrees = collections.Counter(countDegrees)  # count multiplicies of nodes classes
        return countIndegrees
        # # OLD version
        # countIndegrees = collections.Counter(countNodes.values())   # count multiplicies of nodes classes

    else:
        listCountIndegrees = []

        X0 = from_dictionary_beliefs(Xd)
        for col in X0.transpose():
            countDegreesInClass = countDegrees*col      # entry-wise multiplication
            countDegreesInClass = countDegreesInClass[np.nonzero(countDegreesInClass)]
            countIndegreesInClass = collections.Counter(countDegreesInClass)
            listCountIndegrees.append(countIndegreesInClass)

        # # OLD version
        # k = max(Xd.values()) + 1
        # listCounterNodes = [{} for _ in range(k)]
        # for key, value in countNodes.iteritems():
        #     j = Xd[key]
        #     listCounterNodes[j][key] = value
        # for dict in listCounterNodes:
        #     countIndegrees = collections.Counter(dict.values())   # count multiplicies of nodes classes
        #     # listCountIndegrees.append(countIndegrees)

        return listCountIndegrees



def calculate_average_outdegree_from_graph(W, Xd=None):
    """Given a graph 'W', returns the average outdegree for nodes in graph.
    If a dictionary 'Xd' of explicit beliefs is given, returns a list of average degrees, one for each node class.
    Assumes zeroindexing. Ignores weights of 'W'.
    """
    if Xd is None:
        d_dic = calculate_outdegree_distribution_from_graph(W)
        return 1. * np.sum(np.multiply( list(d_dic.keys()), list(d_dic.values()))) / sum(d_dic.values())    # in change to pyton 3, requires list () casting for dictionary entries
    else:
        d_dic_list = calculate_outdegree_distribution_from_graph(W, Xd)
        d_vec = []
        for d_dic in d_dic_list:
            d_vec.append(1. * np.sum(np.multiply(list(d_dic.keys()), list(d_dic.values()))) / sum(d_dic.values()))
        return d_vec



def create_blocked_matrix_from_graph(W, Xd):
    """Given a graph 'W' and the classes of each node, permutes the labels of the nodes
        as to allow nicer block visualization with np.matshow.
        Thus nodes of same type will have adjacent ids.
    The new matrix starts with nodes of class 0, then 1, then 2, etc.
    Assumes zeroindexing. Xd needs to have appropriate size.
    Returns a new matrix W and new dictionary Xd
    """
    row, col = W.nonzero()                      # transform the sparse W back to row col format
    weight = W.data
    nodes = np.array(list(Xd.keys()))
    classes = np.array(list(Xd.values()))
    inds = np.lexsort((nodes, classes))         # !!! Sort by classes, then by nodes
    ranks = inds.argsort()                      # !!! gives the new ranks of an original node
    W_new = csr_matrix((weight, (ranks[row], ranks[col])), shape=W.shape)  # edges only in one direction
    Xd_new = dict(zip(ranks[nodes], classes))
    return W_new, Xd_new



def forced_block_model(n, d, H, alpha, directed=True, clamped=True):
    """Returns a graph with n nodes and n*b edges.
    Nodes are divided into classes exactly according to alpha (no uncertainty).
    Each node as source actively tries to connect to exactly b other nodes as targets. Thus outdegree = b.
    Targets are chosen according to row-normalized H matrix.

    Parameters
    ----------
    n : int
        The number of nodes
    d : int
        The exact [!!! average???] outdegree of each node, thus the number of edges per node
    H : [k,k] ndarray
        row-normalized homophily matrix
    alpha : k ndarray
        a prior probability distribution of classes
    seed : int, optional
       seed for random number generator (default=None)
    directed : bool, optional (Default = true)
        model creates directed edges.
    clamped : bool, optional (Default = true)
        new model with fixed Potential. If False, then original model with expected alpha and H

    Returns
    -------
    W : sparse.csr_matrix
        sparse weighted adjacency matrix
    X : np.array int
        Explicit belief matrix
        ! maybe should be better Xd

    Notes
    -----
    Some nodes may be more incoming edges than others. But average indegree is also b.
    Uses function weighted_sample for random neighbor draws.
    Directed edges are drawn as to never go in both directions (!!! may want to put a flag later)
    internal parameter repMax: edge node tries to connect to randomly chosen node type.
        If not possible repMax times, then it gives up. Thus can lead to graphs with fewer edges b than expected
    """

    repMax = 10     # !!! max number of times of attempts connect to a certain node type
    k = len(alpha)

    # Determine node classes
    # Xl: list that maps each index i to the respective class k of that node i
    # Xd, X
    if clamped:                        # clamp the number of nodes for each class exactly
        classNum = np.array(alpha*n, int)  # array of number of nodes in each class
        delta = np.sum(classNum) - n
        classNum[k-1] = classNum[k-1] - delta     # make sure sum(N)=n, in case there are rounding errors, correct the last entry
        Xl = [ [i]*classNum[i] for i in range(k) ]     # create a list of that maps for each index of a node to its class, in 3 steps
        Xl = np.hstack(Xl)          # flatten nested array
        np.random.shuffle(Xl)       # random order of those classes. Array that maps i -> k
    else:                           # old code that chooses independently. Led to random instantiaton for actual alpha
        Xl = np.random.choice(k, n, replace=True, p=alpha)  # Array that maps i -> k
    Xd = {i : Xl[i] for i in range(n)}      # Xd: dictionary that maps i -> k
    X = from_dictionary_beliefs(Xd, n, k)

    # ClassNodes [maps k -> array of nodes of class k]
    classNodes =[[] for i in range(k)]
    for c in range(k):
        classNodes[c] = np.array([i for (i,j) in Xd.items() if j == c])

    # legacy
    if not clamped:
        classNum = []
        for c in range(k):
            classNum.append( np.size(classNodes[c]) )
        assert(min(classNum) > 0)   # at least one node for each class

    # row, col: index structure for edges
    # edges: set of edges
    # He: count edge matrix
    # T: T[j] is list T[j,i] of node types to which the i-th edge links
    row = []
    col = []
    edges = set()       # set of edges, used to verify if edge already exists

    # Determine end classes for each edge with a given edge source type
    if clamped:
        He = []         # Count edge matrix
        for j in range(k):
            Z = np.array(H[j]*classNum[j]*d, int)  # number of nodes in each class
            delta = np.sum(Z) - classNum[j]*d      # balancing due to rounding possible errors
            Z[k-1] = Z[k-1] - delta
            He.append(Z)
            # T[j]
        He = np.array(He)
        delta = np.sum(He) - n*d            # balancing due to rounding possible errors
        He[k-1,k-1] = He[k-1,k-1] - delta

        T = []          # list of lists (for a given source node class) of edge target node classes for each edge
        id_T = []       # list of indexes
        for j in range(k):
            Z1 = [ [i]*He[j,i] for i in range(k) ]
            Z2 = np.hstack(Z1).astype(np.int64)          # flatten nested array. Then make sure it is integer
            np.random.shuffle(Z2)       # random order of those classes. Array that maps i -> k
            T.append(Z2)
            id_T.append(0)

        # determine actual edges
        for i in range(n):
            j = Xd[i]           # class of start node
            for l in range(d):
                c = T[j][id_T[j]]  # class of end node
                connected = False  # has the new node already found another node to connect to
                l=1                # l-th attempt to connect this node type
                while l <= repMax and not connected:
                    v = random.choice(classNodes[c])    # choose a random node of that class
                    if      (not v==i and               # don't connect to any node if edge exists in either direction
                             not (i,v) in edges and
                             not (v,i) in edges):
                        connected = True
                        row.append(i)
                        col.append(v)
                        edges.add((i,v))
                    l += 1
                id_T[j] += 1

    else:
        # Actual loop for each node and edge
        # !!! better use: to_scipy_sparse_matrix(G,nodelist=None,dtype=None):
        for i in range(n):
            pk = X[i].dot(H)
            pk = np.squeeze(np.asarray(pk)) # probability distribuion of neighbor

            for j in range(d):
                c = weighted_sample(pk)     # class of next neighbor to connect to

                connected = False  # has the new node already found another node to connect to
                l=1                # l-th attempt to connect this node type
                while l <= repMax and not connected:
                    v = random.choice(classNodes[c])
                    if      (not v==i and
                             not (i,v) in edges and
                             not (v,i) in edges):
                        connected = True
                        row.append(i)
                        col.append(v)
                        edges.add((i,v))
                    l += 1

    # Create sparse matrix. If directed=False then insert edges in both directions => symmetric W
    if directed is False:
        row2 = list(row)    # need to make a temp copy
        row.extend(col)
        col.extend(row2)
    Ws = csr_matrix(([1]*len(row), (row, col)), shape=(n, n))
    return Ws, X



def weighted_sample(weights):
    """Helper function that chooses a random index from an input vector of weights.
    Each index is chosen with probability proportional to the relative weight.
    Input: weights : array of weigths
    Output: index from 0 to length(weights)-1
    Used for old random graph generators
    """
    bins = np.add.accumulate(1.*weights/np.sum(weights))
    ind = np.searchsorted(bins, random_sample(), "right")
    return ind



def graphGenerator(n, b, H, alpha, model='CBM', seed=RANDOMSEED, directed=True):
    """
    !!! why RANDOMSEED in capital
    """
    # Verify input
    k, k2 = H.shape
    k3 = len(alpha)
    assert(k == k2), "H matrix is not symmetric"
    assert(k == k3), "alpha does not have same dimensions as H"
    assert isinstance(n, int), "n is not an integer: %r" % n    # two ways to check for type
    assert isinstance(b, int), "b is not an integer: %r" % b
    assert type(alpha).__module__ == np.__name__, "alpha is not a numpy array: %r" % alpha
    assert type(H).__module__ == np.__name__, "H is not a numpy array: \n%r" % H
    check_normalized_beliefs(H)          # H needs to contain normalized rows

    if seed is not None:
        random.seed(seed)           # seeds some other python random generator
        np.random.seed(seed=seed)   # seeds the actually used numpy random generator; both are used and thus needed
    if b >= 1.*n/k:                 # should be updated now
        raise nx.NetworkXError("k>=n, choose smaller k or larger n")
    check_normalized_beliefs(alpha), "alpha is not a probability distribution"


    if model == 'NM':
        return forced_block_model(n, b, H, alpha, directed=directed, clamped=False)
    elif model == 'CBM':
        return forced_block_model(n, b, H, alpha, directed=directed, clamped=True)
    else:
        raise Exception("You specified a non-existing graph model_graph")









