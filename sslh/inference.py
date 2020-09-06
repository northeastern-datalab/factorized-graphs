"""
Linearized Belief Propagation (LinBP) as used in
    Factorized Graph Representations for Semi-Supervised Learning from Sparse Data (SIGMOD 2020)
    Krishna Kumar P., Paul Langton, Wolfgang Gatterbauer
    https://arxiv.org/abs/2003.02829
and originally proposed in
    Linearized and Single-Pass Belief Propagation (PVLDB 2015)
    Wolfgang Gatterbauer, Stephan Guennemann, Danai Koutra, Christos Faloutsos
    http://www.vldb.org/pvldb/vol8/p581-gatterbauer.pdf

Author: Wolfgang Gatterbauer
License: Apache Software License
"""


from utils import (degree_matrix,
                   to_centering_beliefs,
                   to_explicit_bool_vector,
                   to_explicit_list,
                   row_normalize_matrix,
                   matrix_convergence_percentage,
                   W_star
                   )
from scipy.sparse import issparse
from copy import deepcopy                       # TODO: necessary?
import numpy as np
PREC = 1e-4
import warnings



def linBP_symmetric_parameterized(X, W, H_prop, H_orig=None,
                                  method='echo', eps = 1,
                                  alpha=0, beta=0, gamma=0,
                                  numMaxIt=10,
                                  convergencePercentage=None, convergenceThreshold=0.9961947, similarity='cosine_ratio',
                                  debug=1):

    """Generalization of linearized belief propagation (linBP) to arbitrary propagation matrix W*(alpha, beta) and clamping(gamma)
    Additional Parameters: alpha, beta, gamma (see under 'SSLH_utils.W_star')
    Also simplifies parameterization of noecho, echo, echo with compensation into one parameter
    """
    assert method in {'noecho', 'echo', 'comp'}
    echo = True
    compensation = False
    if method == 'echo':
        None
    elif method == 'noecho':
        echo = False
    elif method == 'comp':
        compensation = True

    return linBP_symmetric(X, W_star(W, alpha=alpha, beta=beta, gamma=gamma, indices=to_explicit_list(X)),
                           H=H_prop, H_orig=H_orig,
                           eps = eps,
                           echo=echo,
                           numMaxIt=numMaxIt,
                           convergencePercentage=convergencePercentage, convergenceThreshold=convergenceThreshold, similarity=similarity,
                           debug=debug)



def linBP_symmetric(X, W, H, H_orig=None,
                    echo=False,
                    compensation=False,
                    eps = 1,
                    numMaxIt=10,
                    convergencePercentage=None, convergenceThreshold=0.9961947,
                    similarity='cosine_ratio',
                    debug=1):
    """Linearized belief propagation given one symmetric, doubly-stochastic compatibility matrix H

    Parameters
    ----------
    X : [n x k] np array
        seed belief matrix. Can be explicit beliefs or centered residuals
    W : [n x n] sparse.csr_matrix
        sparse weighted adjacency matrix
    H : [k x k] np array
        Propagation Compatibility matrix used directly (to be centered)
    echo:  Boolean (Default = False)
        True to include the echo cancellation term (DEPRECATED)
    compensation : boolean (Default=False)
        True calculates the exact compensation for echo H* (only works if echo=True)
        Only semantically correct if W is unweighted        (TODO: extend with more general formula)
        Only makes sense if H is centered                   (TODO: verify)
    numMaxIt : int (Default = 10)
        number of maximal iterations to perform
    convergencePercentage : float (Default = None)
        percentage of nodes that need to have converged in order to interrupt the iterations.
        If None, then runs until numMaxIt
        Notice that a node with undefined beliefs does not count as converged if it does not change anymore
        (in order to avoid counting nodes without explicit beliefs as converged in first few rounds).
    convergenceThreshold : float (Default = 0.9961947)
        cose similarity (actually, the "cosine_ratio" similarity) between two belief vectors in order to deem them as identicial (thus converged).
        In case both vectors have the same length, then: cos(5 deg) = 0.996194698092. cos(1 deg) = 0.999847695156
    similarity : String (Default = 'cosine_ratio'
        Type of similarity that is used for matrix_convergence_percentage
    debug : int (Default = 1)
        0 : no debugging and just returns F
        1 : tests for correct input, and just returns F
        2 : tests for correct input, and returns (F, actualNumIt, actualNumIt, convergenceRatios)
        3 : tests for correct input, and returns (list of F, actualNumIt, list of convergenceRatios)

    Returns (if debug == 0 or debug == 1)
    -------------------------------------
    F : [n x k] np array
        final belief matrix, each row normalized to form a label distribution

    Returns (if debug == 2)
    -----------------------
    F : [n x k] np array
        final belief matrix, each row normalized to form a label distribution
    actualNumIt : int
        actual number of iterations performed
    actualPercentageConverged : float
        percentage of nodes that converged

    Returns (if debug == 3)
    -----------------------
    List of F : [(actualNumIt+1) x n x k] np array
        list of final belief matrices for each iteration, represented as 3-dimensional numpy array
        Also includes the original beliefs as first entry (0th iteration). Thus has (actualNumIt + 1) entries, not actualNumIt
    actualNumIt : int
        actual number of iterations performed (not counting the first pass = 0th iteration for initializing)
    List of actualPercentageConverged : list of float (with length actualNumIt)
        list of percentages of nodes that converged in each iteration > 0. Thus has actualNumIt entries

    References
    ----------
    [1] W. Gatterbauer, S. Guennemann, D. Koutra, and C. Faloutsos,
        "Linearized and Single-Pass Belief Propagation", PVLDB 8(5): 581-592 (2015).
        http://www.vldb.org/pvldb/vol8/p581-gatterbauer.pdf
    """

    # -- Create variables for convergence checking and debugging
    assert debug in {0, 1, 2, 3}
    if debug >= 1:
        n1, n2 = W.shape
        n = n1
        n3, k1 = X.shape
        k = k1
        k2, k3 = H.shape
        assert(n1 == n2 & n2 == n3)
        assert(k1 == k2 & k2 == k3)
        assert similarity in ('accuracy', 'cosine', 'cosine_ratio', 'l2')
    if convergencePercentage is not None or debug >= 2:
        F1 = X              # F1 needs to be initialized to track the convergence progress (either for stopping condition, or for debug information)
    if debug >= 3:
        listF = [X]         # store the belief matrices for each iteration
        listConverged = []  # store the percentage of converged nodes for each iteration

    F = X

    if echo:
        H2 = H.dot(H)
        D = degree_matrix(W, undirected=True, squared=True)
        if compensation:
            H_star = np.linalg.inv( np.identity(len(H)) - H2 ).dot(H)       # TODO: can become singular matrix. Then error for inverting
            H_star2 = H.dot(H_star)

    # -- Actual loop including convergence conditions
    converged = False
    actualNumIt = 0

    # print("undirected actualNumIt: {}".format(actualNumIt))
    # start = time.time()

    while actualNumIt < numMaxIt and not converged:
        actualNumIt += 1

        # -- Calculate new beliefs
        if echo is False:
            F = X + W.dot(F).dot(H)
        else:
            if not compensation:
                F = X + W.dot(F).dot(H) - D.dot(F).dot(H2)     # W.dot(F) is short form for: sparse.csr_matrix.dot(W, F)

            else:
                F = X + W.dot(F).dot(H_star) - D.dot(F).dot(H_star2)

        # -- Check convergence (or too big divergence) and store information if debug
        if convergencePercentage is not None or debug >= 2:
            actualPercentageConverged = matrix_convergence_percentage(F1, F, threshold=convergenceThreshold, similarity=similarity)
            diff = np.linalg.norm(F - F1)      # interrupt loop if it is diverging (Time 0.1msec per iteration for n = 5000, d = 10)

            if (convergencePercentage is not None and actualPercentageConverged >= convergencePercentage)\
                    or (diff > 1e10):
                converged = True
            F1 = F   # save for comparing in *next* iteration

        if debug == 3:
            listF.append(F)  # stores (actualNumIt+1) values
            listConverged.append(actualPercentageConverged)

    # print("\nundirected time for iterations: {}".format(time.time() - start))

    # -- Various return formats
    if debug <= 1:
        return F
    elif debug == 2:
        return F, actualNumIt, actualPercentageConverged
    else:
        return np.array(listF), actualNumIt, listConverged



def beliefPropagation(X, W, P,
                      numMaxIt=10,
                      convergencePercentage=None, convergenceThreshold=0.9961947,
                      debug=1, damping=1, clamping=False):
    """Standard belief propagation assuming a directed graph with two variants:
        V1: one directed potential across edge direction: P is one potential, and W contains the weights of edges
        V2: a set of potentials on different edges: P is a tensor, and W indexes the potentials
    Dimensions of P (2 or 3) determines variant.
    Uses message-passing with division: see [Koller,Friedman 2009] Section 10.3.1.
    Uses damping: see [Koller,Friedman 2009] Section 11.1.
    Can be run either with given number of maximal iterations or until specified percentage of nodes have converged.
    Convergence of a node is determined by (variant of) cosine similarity between *centered beliefs* from two iterations.
    If convergence criterium is reached, the iterations will stop before maximal iterations.
    Parameter "debug" allows alternative, more detailed outputs, e.g., to get intermediate belief values.
    Checks that every entry in X and P are > 0.
    Can model undirected graphs by (1) specifing every edge only for one direction, an d(2) using symmetric potentials.

    TODO: also implement version without message passing with division

    TODO: future variant with non-constant k and different potential dimensions

    TODO: future variant without echo cancellation

    TODO: alternative convergence condition:
        if np.allclose(x, x_new, atol=1e-10):
            break]

    TODO: clamping not necessary: all depends on relative strength of prior beliefs

    Parameters
    ----------
    X : [n x k] np array
        prior (explicit) belief matrix.
        Rows do not have to be row-normalized.
        Rows can be all 0, which get later replaced by undefined prior belief.
    W : [n x n] sparse.csr_matrix
        directed sparse weighted adjacency matrix (thus a directed graph is assumed)
        Also allows undirected graph by simply specifying only symmetric potentials
        V1: weight determines thea ctual edge weight
        V2: weight determines the index of a potential (from potential tensor P)
    P : V1: [k x k]
        any directed potential (no requirement for normalization or identical row or column sums)
        V2: [num_pot_P x k x k] np array
        set of potentials (as tensor)
    numMaxIt : int (Default = 10)
        number of maximal iterations to perform
    convergencePercentage : float (Default = None)
        percentage of nodes that need to have converged in order to interrupt the iterations.
        Notice that a node with undefined beliefs does not count as converged if it does not change anymore
        (in order to avoid counting nodes without explicit beliefs as converged in first few rounds).
        If None, then runs until numMaxIt
    convergenceThreshold : float (Default = 0.9961947)
        cose similarity (actually, the "cosine_ratio" similarity) between two belief vectors in order to deem them as identicial (thus converged).
        In case both vectors have the same length, then: cos(5 deg) = 0.996194698092. cos(1 deg) = 0.999847695156
    debug : int (Default = 1)
        0 : no debugging and just returns F
        1 : tests for correct input, and just returns F
        2 : tests for correct input, and returns (F, actualNumIt, convergenceRatios)
        3 : tests for correct input, and returns (list of F, list of convergenceRatios)
    damping : float   (Default = 1)
        fraction of message values that come from new iteration (if 1, then no re-use of prior iteration)
    clamping : Boolean (Default = False)
        whether or not the explicit beliefs in X should be clamped to the nodes or not

    Returns (if debug == 0 or debug == 1)
    -------------------------------------
    F : [n x k] np array
        final belief matrix, each row normalized to form a label distribution

    Returns (if debug == 2 )
    ------------------------
    F : [n x k] np array
        final belief matrix, each row normalized to form a label distribution
    actualNumIt : int
        actual number of iterations performed
    actualPercentageConverged : float
        percentage of nodes that converged

    Returns (if debug == 3 )
    ------------------------
    List of F : [(actualNumIt+1) x n x k] np array
        list of final belief matrices for each iteration, represented as 3-dimensional numpy array
        Also includes the original beliefs as first entry (0th iteration). Thus has (actualNumIt + 1) entries
    actualNumIt : int
        actual number of iterations performed (not counting the first pass = 0th iteration for initializing)
    List of actualPercentageConverged : list of float (with length actualNumIt)
        list of percentages of nodes that converged in each iteration > 0. Thus has actualNumIt entries
    """

    # --- create variables for convergence checking and debugging
    n, k = X.shape
    dim_pot = len(P.shape)  # dimensions 2 or 3: determines V1 or V2
    Pot = P                 # for case of dim_pot = 2
    if debug >= 1:
        assert (X >= 0).all(), "All explicit beliefs need to be >=0 "
        assert(issparse(W)), "W needs to be sparse"
        n2, n3 = W.shape
        assert type(P).__module__ == "numpy", "P needs to be numpy array (and not a matrix)"
        assert dim_pot in [2, 3], "Input Potentials need to be 2-dimensional or 3-dimensional"
        if dim_pot == 2:
            assert (P >= 0).all(), "All entries in the potentials need to be >=0 "
            k2, k3 = P.shape
        else:
            num_pot_P, k2, k3 = P.shape
            for P_entry in P:
                assert (P_entry >= 0).all(), "All entries in each potential need to be >=0 "
            assert W.dtype == int, "Entries of weight matrix need to be integers to reference index of the potential"
            weight = W.data
            set_pot = set(weight)
            max_pot_W = max(set_pot)
            assert max_pot_W <= set_pot, "Indices in W refering to P need to be smaller than the number of potentials"
        assert(n == n2 & n2 == n3), "X and W need to have compatible dimensions"
        assert(k == k2 & k2 == k3), "X and P need to have compatible dimensions"
    if debug >= 3:
        listF = []          # store the belief matrices for each iteration
        listConverged = []  # store all L2 norms to previous iteration

    # --- create edge dictionaries
    row, col = W.nonzero()
    nodes = set(np.concatenate((row, col)))
    dict_edges_out = {}                         # dictionary: i to all nodes j with edge (i->j)
    for node in nodes:
        dict_edges_out[node] = set()
    dict_edges_in = deepcopy(dict_edges_out)    # dictionary: i to all nodes j with edge (i<-j)

    for (i,j) in zip(row, col):
        dict_edges_out[i].add(j)
        dict_edges_in[j].add(i)

    if dim_pot == 3:
        dict_edges_pot = {}     # Dictionary: for each directed edge (i,j) -> index of the potential in P[index, :, :]
        for (i, j, d) in zip(row, col, weight):
            dict_edges_pot[(i, j)] = d

    # --- X -> X0: replace all-0-rows with all 1s (no need to normalize initial beliefs)
    implicitVector = 1-1*to_explicit_bool_vector(X)             # indicator numpy array with 1s for rows with only 0s
    implicitVectorT = np.array([implicitVector]).transpose()    # vertical 1 vector for implicit nodes
    X0 = X + implicitVectorT    # X0: prio beliefs: addition of [n x k] matrix with [n x 1] vector is ok

    F1 = X0                     # old F: only for checking convergence (either because convergencePercantage not None or debug >= 2)
    F2 = X0.astype(float)   # new F: copy is necessary as to not change original X0 matrix when F2 is changed

    # --- Actual loop: each loop calculates (a) the new messages (with damping) and (b) the new beliefs
    converged = False
    actualNumIt = -1    # iterations start with 0th iteration
    while actualNumIt < numMaxIt and not converged:
        actualNumIt += 1

        # --- (a) calculate messages
        if actualNumIt == 0:
            # --- first pass (counts as 0th iteration): create message dictionaries and initialize messages with ones
            dict_messages_along_1 = {}        # dictionary: messages for each edge (i->j) in direction i->j
            dict_messages_against_1 = {}      # dictionary: messages for each edge (i<-j) in direction i->j
            default = np.ones(k)            # first message vector: all 1s
            for (i,j) in zip(row, col):
                dict_messages_along_1[(i,j)] = default
                dict_messages_against_1[(j,i)] = default
        else:
            # --- other iterations: calculate "messages_new" using message-passing with division (from F and messages)
            dict_messages_along_2 = {}            # new dictionary: messages for each edge (i->j) in direction i->j
            dict_messages_against_2 = {}          # new dictionary: messages for each edge (i<-j) in direction i->j
            for (i,j) in dict_messages_along_1.keys():  # also includes following case: "for (j,i) in dict_messages_against_1.keys()"
                if dim_pot == 3:                        # need to reference the correct potential in case dim_pot == 3
                    Pot = P[dict_edges_pot[(i,j)]-1, :, :]
                dict_messages_along_2[(i,j)] = (F2[i] / dict_messages_against_1[(j,i)]).dot(Pot)  # entry-wise division
                dict_messages_against_2[(j,i)] = (F2[j] / dict_messages_along_1[(i,j)]).dot(Pot.transpose())
                # TODO above two lines can contain errors

            # --- assign new to old message dictionaries, and optionally damp messages
            if damping == 1:
                dict_messages_along_1 = dict_messages_along_2.copy()        # requires shallow copy because of later division
                dict_messages_against_1 = dict_messages_against_2.copy()
            else:
                for (i,j) in dict_messages_along_1.keys():
                    dict_messages_along_1[(i,j)] = damping*dict_messages_along_2[(i,j)] + \
                                                   (1-damping)*dict_messages_along_1[(i,j)]
                for (i,j) in dict_messages_against_1.keys():
                    dict_messages_against_1[(i,j)] = damping*dict_messages_against_2[(i,j)] + \
                                                     (1-damping)*dict_messages_against_1[(i,j)]

        # --- (b) create new beliefs by multiplying prior beliefs with all incoming messages (pointing in both directions)
        for (i, f) in enumerate(F2):
            if not clamping or implicitVector[i] == 0:  # only update beliefs if those are not explicit and clamped
                F2[i] = X0[i]        # need to start multiplying from explicit beliefs, referencing the row with separate variable did not work out
                for j in dict_edges_out[i]:         # edges pointing away
                    F2[i] *= dict_messages_against_1[(j,i)]
                for j in dict_edges_in[i]:          # edges pointing inwards
                    F2[i] *= dict_messages_along_1[(j,i)]
                    # TODO line can contain errors


        # --- normalize beliefs [TODO: perhaps remove later to optimize except in last round]
        F2 = row_normalize_matrix(F2, norm='l1')

        # --- check convergence and store information if debug
        if convergencePercentage is not None or debug >= 2:
            F1z = to_centering_beliefs(F1)
            F2z = to_centering_beliefs(F2)
            actualPercentageConverged = matrix_convergence_percentage(F1z, F2z, threshold=convergenceThreshold)
            if convergencePercentage is not None \
                    and actualPercentageConverged >= convergencePercentage\
                    and actualNumIt > 0:  # end the loop early
                converged = True
            F1 = F2.copy()  # save for comparing in *next* iteration, make copy since F entries get changed

        if debug == 3:
            listF.append(F2.copy())      # stores (actualNumIt+1) values (copy is important as F2 is later overwritten)
            if actualNumIt > 0:
                listConverged.append(actualPercentageConverged) # stores actualNumIt values

    # --- Various return formats
    if debug <= 1:
        return F2
    elif debug == 2:
        return F2, actualNumIt, actualPercentageConverged
    else:
        return np.array(listF), actualNumIt, listConverged



def maximumMarginals(X, W, P, debug=1):
    """Calculates Maximum Marginals by enumeration over all k^n possible worlds, assuming a directed graph with two variants:
        V1: one directed potential across edge direction: P is one potential, and W contains the weights of edges
        V2: a set of potentials on different edges: P is a tensor, and W indexes the potentials
    Dimensions of P (2 or 3) determines variant.
    Checks that every entry in X and P are > 0.
    Can model undirected graphs by (1) specifing every edge only for one direction, an d(2) using symmetric potentials.

    Creates one n-dimensional array with k^n entries for unnormalized probabilities of all worlds.
    Then sums up appropriate marginals.

    TODO: future variant with non-constant k and different potential dimensions

    Parameters
    ----------
    X : [n x k] np array
        prior (explicit) belief matrix.
        Rows do not have to be row-normalized.
        Rows can be all 0, which get later replaced by undefined prior belief.
    W : [n x n] sparse.csr_matrix
        directed sparse weighted adjacency matrix (thus a directed graph is assumed)
        Also allows undirected graph by simply specifying only symmetric potentials
        V1: weight determines thea ctual edge weight
        V2: weight determines the index of a potential (from potential tensor P)
    P : V1: [k x k]
        any directed potential (no requirement for normalization or identical row or column sums)
        V2: [num_pot_P x k x k] np array
        set of potentials (as tensor)
    debug : int (Default = 1)
        0 : no debugging
        1 : tests for correct input

    Returns
    -------
    F : [n x k] np array
        final belief matrix, each row normalized to form a label distribution
    """

    n, k = X.shape
    dim_pot = len(P.shape)  # dimensions 2 or 3: determines V1 or V2
    dimensions = k*np.ones(n, dtype=int)    # dimensions of multi-dimensional unnormalized probability array
    Prob = np.zeros(dimensions)             # n-dimensional array containing unnormalized probabilities of k^n worlds

    if debug >= 1:
        assert (X >= 0).all(), "All explicit beliefs need to be >=0 "
        assert(issparse(W)), "W needs to be sparse"
        n2, n3 = W.shape
        assert type(P).__module__ == "numpy", "P needs to be numpy array (and not a matrix)"
        assert dim_pot in [2, 3], "Input Potentials need to be 2-dimensional or 3-dimensional"
        if dim_pot == 2:
            assert (P >= 0).all(), "All entries in the potentials need to be >=0 "
            k2, k3 = P.shape
        else:
            num_pot_P, k2, k3 = P.shape
            for P_entry in P:
                assert (P_entry >= 0).all(), "All entries in each potential need to be >=0 "
            assert W.dtype == int, "Given several potentials, the entries of weight matrix need to be integers, in order to reference the index of the potential in the potential tensor"
            weight = W.data
            set_pot = set(weight)
            max_pot_W = max(set_pot)
            assert max_pot_W <= set_pot, "Indices in W refering to P need to be smaller than the number of potentials"
        assert(n == n2 & n2 == n3), "X and W need to have compatible dimensions"
        assert(k == k2 & k2 == k3), "X and P need to have compatible dimensions"

    # --- X -> X0: replace all-0-rows with all 1s (no need to normalize initial beliefs)
    implicitVector = 1-1*to_explicit_bool_vector(X)             # indicator numpy array with 1s for rows with only 0s
    implicitVectorT = np.array([implicitVector]).transpose()    # vertical 1 vector for implicit nodes
    X0 = X + implicitVectorT    # X0: prio beliefs: addition of [n x k] matrix with [n x 1] vector is ok

    # --- Calculate the unnormalized probability of a world (uses implicitly: X, W, P, dim_pot)
    def unnormalized_probability(assignment):
        # Calculates the weight by multiplying all priors and all edge potentials (instead of log sum)
        prob = 1
        # log_prob = 0
        for (row, j) in zip(X0, assignment):    # assignment contains the classes for each node
            prob *= row[j]
            # log_prob += math.log((row[i]), 2)    # log factor 2
        row, col = W.nonzero()
        weight = W.data
        for (r, c, w) in zip(row, col, weight):
            if dim_pot == 2:
                factor = P[assignment[r], assignment[c]]
            else:           # need to reference the correct potential in case dim_pot == 3
                factor = P[w-1, assignment[r], assignment[c]]
            prob *= factor
            # log_prob += math.log(factor, 2)
        return prob
        # return 2**log_prob

    # --- Create the multi-dimensional unnormalized Prob array with k^n entries
    for index, _ in np.ndenumerate(Prob):   # np.ndenumerate allows to iterate over all indices of the numpy array
        Prob[index] = unnormalized_probability(index)
        # print index, Prob[index]

    # --- Calculate marginals by summing up
    F = []
    for i in range(n):
        inverse_axis = list(range(n))     # contains all variables except the one for which we calculate marginals
        del inverse_axis[i]         # Thus sums over all other "axis" of n-dimensional array
        row = np.sum(Prob, axis=tuple(inverse_axis))  # row contains unnormalized marginals for node i
        F.append(row)

    return row_normalize_matrix(F, norm='l1')     # not necessary to transform to numpy











