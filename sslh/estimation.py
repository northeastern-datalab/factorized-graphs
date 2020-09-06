"""
Estimation methods for Compatibility Estimation described in
    Factorized Graph Representations for Semi-Supervised Learning from Sparse Data (SIGMOD 2020)
    Krishna Kumar P., Paul Langton, Wolfgang Gatterbauer
    https://arxiv.org/abs/2003.02829

Author: Wolfgang Gatterbauer
License: Apache Software License

TODO: nicer to separate the 'estimateH' from 'define_energy_H' into separate methods
"""


import numpy as np
from numpy import linalg as LA
import scipy.sparse as sparse
from numbers import Number      # for isinstance(n, Number)
from scipy import optimize
import itertools
import warnings
from random import shuffle
import random
import copy
from time import time
from utils import (W_star,
                   to_centering_beliefs,
                   matrix_difference,
                   eps_convergence_linbp_parameterized)
from inference import linBP_symmetric_parameterized



def define_energy_H(method='DHE',
                    H_vec_observed=None,
                    weights=None, distance=1,
                    X=None, W=None, EC=True, variant=1,
                    alpha=0, beta=0):
    """Returns an energy function 'energy_H(H)' that is used by 'estimateH' to find the optimal H.
    Three options: 'MHE', 'DHE', 'LHE' (for Myopic / Distant / Linear Heterophily (Compatibiltiy) Estimation)
    Original variant: constructed from X, W with chosen methods (e.g. DHE: weights, distance, variant, EC, alpha, beta)
    New variant: allows to give H_vec_observed directly (to estimate it once on graph and use it for both energy and gradient)
    For 'LHE':
        uses: X, W, alpha, beta (alpha, beta not actively used)
        ignores: H_vec_observed, EC, distance, weights, variant
    For 'DHE':
        If 'H_vec_observed' is specified: uses only weights, distance
        else: also uses W, X, EC, variant, alpha, beta to calculate 'H_vec_observed'
    For 'DHE':
        If 'H_vec_observed' is specified: ignores everything
        else: uses W, X, EC, variant, alpha, beta to calculate 'H_vec_observed'

    Parameters
    ----------
    method : string (Default='DHE')
        Three options: 'MHE', 'DHE', 'LHE'
    H_vec_observed : list [H1, H2, ..., Hd]
        Hd being the row-normalized Markov matrix that is used for DHE
    weights : np array (Default = np.ones(distance))
        Notice that 'estimateH' allows to specify a single weight that is then transformed into an array via np.array([np.power(weights, i) for i in range(distance)])
    EC : Boolean (Default=True)  TODO: should better be renamed to NB (Non-Backtracking)
        If true, ignores the backtracking paths (NB = True)
    variant : int (Default=1)
        Mostly historical, variant=1 works best, see SIGMOD 2020 paper for variants
    alpha, beta : real (Default=0)
        Allows different normalizations (not used)
    """
    # TODO: think about centered X as input

    if H_vec_observed is not None:
        _, k = H_vec_observed[0].shape
    else:
        _, k = X.shape
    if weights is None:
        weights = np.ones(distance)
    assert len(weights) >= distance
    if alpha != 0 or beta != 0:
        W = W_star(W, alpha=alpha, beta=beta)

    if method in ('MHE', 'DHE') and H_vec_observed == None:
        H_vec_observed = H_observed(W, X, distance=distance, NB=EC, variant=variant)

    if method == 'MHE':
        def energy_H(H):
            return LA.norm(H_vec_observed[0] - H)

    elif method == 'DHE':
        # the following 5 path lengths are written explicitly in order to speed up calculation
        if distance == 1:
            def energy_H(H):
                return LA.norm(H_vec_observed[0] - H)

        elif distance == 2:
            def energy_H(H):
                return weights[0] * LA.norm(H_vec_observed[0] - H) \
                       + weights[1] * LA.norm(H_vec_observed[1] - H.dot(H))
        elif distance == 3:
            def energy_H(H):
                H2 = H.dot(H)
                return weights[0] * LA.norm(H_vec_observed[0] - H) \
                       + weights[1] * LA.norm(H_vec_observed[1] - H2) \
                       + weights[2] * LA.norm(H_vec_observed[2] - H2.dot(H))
        elif distance == 4:
            def energy_H(H):
                H2 = H.dot(H)
                H3 = H2.dot(H)
                return weights[0] * LA.norm(H_vec_observed[0] - H) \
                       + weights[1] * LA.norm(H_vec_observed[1] - H2) \
                       + weights[2] * LA.norm(H_vec_observed[2] - H3) \
                       + weights[3] * LA.norm(H_vec_observed[3] - H3.dot(H))
        elif distance == 5:
            def energy_H(H):
                H2 = H.dot(H)
                H3 = H2.dot(H)
                H4 = H3.dot(H)
                return weights[0] * LA.norm(H_vec_observed[0] - H) \
                       + weights[1] * LA.norm(H_vec_observed[1] - H2) \
                       + weights[2] * LA.norm(H_vec_observed[2] - H3) \
                       + weights[3] * LA.norm(H_vec_observed[3] - H4) \
                       + weights[4] * LA.norm(H_vec_observed[4] - H4.dot(H))
        else:
            def energy_H(H):
                S = 0
                for i in range(distance):
                    S += weights[i] * LA.norm(H_vec_observed[i] - LA.matrix_power(H, i + 1))
                return S

    elif method == 'LHE':
        def energy_H(H):
            return LA.norm(X - W.dot(X).dot(H))

    else:
        raise Exception("You specified a non-existing method")
    return energy_H



# default numberOfSplits = 4
# needs all the parameters required for propagation
def estimateH_baseline_parallel(X2, ind, W, numberOfSplits=4, numMax=20, ignore_rows=None, method='noecho', variant=1, distance=1, EC=False, randomize=False, weights=None,
                                initial_H0=None, constraints=False, alpha=0., beta=0., gamma=1., doubly_stochastic=True):
    """Estimation of H, using train-test methods and minimizing the aggregated accuracies over several splits

    TODO: should be integrated with the _serial function; only minor modifications

        Parameters
        ----------
        X2 : [n x k] np array
            seed belief matrix
        ind: indices of seed beliefs in matrix X2
        W : [n x n] sparse.csr_matrix
            sparse weighted adjacency matrix
        numberOfSplits: splits for train-test
        methods:  string
            choices: ['MHE', 'LHE', 'LHEregular', 'LHE2']
            MHE: Myopic Heterophily Estimation
            LHE: Linear Heterophily Estimation
            LHEregular: Summation formula like in standard regularization framework (very slow)
            LHE2: with echo
        weights: list(Default = None)
            List of weights of various pathlength. w[0] is weight of neighbors (path length = 1), etc.
        distance: int (Default = 1)
            maximal distance (path length) that is included for estimations
        EC: Boolean (Default = True)
            For method MHE: If True: Calculates the neighbors in distance d while ignoring back-tracking paths
        variant: int (Default = 1)
            For method MHE for H_observed: variant 1: row-normalized. (2: symmetric normalized. 3: symmetric scaled)
        randomize: Boolean (Default = False)
            If True, then permutes the initial h0 vector a few times for estimation, then picks the best one.
            An attempt to deal with non-convexity
        initial_H0: list of list of float (Default=None)
            Option to specify a list of initial values for h0 from which to start the optimization when searching for best H
        constraints: Boolean (Default = False)
            True constrains all independent entries for H in [0, 1] (unfortunately does not constrain all entries in H)
        alpha, beta: Float (Default = 0)
            Optional normalization of propagation matrix


        Returns
        -------
        H : [k x k] np array
            Symmetric compatibility matrix

        """

    n, k = X2.shape
    if doubly_stochastic:
        k_star = k * (k - 1) // 2  # integer division, returns integer
    else:
        k_star = k * (k - 1)

    if weights is None:
        weights = np.ones(distance)

    indices = list(ind)
    halfLen = int(len(indices) / 2)

    X_train_split = {}
    X_test_split = {}
    train_split_idx = {}
    test_split_idx = {}

    print('Baseline parallel over {} splits'.format(numberOfSplits))
    for iter in range(numberOfSplits):
        shuffle(indices)
        train_split_idx[iter] = indices[:halfLen]  # indices of train set
        test_split_idx[iter] = indices[halfLen:]  # indices of test set

        X_train_split[iter] = copy.deepcopy(X2)
        for i in range(n):
            if i not in train_split_idx[iter]:
                X_train_split[iter][i] = np.zeros(k)


        X_test_split[iter] = copy.deepcopy(X2)
        for i in range(n):
            if i not in test_split_idx[iter]:
                X_test_split[iter][i] = np.zeros(k)


    def energy_h(h):
        H = transform_hToH(h, k, doubly_stochastic=doubly_stochastic)

        p = dict()
        q = dict()
        all_acc = np.zeros(numberOfSplits)

        for iter in range(numberOfSplits):
            q[iter] = Queue()  # create a queue object for returning values

            p[iter] = Process(target=calculate_accuracy, args=(H, X_train_split.get(iter), X_test_split.get(iter), train_split_idx.get(iter), test_split_idx.get(iter), W, q[iter]))
            p[iter].start()


        for iter in range(numberOfSplits):
            all_acc[iter] = q[iter].get()

        for iter in range(numberOfSplits):
            p[iter].join()


        return (-1 * np.average(all_acc))


    def calculate_accuracy(H, X_train, X_test, train_ind, test_ind, W, return_output, s=0.5):  # all that is needed to propagate
        H0c = to_centering_beliefs(H)


        eps_max = eps_convergence_linbp_parameterized(H0c, W,
                                                      method='noecho',
                                                      alpha=alpha, beta=beta, gamma=gamma,
                                                      X=X2)

        eps = s * eps_max

        F, actualIt, actualPercentageConverged = \
            linBP_symmetric_parameterized(X_train, W, H*eps,
                                          method='noecho',
                                          alpha=alpha, beta=beta, gamma=gamma,
                                          numMaxIt=numMax,
                                          convergencePercentage=0.99,
                                          convergenceThreshold=0.99,
                                          debug=2)

        n, k = F.shape
        for i in range(n):
            if i not in test_ind:
                F[i] = np.zeros(k)

        accuracy_X = matrix_difference(X_test, F, ignore_rows=list(train_ind), similarity='accuracy')

        print("Holdout accuracy: {}".format(accuracy_X))
        return_output.put(accuracy_X)           ## For Parallel
        # return accuracy_X      ## For Non Parallel


    def optimize_once(h0, energy_h):
        if constraints:
            bnds = [(0, 1) for _ in range(k_star)]
            result = optimize.minimize(fun=energy_h, x0=h0, method='L-BFGS-B', bounds=bnds)  # 'L-BFGS-B'
        else:
            # bnds = [(0, 1) for _ in range(k_star)]
            # result = optimize.minimize(fun=energy_h, x0=h0, method='SLSQP', bounds=bnds)  # 'SLSQP' 'BFGS'

            # result = optimize.minimize(fun=energy_h, x0=h0, method='SLSQP')  # 'SLSQP' 'BFGS'
            result = optimize.minimize(fun=energy_h, x0=h0, method='Nelder-Mead')  #  for non continuous

        h = result.get('x')
        E = result.get('fun')

        return h, E


    h0 = np.ones(k_star).dot(1 / k)  # use uninformative matrix to start with
    if not randomize and initial_H0 is None:
        h, _ = optimize_once(h0, energy_h)

    elif initial_H0 is not None:  # use supplied vector to start with
        E_dic = {}  # dictionary of Energy functions
        for h0 in initial_H0:
            h, fun = optimize_once(h0, energy_h)
            E_dic[fun] = h

        E_min = min(E_dic.keys())  # pick the minimum Energy
        h = E_dic[E_min]  # then look up the corresponding h vector

    # If randomize: then initiate the optimization with slightly permuated initial vectors (around default h0 value). Then pick the best one
    else:
        delta = 0.1
        permutations = list(itertools.product([-delta, delta], repeat=k_star))  # permutation for default value
        E_dic = {}  # dictionary of Energy functions
        for lst in permutations:
            h1 = h0 + np.array(lst)
            h, fun = optimize_once(h1, energy_h)
            E_dic[fun] = h

        E_min = min(E_dic.keys())  # pick the minimum Energy
        h = E_dic[E_min]  # then look up the corresponding h vector

    # final return statement
    return transform_hToH(h, k, doubly_stochastic=doubly_stochastic)








def estimateH_baseline_serial(X2, ind, W, numberOfSplits=4, numMax=20,
                              EC=False,
                              randomize=False,
                              weights=None,
                              initial_H0=None, constraints=False,
                              alpha=0., beta=0., gamma=1., doubly_stochastic=True):
    """Estimation of H, using train-test methods and minimizing the aggregated accuracies over several splits.
    Splits the labeled set into half test / half train. Does this numberOfSplits times.
    Needs all the parameters required for propagation.
    TODO: why does it need EC, weights, etc? Given it is a different function
    TODO: explain here some details of the parameters. E.g. does it split into two equally sized sets (seems to be implied by code below)

    Parameters
    ----------
    X2 : [n x k] np array
        seed belief matrix
    ind: indices of seed beliefs in matrix X2
    W : [n x n] sparse.csr_matrix
        sparse weighted adjacency matrix
    numberOfSplits : int (Default=4)
        how many splits for train-test evaluation, taking average accuracy over all splits
    numMax : int (Default=20)
        TODO: why is default==20, for experiments, did you use 10?
    EC : TODO delete?
    randomize: Boolean (Default = False)
        If True, then permutes the initial h0 vector a few times for estimation, then picks the best one.
        An attempt to deal with non-convexity
        TODO: why is this needed here?
    weights: TODO delete?
    initial_H0: list of list of float (Default=None)
        Option to specify a list of initial values for h0 from which to start the optimization when searching for best H
    constraints: Boolean (Default = False)
        True constrains all independent entries for H in [0, 1] (unfortunately does not constrain all entries in H)
    alpha, beta, gamme: Float (Default = 0)
        Optional normalization and clamping of propagation matrix
    doubly_stochastic: True

    Returns
    -------
    H : [k x k] np array
        Symmetric compatibility matrix
    """


    n, k = X2.shape
    if doubly_stochastic:
        k_star = k * (k - 1) // 2  # integer division, returns integer
    else:
        k_star = k * (k - 1)

    indices = list(ind)
    halfLen = int(len(indices) / 2)     # for each numberOfSplits, splits the labeled data half into test/train sets    TODO: is that correct

    X_train_split = {}
    X_test_split = {}
    train_split_idx = {}
    test_split_idx = {}

    for iter in range(numberOfSplits):
        shuffle(indices)
        train_split_idx[iter] = indices[:halfLen]  # indices of train set
        test_split_idx[iter] = indices[halfLen:]  # indices of test set

        X_train_split[iter] = copy.deepcopy(X2)
        for i in range(n):
            if i not in train_split_idx[iter]:
                X_train_split[iter][i] = np.zeros(k)

        X_test_split[iter] = copy.deepcopy(X2)
        for i in range(n):
            if i not in test_split_idx[iter]:
                X_test_split[iter][i] = np.zeros(k)

    def energy_h(h):
        """energy function to minimize as the negative average accuracy over all splits"""
        H = transform_hToH(h, k, doubly_stochastic=doubly_stochastic)
        all_acc = np.zeros(numberOfSplits)
        for iter in range(numberOfSplits):
            all_acc[iter] = calculate_accuracy(H, X_train_split.get(iter), X_test_split.get(iter), train_split_idx.get(iter), test_split_idx.get(iter), W, numMax)
        return (-1 * np.average(all_acc))

    def calculate_accuracy(H, X_train, X_test, train_ind, test_ind, W, s=0.5):
        """Propagates from X_train numMax times, calculates accuracy over X_test
        """
        H0c = to_centering_beliefs(H)
        eps_max = eps_convergence_linbp_parameterized(H0c, W,  # TODO: an optimized version could attempt to calculate the spectral radius fewer times and re-use it for multiple splits
                                                      method='noecho',
                                                      alpha=alpha, beta=beta, gamma=gamma,
                                                      X=X2)
        eps = s * eps_max
        F, actualIt, actualPercentageConverged = linBP_symmetric_parameterized(X_train, W, H*eps,
                                                                               method='noecho',
                                                                               alpha=alpha, beta=beta, gamma=gamma,
                                                                               numMaxIt=numMax,
                                                                               convergencePercentage=0.99,
                                                                               convergenceThreshold=0.99,
                                                                               debug=2)
        n, k = F.shape
        for i in range(n):
            if i not in test_ind:
                F[i] = np.zeros(k)

        # TODO For label imbalance, better to use CLASSWISE (macro-averaging) here
        accuracy_X = matrix_difference(X_test, F, ignore_rows=list(train_ind), similarity='accuracy')
        # print("accuracy now is {}".format(accuracy_X))
        return accuracy_X

    def optimize_once(h0, energy_h):
        if constraints:
            bnds = [(0, 1) for _ in range(k_star)]
            result = optimize.minimize(fun=energy_h, x0=h0, method='L-BFGS-B', bounds=bnds)  # 'L-BFGS-B'
        else:
            # bnds = [(0, 1) for _ in range(k_star)]
            # result = optimize.minimize(fun=energy_h, x0=h0, method='SLSQP', bounds=bnds)  # 'SLSQP' 'BFGS'
            # result = optimize.minimize(fun=energy_h, x0=h0, method='SLSQP')  # 'SLSQP' 'BFGS'
            result = optimize.minimize(fun=energy_h, x0=h0, method='Nelder-Mead')  # for non continuous optimization, other approaches don't work as well
        h = result.get('x')
        E = result.get('fun')
        return h, E

    h0 = np.ones(k_star).dot(1 / k)  # use uninformative matrix to start with
    if not randomize and initial_H0 is None:
        h, _ = optimize_once(h0, energy_h)

    elif initial_H0 is not None:  # use supplied vector to start with
        E_dic = {}  # dictionary of Energy functions
        for h0 in initial_H0:
            h, fun = optimize_once(h0, energy_h)
            E_dic[fun] = h

        E_min = min(E_dic.keys())  # pick the minimum Energy
        h = E_dic[E_min]  # then look up the corresponding h vector

    # If randomize: then initiate the optimization with slightly permuated initial vectors (around default h0 value). Then pick the best one
    else:
        delta = 0.1
        permutations = list(itertools.product([-delta, delta], repeat=k_star))  # permutation for default value
        E_dic = {}  # dictionary of Energy functions
        for lst in permutations:
            h1 = h0 + np.array(lst)
            h, fun = optimize_once(h1, energy_h)
            E_dic[fun] = h

        E_min = min(E_dic.keys())  # pick the minimum Energy
        h = E_dic[E_min]  # then look up the corresponding h vector

    # final return statement
    return transform_hToH(h, k, doubly_stochastic=doubly_stochastic)



def transform_hToH(h_vector, k, doubly_stochastic=True):
    """Transforms a parameter vector for a k dimensional symmetric stochastic matrix into the matrix.
    Allows the optimization problem to become unconstrained.
    Used by the various estimation functions.

    doubly_stochastic: Boolean
    - k*(k-1)/2 parameters if True (creates a doubly stochastic matrix)
    - k*(k-1) parameters if False (creates a row stochastic matrix)

    H(1,1)  *       *       *
    H(2,1)  H(2,2)  *       *
    H(3,1)  H(3.2)  H(3,3)  *
    *       *       *       *

    h(0)    *       *       *
    h(1)    h(3)    *       *
    h(2)    h(4)    h(5)    *
    *       *       *       *

    """
    if np.isnan(h_vector).any() or (np.abs(h_vector) > 10e10).any():
        print("Problem in 'transform_hToH' input:", h_vector)

    if k == 3 and doubly_stochastic:      # slight speed-up
        return np.array([[h_vector[0], h_vector[1], 1 - h_vector[0] - h_vector[1]],
                         [h_vector[1], h_vector[2], 1 - h_vector[1] - h_vector[2]],
                         [1 - h_vector[0] - h_vector[1], 1 - h_vector[1] - h_vector[2], h_vector[0] + 2 * h_vector[1] + h_vector[2] - 1]])
    elif doubly_stochastic:
        assert len(h_vector) == k*(k-1)//2
        H = np.zeros((k, k))
        for j in range(0, k-1):
            s = int((k - (1+j)/2)*j)          # starting index for column
            for i in range(0, j):
                H[i, j] = H[j, i]
            for i in range(j, k-1):
                H[i, j] = h_vector[s+i-j]
        for j in range(0, k-1):
            H[k-1, j] = 1 - H[:, j].sum()
            H[j, k-1] = H[k-1,j]
        H[k-1, k-1] = 1 - H[:, k-1].sum()
    else:
        assert len(h_vector) == k*(k-1)
        H = np.zeros((k, k))
        for i in range(0, k-1):
            H[:,i] = h_vector[i*k:i*k+k]
        H[:, k-1] = list(map(lambda r: 1-sum(r), H))
    return H



def transform_HToh(H0, k=None, doubly_stochastic=True):
    """Extracts the k* free parameters from a k*k dimensional row stochastic matrix
    """
    _, k = H0.shape
    h0_vec = list()
    if doubly_stochastic:
        for j in range(k-1):
            for i in np.arange(j, k-1):
                h0_vec.append(H0[i][j])
    else:
        for i in range(k-1):
            h0_vec.append(H0[:, i])
    return h0_vec



def define_gradient_energy_H(H_vec_observed, weights=None, distance=1):
    """Returns the gradient for the energy function 'gradient_energy_H(H)' that is constructed from H_vec_observed.
    Input: All observed H_vec_observed, the weight vector and maximal distance
    """
    length = min(len(H_vec_observed), distance)
    if weights is None:
        weights = np.ones(length)
    assert len(weights) >= distance

    def gradient_energy_H(H):
        S = 0
        for i in range(length):
            i2 = i+1                # 1 indexing
            M = H_vec_observed[i]   # 0 indexing
            Si = i2 * LA.matrix_power(H, 2*i2-1)    # 1 indexing
            for j in range(i2):
                Si -= matrix_power(H, j).dot(M).dot(matrix_power(H, i2-j-1))  # 1 indexing      Also requires wrapper that makes sure that H^0 = I
            S += weights[i] * Si    # 0 indexing
        return 2*S

    return gradient_energy_H



def matrix_power(M, j):
    """H^0 of matrices with 0s on diagonal don't return the identify matrix.
    But that's needed in define_gradient_energy_H. Thus this wrapper function"""
    if j == 0:
        return np.identity(M.shape[0])
    else:
        return LA.matrix_power(M, j)



def derivative_H_to_h(H, doubly_stochastic=True):
    """Calculates the partial derivative of a kxk doubly stochastic matrix with regard to the k(k-1)/2 free parameters
    TODO: don't remember if the else is correct for row-stochastic values
    """
    (_, k) = H.shape
    h0_vec = list()
    if doubly_stochastic:
        for j in range(k-1):
            for i in np.arange(j, k-1):
                if i == j:
                    value = H[i][j]           - H[i][-1] - H[-1][j]                       + H[-1][-1]
                    h0_vec.append(value)
                else:
                    value = H[i][j] + H[j][i] - H[i][-1] - H[-1][j] - H[j][-1] - H[-1][i] + 2 * H[-1][-1]
                    h0_vec.append(value)
        return np.array(h0_vec)
    else:
        for i in range(k):
            for j in np.arange(0, k-1):
                if i == j:
                    value = H[i][j]           - H[i][-1] - H[-1][j]                       + H[-1][-1]
                    h0_vec.append(value)
                else:
                    value = H[i][j] + H[j][i] - H[i][-1] - H[-1][j] - H[j][-1] - H[-1][i] + 2 * H[-1][-1]
                    h0_vec.append(value)
        return np.array(h0_vec)



def calculate_H_entry(i, j, k):
    """Returns the index of the free parameter for entry (i ,j) of a symmetric doubly stochastic matrix.
    Not applicable to last row and last column
    """
    assert i < k-1 and j < k-1, "unique index of the free parameters only exists for entries except last row or column"
    if i < j:
        return calculate_H_entry(j, i, k)
    else:
        return int(i+j*(k-1.5-j/2))



# def check_Diagonal(j, k):
#     """determines whether the jth free parameter in a kxk symmetric doubly stochastic matrix is at the diagonal or not
#     """
#     if i < j:
#         return calculate_H_entry(j, i, k)
#     else:
#         return int(i+j*(k-1.5-j/2))



def create_constraints(k, doubly_stochastic=True):
    """Create constraints for the k^* vector, i.e. the free parameters for a k x k dimensional stochastic matrix.
    Epsilon away from the actual borer used, because the optimization did not obey the constraints exactly
    """
    epsilon = 0.00001
    if doubly_stochastic:
        if k == 3:
            cons = (
                {'type': 'ineq', 'fun': lambda x: x[0] - epsilon},
                {'type': 'ineq', 'fun': lambda x: x[1] - epsilon},
                {'type': 'ineq', 'fun': lambda x: x[2] - epsilon},
                {'type': 'ineq', 'fun': lambda x: 1 - x[0] - x[1] - epsilon},
                {'type': 'ineq', 'fun': lambda x: 1 - x[1] - x[2] - epsilon},
                {'type': 'ineq', 'fun': lambda x: x[0] + 2 * x[1] + x[2] - 1 - epsilon})
        else:
            k_star = k*(k-1)//2
            constraint = '('
            for i in range(k_star):     # 1. constraints for all free parameters
                constraint += "{{'type': 'ineq', 'fun': lambda x: x[{0}] - {1} }},\n".format(i, epsilon)
            for j in range(k-1):        # 2. constraints for last row, except last cell
                s = '{} '.format(1-epsilon)
                for i in range(k-1):
                    s += ' - x[{}] '.format(calculate_H_entry(i, j, k))
                constraint += "{{'type': 'ineq', 'fun': lambda x: {} }},\n".format(s)
            s = '{} '.format(2-k-epsilon)   # 3. constraint for last cell
            for i in range(k-1):
                for j in range(k - 1):
                    s += ' + x[{}] '.format(calculate_H_entry(i, j, k))
            constraint += "{{'type': 'ineq', 'fun': lambda x: {} }})".format(s)
            # print("\n\n{}".format(constraint))
            cons = eval(constraint)
    else:
        k_star = k*(k-1)
        constraint = '('
        for i in range(k_star):     # 1. constraints for all free parameters
            constraint += "{{'type': 'ineq', 'fun': lambda x: x[{0}] - {1} }},\n".format(i, epsilon)


        for i in range(k):        # 2. constraints for last column
            s = '{} '.format(1-epsilon)
            for j in range(k-1):
                s += ' - x[{}] '.format(j)
            constraint += "{{'type': 'ineq', 'fun': lambda x: {} }},\n".format(s)

        constraint += ")"
        cons = eval(constraint)
    return cons



def estimateH(X, W, method='DHE',
              weights=None, distance=1, EC=True, variant=1,
              randomize=False, delta=0.1,
              randomrestarts=None,
              initial_H0=None,
              initial_h0=None,
              constraints=False,
              alpha=0, beta=0,
              logarithm=False,
              gradient=False,
              verbose=False,
              doubly_stochastic=True,
              initial_permutationFlag=False,
              initial_permutation=None,
              returnExtraStats = False,
              return_min_energy=False,
              increamentalRandomRestartsFlag=False,
              increamentalRandomRestarts = None,
              ):
    """Estimation of symmetric stochastic H, with various methods

    Parameters
    ----------
    X : [n x k] np array
        seed belief matrix
    W : [n x n] sparse.csr_matrix
        sparse weighted adjacency matrix
    method:  string
        choices
        'MHE': Myopic Heterophily Estimation, new name: MCE
        'DHE': Distant Heterophily Estimation, new name: DCE
        'LHE': Linear Heterophily Estimation, new name: LCE
        ('LHEregular': Summation formula like in standard regularization framework (very slow)... not supported anymore)
        ('LHE2': with echo ... not supported anymore)
    weights: list(Default = None)
        List of weights of various pathlength. w[0] is weight of neighbors (path length = 1), etc.
        Alternatively, just a single number that gets transformed into an array later
    distance: int (Default = 1)
        maximal distance (path length) that is included for estimations (except for MHE)
    EC: Boolean (Default = True)
        For method DHE: If True: Calculates the neighbors in distance d while ignoring back-tracking paths
        TODO: rename to NB for non-backtracking paths
    variant: int (Default = 1)
        For method MHE for H_observed: variant 1: row-normalized. (2: symmetric normalized. 3: symmetric scaled)
    randomize: Boolean (Default = False)
        If True, then permutes the initial h0 vector a few times for estimation, then picks the best one.
        2^k_star initial vectors (each one slightly permutes the intial values aroun the default uninformative value)
        Approach to deal with non-convexity for longer paths
    delta: Float (Default = 0.1)
        Optional parameter that determines the starting point for random initial points
    randomrestarts: int (Default = None)
        allows to use fixed number of restarts which are randomly sampled from the randomize approach
    initial_H0: value of H0 to initialize with (used to be a vector, now a k times k matrix)
        Option to specify a initial H0 to start the optimization
    initial_h0: value of doubly stochastic parameterized h0 to initialize with (originally used to be a vector, this parameter restores this option)
        Option to specify a initial H0 to start the optimization
        H0 has precedence over h0
    constraints: Boolean (Default = False)
        True constrains all independent entries for H in [0, 1] (unfortunately does not constrain all entries in H)
        TODO: Constraints are not working correctly. It seems, minimize enforces constraints as soft with a penalty function. And that screws up something
    alpha, beta: Float (Default = 0)
        Optional normalization of propagation matrix
    logarithm: Boolean (Default = False)
        True transforms the energy function with logarithm before optimizing
        TODO: remove
    gradient: Boolean (Default = False)
        True uses the gradiant calculation to speed up calculation for MHE and DHE
    verbose: Boolean (Default = False)
        prints extra infos: vector being optimized, resultant vector, minimum energy
    doubly_stochastic: Boolean (Default = True)
        False to estimate a row stochastic matrix with k(k-1) parameters instead of k(k-1)/2.
        Do not modify this variable unless the graph has significant skew towards certain classes over others
    initial_permutationFlag: Boolean (Default = False)
        True if DCEr is supplied with a list of  initial restart vectors
    initial_permutation: [] (Default = None)
        list of initial random restarts, each restart is a vector of k_star variables
    returnExtraStats: Boolean (Default = False)
        flag to return time and number of iterations taken by optimizer
    return_min_energy: Boolean (Default = False)
        If True, returns the optimal h vector and fun (instead of just the H matrix)
    increamentalRandomRestartsFlag                              # TODO remove, poorly done, could be just one variable with flag and default as None
    increamentalRandomRestarts = use the supplied restart points    # also still adds additional permutations to it?


    Returns
    -------
    H : [k x k] np array
        Symmetric compatibility matrix

    Notes
    -----
    Retrieves an energy function from 'define_energy_H', and then uses a common minimization routine
    with a trick to optimize over k(k-1)/2 independent entries of H function
    """
    _, k = X.shape
    end = " "   # time initializer, TODO remove

    k_star = k * (k - 1)

    if doubly_stochastic:
        k_star = k_star // 2  # integer division, returns integer. Example: has 3 parameters for 3x3 H matrix

    if weights is None:
        weights = np.ones(distance)
    elif isinstance(weights, Number):       # allow scalar to be passed as argument
        weights = np.array([np.power(weights, i) for i in range(distance)])
    assert len(weights) >= distance

    if initial_H0 is not None:
        initial_H0 = np.atleast_2d(initial_H0)  # requires 2d array
        h0 = transform_HToh(initial_H0, k, doubly_stochastic=doubly_stochastic)
    elif initial_h0 is None:
        h0 = np.ones(k_star).dot(1 / k)     # use uninformative matrix to start with
    else:
        h0 = initial_h0

    H_vec_observed = H_observed(W, X, distance=distance, NB=EC, variant=variant)    # get normalized path statistics P_hat

    energy_H = define_energy_H(W=W, X=X, method=method, weights=weights, alpha=alpha,   # TODO: remove W X as input later once alpha and beta not used anymore
                               beta=beta, distance=distance, EC=EC, variant=variant,
                               H_vec_observed=H_vec_observed)

    def energy_h(h):
        """changes parameters for energy function from matrix H to free parameters in array"""
        H = transform_hToH(h, k, doubly_stochastic=doubly_stochastic)
        if not logarithm:           # TODO: remove logarithm if not used anymore
            return energy_H(H)
        else:
            return np.log(energy_H(H))

    if gradient:
        if method not in ['DHE', 'MHE']:
            warnings.warn("\nGradient only defined for DCE")

        gradient_energy_H = define_gradient_energy_H(H_vec_observed=H_vec_observed, weights=weights, distance=distance)

        def gradient_energy_h(h):
            """changes parameters for gradient from matrix H to free parameters in array"""
            H = transform_hToH(h, k, doubly_stochastic=doubly_stochastic)
            g = gradient_energy_H(H)

            if np.isnan(g).any():   # TODO obsolete, delete
                print("Gradient with nan:")
                print(g)
                print("Original H:")
                print(H)
                print("Original h:")
                print(h)
                print("H_vec observed:")
                print(H_vec_observed)

            h = derivative_H_to_h(g, doubly_stochastic=doubly_stochastic)
            return h

    PRINTINTERMEDIATE = False       # option to print intermediate results from optimizer (for debugging)
    global Nfeval, permutations
    Nfeval = 1

    def callbackfct(x):             # print intermediate results, commented out in non-gradient loops below
        global Nfeval
        if PRINTINTERMEDIATE:
            np.set_printoptions(precision=4)
            print('{0:4d}   {2} {1}   '.format(Nfeval, energy_h(x), x))
            print('Iter: {}'.format(Nfeval))

        Nfeval += 1

    bnds = [(0,1) for i in range(k_star)]   # optional bounds for variables to be used in optimization function

    if constraints:
        cons = create_constraints(k)

    def optimize_once(h0, energy_h):
        """actual optimization step that can be repeated multiple times from random starting point"""
        if gradient:
            # using gradients:
            # BFGS: slightly slower than SLSQP, but converges better (can escape from weak saddle points), even finds optimum from some wrong points
            # SLSQP: slightly faster than BFGS, but gets more likely stuck in saddle points
            # L-BFGS-B: no seemingly advantage. Cannot deal with constraints although specifed
            # CG: considerably slower :(
            # Newton-CG: considerably slower :(
            if constraints:         # TODO not yet tested
                result = optimize.minimize(fun=energy_h, x0=h0, method='SLSQP',
                                           constraints=cons,
                                           jac=gradient_energy_h,
                                           # bounds=bnds,
                                           callback=callbackfct,
                                           )  # 'SLSQP' 'BFGS'
            else:
                result = optimize.minimize(fun=energy_h, x0=h0, method='BFGS',
                                           jac = gradient_energy_h,
                                           # bounds=bnds,               # bounds make it perform worse. Problem of optimization implementation
                                           callback = callbackfct,
                                           )  # 'SLSQP' 'BFGS'
            # print('Iter: {}'.format(Nfeval))
            h = result.get('x')
            E = result.get('fun')
            return h, E
        else:
            if constraints:
                # TODO: constraints are not working correctly. Problem seems to be with actual method 'SLSQP': (1) ignoring the constraints
                # TODO: and (2) even missing local optima it finds without those constraints ...
                result = optimize.minimize(fun=energy_h, x0=h0, method='SLSQP',
                                           constraints=cons,
                                           # bounds=bnds,
                                           callback=callbackfct,
                                           )  # 'SLSQP' 'BFGS'
            else:
                result = optimize.minimize(fun=energy_h, x0=h0, method='SLSQP',
                                           # bounds=bnds,                       # bounds make it perform worse. Problem of optimization implementation
                                           callback = callbackfct,
                                           )  # 'SLSQP' 'BFGS'
            h = result.get('x')
            E = result.get('fun')
            return h, E

    if not randomize:        # use supplied vector uninformative vector to start with
        start = time()
        h, fun = optimize_once(h0, energy_h)
        end = time() - start
        if verbose:
            print("Initial:{} Result:{} Energy:{}".format(np.round(h0, decimals=3), np.round(h, decimals=3), fun))
            # print("DCE fun: {}".format(fun))
            print("Time taken by energy optimize: {}".format(str(end)))

    else:
        if initial_permutationFlag:                 # TODO: allows explicit starting point. Cleaner to use without the flag
            permutations = initial_permutation
        else:
            # If randomize: then initiate the optimization with slightly permuted initial vectors (around default h0 value).
            # Then pick the best one result after optimization
            # Original variant used in total 2^k_star initial vectors, each one for each of the "quadrants" around the default values:
            #   permutations = list(itertools.product([-delta, delta], repeat=k_star))    # permutation for default value

            # TODO: This code part below is not done well. Also runs into coupon collector's problem.
            #   Problem it tried to solve is We cant hard code to 10 random restarts if none is specified
            #   Does not allow to specify the GT matrix as initial starting point, just to compare the ability of variant to recover it is local optimum
            # TODO: old likely obsolete comment: "What happens when pruneRandom and Gradient is True for LCE and MCE"
            if randomrestarts == None:
                randomrestarts = k_star
            if increamentalRandomRestartsFlag:              # TODO: remove, as the code below adds to an initial permutation?
                permutations = increamentalRandomRestarts
            else:
                permutations = []
            while len(permutations) < randomrestarts:
                perm = []
                for _ in range(k_star):
                    perm.append(random.choice([-delta, delta]))
                if perm not in permutations:            # TODO: Coupon collector's problem, what happens if 10 specified, but only 8 quadrants?
                    permutations.append(perm)
            # print("permutations length: {}, check: {}, {}".format(len(permutations), k*(k-1)/2, 2**(k*(k-1)/2)))

        # print("JUST AFTER Perms ", permutations)
        E_dic = {}  # dictionary of Energy functions
        for lst in permutations:
            Nfeval = 1
            h1 = h0 + np.array(lst)

            h, fun = optimize_once(h1, energy_h)
            E_dic[fun] = h                                  # resolve ties randomly

            if verbose:
                print("Initial:{} Result:{} Energy:{}".format(h1, np.round(h,decimals=3), fun))
                # print("  h0: {}".format(h1))
                # print("   h: {}".format(h))
                print(" fun: {}".format(fun))

        fun = min(E_dic.keys())   # pick the minimum Energy
        h = E_dic[fun]             # then look up the corresponding h vector
        # print("DCEr fun: {}".format(fun))

    if returnExtraStats:
        return transform_hToH(h, k, doubly_stochastic=doubly_stochastic), end, Nfeval   #TODO:remove End, Nfeval
    elif increamentalRandomRestarts:
        return [transform_hToH(h, k, doubly_stochastic=doubly_stochastic), permutations]  # TODO:remove End, Nfeval, permutations
    elif return_min_energy:
        return h, fun
    else:
        return transform_hToH(h, k, doubly_stochastic=doubly_stochastic)



def H_observed(W, X, distance, NB=True, variant=1):
    """Calculates a list [H1, H2, ..., Hd] with Hd being the row-normalized Markov matrix that is used for DHE
    Parameters distance and NB are passed through to 'M_estimation'
    Notice index starts at 1, in contrast to M_observed
    variant == 1: row-stochastic
    variant == 2: symmetric, normalized like LGC
    variant == 3: divides by sum of matrix

    SIGMOD 2020 refers to a 'two-step process' for estimating compatibilities.
    This method can be thought of as step 1, or calculating the graph statistics P-hat
    """
    n, n2 = W.shape
    n3, k = X.shape
    assert (n == n2) & (n == n3)
    assert distance in range(1, 10)
    assert variant in [1, 2, 3]

    M_vec = M_observed(W, X, distance, NB=NB)
    H_vec = []
    for M in M_vec[1:]:
        Mrowtot = M.dot(np.ones([k]))       # row total for M. problem of "sum": returns integer data type (does not work with np.power)
        Mrowtot[Mrowtot == 0] = 1           # don't normalize rows that have no entries
        if variant == 1:
            D_inv = np.diag(np.power(Mrowtot, -1))          # 'sparse.diags' does not work
            H_vec.append(D_inv.dot(M))
        elif variant == 2:
            D_inv_red = np.diag(np.power(Mrowtot, -0.5))
            H_vec.append((D_inv_red.dot(M)).dot(D_inv_red))
        else:   # variant == 3
            if np.sum(M) == 0:                  # TODO
                print("\ndivision by zero because M is \n{}\nfor\n{}".format(M, M_vec))
                H_vec.append(M + 1/k)
                print("normalizer is \n{}\nfor{}".format(normalizer, H_vec))
            else:
                normalizer = 1 / (np.sum(M) / k)
                H_vec.append(M.dot(normalizer))
    return H_vec



def M_observed(W, X, distance, NB=True):
    """Calculates a list [M0, M1, M2, ..., Md] with Md = X^T W^d X where d=distance
    Optional with or without NB
    Assumes W is symmetric
    Intermediate result Nd = W^d X for memoization (then Md = X^T Nd)
    Tip: Important trick for speed: never multiply W with W directly
    Notice index starts at 0, in contrast to H_observed"""
    n, n2 = W.shape
    n3, k = X.shape
    assert (n == n2) & (n == n3)
    assert isinstance(distance, int) & (n >= 0)

    N_vec = []
    M_vec = []

    N_vec.append(X)
    M_vec.append(X.transpose().dot(N_vec[-1]))

    if distance >= 1:
        N_vec.append(W.dot(X))
        M_vec.append(X.transpose().dot(N_vec[-1]))

    if distance >= 2:
        if not NB:
            N_vec.append(W.dot(N_vec[-1]))
        else:
            D = sparse.diags(W.dot(np.ones(n, dtype=int)), 0, dtype=int)
            D_I = D - sparse.diags(np.ones(n, dtype=int), 0, dtype=int)
            N_vec.append(W.dot(N_vec[-1]) - D.dot(X))
        M_vec.append(X.transpose().dot(N_vec[-1]))

    for _ in range(3, distance+1):
        if not NB:
            N_vec.append(W.dot(N_vec[-1]))
        else:
            N_vec.append(W.dot(N_vec[-1]) - D_I.dot(N_vec[-2]))
        M_vec.append(X.transpose().dot(N_vec[-1]))

    return M_vec
