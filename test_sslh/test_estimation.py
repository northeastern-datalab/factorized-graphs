"""
Test class for 'sslh/estimation'
Author: Wolfgang Gatterbauer
"""

import sys
sys.path.append('./../sslh')
import numpy as np
import scipy.sparse as sparse
import time
import random
from dill.source import getsource
from numpy import linalg as LA
from utils import (to_explicit_bool_vector,
                   to_centering_beliefs,
                   from_dictionary_beliefs,
                   replace_fraction_of_rows,
                   create_parameterized_H,
                   create_parameterized_alpha)
from estimation import (transform_hToH,
                        transform_HToh,
                        M_observed,
                        H_observed,
                        define_energy_H,
                        define_gradient_energy_H,
                        derivative_H_to_h,
                        calculate_H_entry,
                        create_constraints,
                        estimateH,
                        estimateH_baseline_serial)
from graphGenerator import (planted_distribution_model_H)


def test_DotProduct():
    """Just illustrates simple matrix dot product"""
    print("\n-- test_DotProduct(): '.dot', '.transpose()' --")
    a = np.array([[1, 2, 3]])
    b = np.array([[4, 5, 6]])

    print("a:\n{}".format(a))
    print("b:\n{}\n".format(b))
    print("a.dot(b.transpose()):\n{}\n".format(a.dot(b.transpose())))
    print("a.transpose().dot(b):\n{}\n".format(a.transpose().dot(b)))



def test_Diff_matrices():
    """Creates two parameterized Hs, shows that standardized versions are identical"""
    print("\n-- test_Diff_matrices(): 'create_parameterized_H', 'to_centering_beliefs', 'np.std', 'LA.norm' --")
    h = 2
    H0 = create_parameterized_H(3, h, symmetric=True)
    print("H0:\n{}\n".format(H0))

    H0c = to_centering_beliefs(H0)
    print("H0c (centered):\n{}\n".format(H0c))

    std_H0 = np.std(H0)
    print("std(H0): {}".format(std_H0))
    std_H0c = np.std(H0c)
    print("std(H0c): {}\n".format(std_H0c))
    H0c_s = H0c.dot(1/std_H0c)
    print("H0c_s (standardized centered):\n{}\n".format(H0c_s))

    H1 = create_parameterized_H(3, h*4, symmetric=True)
    print("H1 (4 times stronger potential):\n{}\n".format(H1))
    H1c = to_centering_beliefs(H1)
    H1c_s = H1c.dot(1 / np.std(H1c))
    print("H1c_s (standardized centered):\n{}\n".format(H1c_s))

    diff = LA.norm(H0c_s - H1c_s)
    print("LA.norm(H0c_s - H1c_s) is quasi 0:\n{}\n".format(diff))



def test_hToH_and_HToh():
    """Illustrate transform_hToH"""
    print("\n-- test_hToH_and_HToh(): 'transform_hToH', 'transform_HToh' --")
    h = np.array([0.2, 0.3, 0.4])
    H = transform_hToH(h, 3)
    print("h:{}".format(h))
    print("H:\n{}".format(H))
    h = transform_HToh(H)
    print("h:{}\n".format(h))

    h = np.array([1, 2, 3, 4, 5, 6])
    H = transform_hToH(h, 4)
    print("h:{}".format(h))
    print("H:\n{}".format(H))
    h = transform_HToh(H)
    print("h:{}\n".format(h))

    h = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    H = transform_hToH(h, 5)
    print("h:{}".format(h))
    print("H:\n{}".format(H))
    h = transform_HToh(H)
    print("h:{}\n".format(h))

    h = np.array([1./5, 1./5, 1./5, 1./5, 1./5, 1./5, 1./5, 1./5, 1./5, 1./5])
    H = transform_hToH(h, 5)
    print("h:{}".format(h))
    print("H:\n{}\n".format(H))



def test_M_observed():
    """Illustrate M_observed: non-backtracking or not
    Also shows that W^2 is denser for powerlaw graphs than uniform
    """
    print("\n-- test_M_observed(): 'M_observed', uses: 'planted_distribution_model_H' --")

    # --- Parameters for graph
    n = 3000
    a = 1
    h = 8
    d = 10  # variant 2
    d = 2   # variant 1
    k = 3
    distribution = 'powerlaw'   # variant 2
    distribution = 'uniform'    # variant 1
    exponent = -0.5

    alpha0 = np.array([a, 1., 1.])
    alpha0 = alpha0 / np.sum(alpha0)
    H0 = create_parameterized_H(k, h, symmetric=True)

    # --- Create graph
    RANDOMSEED = None    # For repeatability
    random.seed(RANDOMSEED)  # seeds some other python random generator
    np.random.seed(seed=RANDOMSEED)  # seeds the actually used numpy random generator; both are used and thus needed

    W, Xd = planted_distribution_model_H(n, alpha=alpha0, H=H0, d_out=d,
                                              distribution=distribution,
                                              exponent=exponent,
                                              directed=False,
                                              debug=False)
    X0 = from_dictionary_beliefs(Xd)

    # --- Print results
    distance = 8

    M_vec = M_observed(W, X0, distance=distance, NB=False)
    M_vec_EC = M_observed(W, X0, distance=distance, NB=True)

    print("Graph with n={} nodes and uniform d={} degrees".format(n, d))
    print("\nSum of entries and first rows of M_vec (without NB)")
    for i, M in enumerate(M_vec):                         # M_vec[1:] to skip the first entry in list
        print("{}: {}, {}".format(i, np.sum(M), M[0]))

    print("\nSum of entries and first rows of M_vec (with NB)")
    for i, M in enumerate(M_vec_EC):
        print("{}: {}, {}".format(i, np.sum(M), M[0]))

    if True:
        print("\nFull matrices:")
        print("M_vec")
        for i, M in enumerate(M_vec):  # skip the first entry in list
            print("{}: \n{}".format(i, M))

        print("\nM_vec_EC")
        for i, M in enumerate(M_vec_EC):  # skip the first entry in list
            print("{}: \n{}".format(i, M))



def test_H_observed():
    """Illustrate H_observed"""
    print("\n\n-- test_H_observed(): 'H_observed', uses: 'planted_distribution_model_H' --")

    # --- Parameters for graph
    n = 3000
    a = 1
    h = 8
    d = 2
    k = 3
    distribution = 'uniform'

    alpha0 = np.array([a, 1., 1.])
    alpha0 = alpha0 / np.sum(alpha0)
    H0 = create_parameterized_H(k, h, symmetric=True)

    # --- Create graph
    RANDOMSEED = None  # For repeatability
    random.seed(RANDOMSEED)  # seeds some other python random generator
    np.random.seed(seed=RANDOMSEED)  # seeds the actually used numpy random generator; both are used and thus needed

    W, Xd = planted_distribution_model_H(n, alpha=alpha0, H=H0, d_out=d,
                                              distribution=distribution,
                                              exponent=None,
                                              directed=False,
                                              debug=False)
    X0 = from_dictionary_beliefs(Xd)

    # --- Print first rows of matrices
    distance = 8

    print("First rows of powers of H0:")
    for k in range(1, distance + 1):
        print("{}: {}".format(k, np.linalg.matrix_power(H0, k)[0]))

    H_vec = H_observed(W, X0, distance=distance, NB=False)
    H_vec_EC = H_observed(W, X0, distance=distance, NB=True)

    print("First rows of H_vec (without NB)")
    for i, H in enumerate(H_vec):  # skip the first entry in list
        print("{}: {}".format(i+1, H[0]))

    print("First rows of H_vec (with NB)")
    for i, H in enumerate(H_vec_EC):
        print("{}: {}".format(i+1, H[0]))

    # --- Print just the top entry in first row (easier to compare)
    h_vec = []
    for k in range(1, distance + 1):
        h_vec.append(np.max(np.linalg.matrix_power(H0, k)[0]))

    hrow_vec = []
    for H in H_vec:
        hrow_vec.append(np.max(H[0]))

    hrow_EC_vec = []
    for H in H_vec_EC:
        hrow_EC_vec.append(np.max(H[0]))

    print("\nh_vec:\n{}".format(np.around(h_vec, 3)))
    print("hrow_vec (estimated without NB):\n{}".format(np.around(hrow_vec, 3)))
    print("hrow_EC_vec (estimated with NB):\n{}".format(np.around(hrow_EC_vec, 3)))



def test_H_observed_EC2_variants():
    """Illustrate the variants of H_observed"""
    print("\n\n-- test_H_observed_EC2_variants(): 'H_observed', 'M_observed', uses: 'planted_distribution_model_H' --")

    # --- Parameters for graph
    n = 3000
    a = 1
    h = 8
    d = 2
    k = 3
    f = 0.2
    distribution = 'uniform'

    alpha0 = np.array([a, 1., 1.])
    alpha0 = alpha0 / np.sum(alpha0)
    H0 = create_parameterized_H(k, h, symmetric=True)

    # --- Create graph
    RANDOMSEED = None  # For repeatability
    random.seed(RANDOMSEED)  # seeds some other python random generator
    np.random.seed(seed=RANDOMSEED)  # seeds the actually used numpy random generator; both are used and thus needed

    W, Xd = planted_distribution_model_H(n, alpha=alpha0, H=H0, d_out=d,
                                              distribution=distribution,
                                              exponent=None,
                                              directed=False,
                                              debug=False)
    X0 = from_dictionary_beliefs(Xd)
    X1, _ = replace_fraction_of_rows(X0, f, avoidNeighbors=False)

    # --- Print first rows of matrices
    distance = 3

    print("First rows of powers of H0:")
    for k in range(1, distance + 1):
        print("{}: {}".format(k, np.linalg.matrix_power(H0, k)[0]))

    print("\nNumber of observed edges between labels (M_observed):")
    M = M_observed(W, X1, distance=distance, NB=True)
    print("M[0]:\n{}".format(M[0]))
    print("M[2]:\n{}".format(M[1]))

    for EC in [False, True]:
        for variant in [1, 2]:
            print("\nP (H observed): variant {} with EC={}".format(variant, EC))
            H_vec = H_observed(W, X1, distance=distance, NB=EC, variant=variant)
            for i, H in enumerate(H_vec):
                print("{}:\n{}".format(i, H))



def test_estimate_synthetic():
    print("\n\n-- test_estimate_synthetic(): 'estimateH', uses: 'M_observed', 'planted_distribution_model_H', --")

    # --- Parameters for graph
    n = 1000
    a = 1
    h = 8
    d = 25
    k = 3
    distribution = 'powerlaw'
    exponent = -0.3
    f = 0.05
    print("n={}, a={},d={}, f={}".format(n, a, d, f))

    alpha0 = np.array([a, 1., 1.])
    alpha0 = alpha0 / np.sum(alpha0)
    H0 = create_parameterized_H(k, h, symmetric=True)
    print("H0:\n{}".format(H0))

    # --- Create graph
    RANDOMSEED = None    # For repeatability
    random.seed(RANDOMSEED)  # seeds some other python random generator
    np.random.seed(seed=RANDOMSEED)  # seeds the actually used numpy random generator; both are used and thus needed

    W, Xd = planted_distribution_model_H(n, alpha=alpha0, H=H0, d_out=d,
                                              distribution=distribution,
                                              exponent=exponent,
                                              directed=False,
                                              debug=False)
    X0 = from_dictionary_beliefs(Xd)
    X1, ind = replace_fraction_of_rows(X0, 1-f)

    # --- Print some neighbor statistics
    M_vec = M_observed(W, X0, distance=3, NB=True)
    print("\nNeighbor statistics in fully labeled graph:")
    print("M^(1): direct neighbors:\n{}".format(M_vec[1]))
    print("M^(2): distance-2 neighbors:\n{}".format(M_vec[2]))
    print("M^(3): distance-3 neighbors:\n{}".format(M_vec[3]))

    # --- MHE ---
    print("\nMHE: Estimate H based on X0 (fully labeled graph):")
    start = time.time()
    H1 = estimateH(X0, W, method='MHE', variant=1)
    H2 = estimateH(X0, W, method='MHE', variant=2)
    H3 = estimateH(X0, W, method='MHE', variant=3)
    time_est = time.time() - start
    print("Estimated H based on X0 (MHE), variant 1:\n{}".format(H1))
    print("Estimated H based on X0 (MHE), variant 2:\n{}".format(H2))
    print("Estimated H based on X0 (MHE), variant 3:\n{}".format(H3))
    print("Time for all three variants:{}".format(time_est))

    print("\nMHE: Estimate H based on X1 with f={}:".format(f))
    start = time.time()
    H1 = estimateH(X1, W, method='MHE', variant=1)
    H2 = estimateH(X1, W, method='MHE', variant=2)
    H3 = estimateH(X1, W, method='MHE', variant=3)
    time_est = time.time() - start
    print("Estimated H based on X1 (MHE), variant 1:\n{}".format(H1))
    print("Estimated H based on X1 (MHE), variant 2:\n{}".format(H2))
    print("Estimated H based on X1 (MHE), variant 3:\n{}".format(H3))
    print("Time for all three variants:{}".format(time_est))

    print("\nMHE, variant=1: Estimate H based on X1 with f={}, but with initial correct vector:")
    weight = [0, 0, 0, 0, 0]    # ignored for MHE
    initial_h0 = [0.1, 0.8, 0.1]
    H5 = estimateH(X1, W, method='MHE', weights=weight)
    H5_r = estimateH(X1, W, method='MHE', weights=weight, randomize=True)
    H5_i = estimateH(X1, W, method='MHE', weights=weight, initial_H0=transform_hToH(initial_h0, 3))
    print("Estimated H based on X5 only (MHE): \n{}".format(H5))
    print("Estimated H based on X5 only (MHE), randomize:\n{}".format(H5_r))
    print("Estimated H based on X5 only (MHE), initial=GT:\n{}".format(H5_i))

    # --- DHE ---
    print("\nDHE: Estimate H based on X1 with f={}:".format(f))
    start = time.time()
    H1 = estimateH(X1, W, method='DHE', variant=1, distance=1)
    H2 = estimateH(X1, W, method='DHE', variant=2, distance=1)
    H3 = estimateH(X1, W, method='DHE', variant=3, distance=1)
    time_est = time.time() - start
    print("Estimated H based on X1 (DHE, distance=1), variant 1:\n{}".format(H1))
    print("Estimated H based on X1 (DHE, distance=1), variant 2:\n{}".format(H2))
    print("Estimated H based on X1 (DHE, distance=1), variant 3:\n{}".format(H3))
    print("Time for all three variants:{}".format(time_est))

    # --- LHE ---
    print("\nLHE: Estimate H based on X1 with f={}:".format(f))
    start = time.time()
    H1 = estimateH(X1, W, method='LHE')
    time_est = time.time() - start
    print("Estimated H based on X1 (LHE):\n{}".format(H1))
    print("Time for LHE:{}".format(time_est))

    # --- Baseline holdout method ---
    f2 = 0.5
    X2, ind2 = replace_fraction_of_rows(X0, 1-f2)
    print("\nHoldout method: Estimate H based on X2 with f={}):".format(f2))
    start = time.time()
    H2 = estimateH_baseline_serial(X2=X2, ind=ind2, W=W, numberOfSplits=1, numMax=10)
    time_est = time.time() - start
    print("Estimated H based on X2 (Holdout method) with f={}:\n{}".format(f2, H2))
    print("Time for Holdout method:{}".format(time_est))            # TODO: result suggests this method does not work?



def test_derivative_H():
    print("\n\n-- test_derivative_H(): 'derivative_H_to_h', --")

    listOfMatrices = []
    listOfMatrices.append(
        np.array(
            [[1, 0, 0],
             [0, 0, 0],
             [0, 0, 0]])
    )
    listOfMatrices.append(
        np.array(
            [[0, 0, 0],
             [1, 0, 0],
             [0, 0, 0]])
    )
    listOfMatrices.append(
        np.array(
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 0]])
    )
    listOfMatrices.append(
        np.array(
            [[0, 0, 1],
             [0, 0, 1],
             [1, 1, 0]])
    )
    listOfMatrices.append(
        np.array(
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 1]])
    )
    listOfMatrices.append(
        np.array(
            [[1, 0, 0],
             [1, 1, 0],
             [0, 0, 0]])
    )
    listOfMatrices.append(
        np.array(
            [[1, 0, 0, 0],
             [1, 1, 0, 0],
             [1, 1, 1, 0],
             [0, 0, 0, 0]])
    )
    listOfMatrices.append(
        np.array(
            [[0, 0, 0, 1],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]])
    )
    listOfMatrices.append(
        np.array(
            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 1]])
    )

    for P in listOfMatrices:
        print("Matrix P:\n{}".format(P))
        h = derivative_H_to_h(P)
        print("Derivative: {}".format(h))





def test_gradient():
    print("\n-- 'define_gradient_energy_H, define_energy_H, uses: planted_distribution_model_H, H_observed, M_observed, --")

    # --- Parameters for graph
    n = 1000
    a = 1
    h = 8
    d = 25
    k = 3
    distribution = 'powerlaw'
    exponent = -0.3

    alpha0 = np.array([a, 1., 1.])
    alpha0 = alpha0 / np.sum(alpha0)
    H0 = create_parameterized_H(k, h, symmetric=True)
    f = 0.5
    print("Graph n={}, d={}, f={}".format(n, d, f))
    print("H0:\n{}\n".format(H0))

    # --- Create graph
    RANDOMSEED = None    # For repeatability
    random.seed(RANDOMSEED)  # seeds some other python random generator
    np.random.seed(seed=RANDOMSEED)  # seeds the actually used numpy random generator; both are used and thus needed

    W, Xd = planted_distribution_model_H(n, alpha=alpha0, H=H0, d_out=d,
                                              distribution=distribution,
                                              exponent=exponent,
                                              directed=False,
                                              debug=False)
    X0 = from_dictionary_beliefs(Xd)
    X1, ind = replace_fraction_of_rows(X0, 1-f)

    # --- M_vec, H_vec statistics
    distance = 5

    print("M_vec:")
    M_vec = M_observed(W, X1, distance=distance)
    for i, M in enumerate(M_vec):
        print("{}:\n{}".format(i, M))

    print("H_vec:")
    H_vec = H_observed(W, X1, distance=distance)
    for i, H in enumerate(H_vec):
        print("{}:\n{}".format(i, H))

    # --- Gradient at multiple points for distance 1
    print("\n=Defining the gradient function with distance 1")
    distance = 1
    weights = [1, 0, 0, 0, 0]
    gradient_energy_H = define_gradient_energy_H(H_vec_observed=H_vec, weights=weights, distance=distance)
    energy_H = define_energy_H(weights=weights, distance=1, H_vec_observed=H_vec)

    H_actual = H_vec[0]
    print("1st example point: H_actual (row-stochastic frequencies of neighbors):\n{}".format(H_actual))
    e = energy_H(H_actual)
    g = gradient_energy_H(H_actual)
    h = derivative_H_to_h(g)
    print("energy: ", e)
    print("gradient:\n{}".format(g))
    print("projected gradient: ", h)

    H_point = transform_hToH(np.array([0.2, 0.6, 0.2]), 3)
    print("\n2nd example point: H_point:\n{}".format(H_point))
    e = energy_H(H_point)
    g = gradient_energy_H(H_point)
    h = derivative_H_to_h(g)
    print("energy: ", e)
    print("gradient:\n{}".format(g))
    print("projected gradient: ", h)

    H_point2 = H_point-0.45*g
    print("\n3rd example point in opposite direction of gradient: H_point2=H_point-0.45*gradient:\n{}".format(H_point2))
    e = energy_H(H_point2)
    g = gradient_energy_H(H_point2)
    h = derivative_H_to_h(g)
    print("energy: ", e)
    print("gradient:\n{}".format(g))
    print("projected gradient: ", h)


    # --- Gradient at multiple points for distance 5
    distance = 5
    weights = [0, 0, 0, 0, 1]
    print("\n= Defining the gradient function with distance={} and weights={}".format(distance, weights))
    gradient_energy_H = define_gradient_energy_H(H_vec_observed=H_vec, weights=weights, distance=distance)
    energy_H = define_energy_H(weights=weights, distance=1, H_vec_observed=H_vec)

    H_actual = H_vec[0]
    print("1st point: H_actual:\n{}".format(H_actual))
    e = energy_H(H_actual)
    g = gradient_energy_H(H_actual)
    h = derivative_H_to_h(g)
    print("energy: ", e)
    print("gradient:\n{}".format(g))
    print("projected gradient: ", h)

    H_point = transform_hToH(np.array([0.2, 0.6, 0.2]), 3)
    print("\n2nd point: H_point:\n{}".format(H_point))
    e = energy_H(H_point)
    g = gradient_energy_H(H_point)
    h = derivative_H_to_h(g)
    print("energy: ", e)
    print("gradient:\n{}".format(g))
    print("projected gradient: ", h)

    H_point2 = H_point-1.5*g
    print("\n3rd point in opposite direction of gradient: H_point2:\n{}".format(H_point2))
    e = energy_H(H_point2)
    g = gradient_energy_H(H_point2)
    h = derivative_H_to_h(g)
    print("energy: ", e)
    print("gradient:\n{}".format(g))
    print("projected gradient: ", h)



def test_gradient_optimization():
    print("\n-- 'estimateH, define_gradient_energy_H, define_energy_H, uses: planted_distribution_model_H, H_observed, M_observed, --")

    # --- Parameters for graph
    n = 1000
    a = 1
    h = 8
    d = 25
    k = 3
    distribution = 'powerlaw'
    exponent = -0.3

    alpha0 = np.array([a, 1., 1.])
    alpha0 = alpha0 / np.sum(alpha0)
    H0 = create_parameterized_H(k, h, symmetric=True)
    f = 0.1
    print("Graph n={}, d={}, f={}".format(n, d, f))
    print("H0:\n{}".format(H0))

    # --- Create graph
    RANDOMSEED = None  # For repeatability
    random.seed(RANDOMSEED)  # seeds some other python random generator
    np.random.seed(seed=RANDOMSEED)  # seeds the actually used numpy random generator; both are used and thus needed
    W, Xd = planted_distribution_model_H(n, alpha=alpha0, H=H0, d_out=d,
                                         distribution=distribution,
                                         exponent=exponent,
                                         directed=False,
                                         debug=False)
    X0 = from_dictionary_beliefs(Xd)
    X1, ind = replace_fraction_of_rows(X0, 1 - f)

    # --- M_vec, H_vec statistics
    distance = 5
    print("\nH_vec_observed:")
    H_vec = H_observed(W, X1, distance=distance)
    for i, H in enumerate(H_vec):
        print("{}:\n{}".format(i, H))


    # --- estimate_H based on distance 1
    print("\n= Estimate H based on X1 and distance=1 (old without or with gradient):")
    distance = 1
    weights = [1, 0, 0, 0, 0]

    start = time.time()
    H1 = estimateH(X1, W, distance = distance, weights=weights, gradient=False)
    time_est = time.time() - start
    print("Estimated H without gradient:\n{}".format(H1))
    print("Time :{}".format(time_est))

    start = time.time()
    H2 = estimateH(X1, W, distance = distance, weights=weights, gradient=True)
    time_est = time.time() - start
    print("Estimated H with gradient:\n{}".format(H2))
    print("Time :{}".format(time_est))


    # --- estimate_H based on distance 5 and uninformative point
    print("\n= Estimate H based on X1 and distance=5 (ignoring distances 1-4) from various points (old without or with gradient):")
    print("From uninformative point (all methods get stuck, even with gradient !!!:")
    distance = 5
    weights = [0, 0, 0, 0, 1]
    h0 = np.ones(3).dot(1 / k)  # use uninformative matrix to start with

    start = time.time()
    H1 = estimateH(X1, W, distance = distance, weights=weights, gradient=False)
    time_est = time.time() - start
    print("Estimated H without gradient:\n{}".format(H1))
    print("Time :{}".format(time_est))

    start = time.time()
    H2 = estimateH(X1, W, distance = distance, weights=weights, gradient=True)
    time_est = time.time() - start
    print("Estimated H with gradient:\n{}".format(H2))
    print("Time :{}".format(time_est))

    gradient_energy_H = define_gradient_energy_H(H_vec_observed=H_vec, weights=weights, distance=distance)
    g = gradient_energy_H(transform_hToH(h0, 3))
    h = derivative_H_to_h(g)
    print("Gradient at uninformative point:\n{}".format(g))
    print("Gradient at uninformative point: {}".format(h))
    energy_H = define_energy_H(H_vec_observed=H_vec, weights=weights, distance=distance)
    e = energy_H(H2)
    print("Energy at estimated point: {}".format(e))

    # --- estimate_H based on distance 5 and wrong point
    print("\n= From wrong point (gradient method with BFGS can fix it, SLSQP stays stuck !!!")
    distance = 5
    weights = [0, 0, 0, 0, 1]
    h0 = np.array([0.4, 0.3, 0.3])

    start = time.time()
    H1 = estimateH(X1, W, distance = distance, weights=weights, gradient=False, initial_h0=h0)
    time_est = time.time() - start
    print("Estimated H without gradient:\n{}".format(H1))
    print("Time :{}".format(time_est))

    start = time.time()
    H2 = estimateH(X1, W, distance = distance, weights=weights, gradient=True, initial_h0=h0)
    time_est = time.time() - start
    print("Estimated H with gradient:\n{}".format(H2))
    print("Time :{}".format(time_est))

    gradient_energy_H = define_gradient_energy_H(H_vec_observed=H_vec, weights=weights, distance=distance)
    g = gradient_energy_H(transform_hToH(h0, 3))
    h = derivative_H_to_h(g)
    print("Gradient at wrong point:\n{}".format(g))
    print("Gradient at wrong point: {}".format(h))
    energy_H = define_energy_H(H_vec_observed=H_vec, weights=weights, distance=distance)
    e = energy_H(H2)
    print("Energy at estimated point: {}".format(e))

    # --- estimate_H based on distance 5 and some closer point
    print("\n= From closer point (converges for BFGS, but not always for SLSQP!!!):")
    distance = 5
    weights = [0, 0, 0, 0, 1]
    h0 = np.array([0.3, 0.4, 0.3])

    start = time.time()
    H1 = estimateH(X1, W, distance = distance, weights=weights, gradient=False, initial_h0=h0)
    time_est = time.time() - start
    print("Estimated H without gradient:\n{}".format(H1))
    print("Time :{}".format(time_est))

    start = time.time()
    H2 = estimateH(X1, W, distance = distance, weights=weights, gradient=True, initial_h0=h0)
    time_est = time.time() - start
    print("Estimated H with gradient:\n{}".format(H2))
    print("Time :{}".format(time_est))

    gradient_energy_H = define_gradient_energy_H(H_vec_observed=H_vec, weights=weights, distance=distance)
    g = gradient_energy_H(transform_hToH(h0, 3))
    h = derivative_H_to_h(g)
    print("Gradient at closer point:\n{}".format(g))
    print("Gradient at closer point: {}".format(h))
    energy_H = define_energy_H(H_vec_observed=H_vec, weights=weights, distance=distance)
    e = energy_H(H2)
    print("Energy at estimated point: {}".format(e))

    # --- estimate_H based on distance 5 and some closer point
    print("\n= From even closer point:")
    distance = 5
    weights = [0, 0, 0, 0, 1]
    h0 = np.array([0.2, 0.4, 0.2])

    start = time.time()
    H1 = estimateH(X1, W, distance = distance, weights=weights, gradient=False, initial_h0=h0)
    time_est = time.time() - start
    print("Estimated H without gradient:\n{}".format(H1))
    print("Time :{}".format(time_est))

    start = time.time()
    H2 = estimateH(X1, W, distance = distance, weights=weights, gradient=True, initial_h0=h0)
    time_est = time.time() - start
    print("Estimated H with gradient:\n{}".format(H2))
    print("Time :{}".format(time_est))

    gradient_energy_H = define_gradient_energy_H(H_vec_observed=H_vec, weights=weights, distance=distance)
    g = gradient_energy_H(transform_hToH(h0, 3))
    h = derivative_H_to_h(g)
    print("Gradient at closer point:\n{}".format(g))
    print("Gradient at closer point: {}".format(h))
    energy_H = define_energy_H(H_vec_observed=H_vec, weights=weights, distance=distance)
    e = energy_H(H2)
    print("Energy at estimated point: {}".format(e))

    # --- estimate_H based on distance 5 and some closer point
    print("\n= Variant with constraints (constraints only work with SLSQP !!!):")

    start = time.time()
    H2 = estimateH(X1, W, distance = distance, weights=weights, gradient=True, initial_h0=h0, constraints=True)
    time_est = time.time() - start
    print("Estimated H with gradient and constraints:\n{}".format(H2))
    print("Time :{}".format(time_est))

    e = energy_H(H2)
    print("Energy at estimated point: {}".format(e))






def test_gradient_optimization2():
    print("\n-- 'estimateH, define_gradient_energy_H, define_energy_H, uses: planted_distribution_model_H, H_observed, M_observed, --")

    # --- Parameters for graph
    n = 10000
    a = 1
    h = 2
    d = 10
    k = 7
    distribution = 'powerlaw'
    exponent = -0.3

    np.set_printoptions(precision=4)

    alpha0 = create_parameterized_alpha(k, a)
    H0 = create_parameterized_H(k, h, symmetric=True)
    f = 0.02
    print("Graph n={}, d={}, f={}".format(n, d, f))
    print("H0:\n{}".format(H0))

    # --- Create graph
    RANDOMSEED = None  # For repeatability
    random.seed(RANDOMSEED)  # seeds some other python random generator
    np.random.seed(seed=RANDOMSEED)  # seeds the actually used numpy random generator; both are used and thus needed

    W, Xd = planted_distribution_model_H(n, alpha=alpha0, H=H0, d_out=d,
                                         distribution=distribution,
                                         exponent=exponent,
                                         directed=False,
                                         debug=False)
    X0 = from_dictionary_beliefs(Xd)
    X1, ind = replace_fraction_of_rows(X0, 1 - f)

    # --- M_vec, H_vec statistics
    distance = 5

    print("M_vec:")
    M_vec = M_observed(W, X1, distance=distance)
    for i, M in enumerate(M_vec):
        print("{}:\n{}".format(i, M))

    print("\nH_vec_observed:")
    H_vec = H_observed(W, X1, distance=distance)
    for i, H in enumerate(H_vec):
        print("{}:\n{}".format(i, H))

    # --- estimate_H based on distance 1 and uninformative point
    distance = 1
    weights = [1, 0, 0, 0, 0]
    print("\n= Estimate H based on X1 and distance={} from uninformative point:".format(distance))
    h0 = np.ones(int(k*(k-1)/2)).dot(1 / k)  # use uninformative matrix to start with
    energy_H = define_energy_H(H_vec_observed=H_vec, weights=weights, distance=distance)
    gradient_energy_H = define_gradient_energy_H(H_vec_observed=H_vec, weights=weights, distance=distance)

    start = time.time()
    H1 = estimateH(X1, W, distance = distance, weights=weights, gradient=False)
    time_est = time.time() - start
    print("Estimated H without gradient:\n{}".format(H1))
    print("Time :{}".format(time_est))
    e = energy_H(H1)
    print("Energy at estimated point: {}".format(e))

    start = time.time()
    H2 = estimateH(X1, W, distance = distance, weights=weights, gradient=True)
    time_est = time.time() - start
    print("Estimated H with gradient:\n{}".format(H2))
    print("Time :{}".format(time_est))
    e = energy_H(H2)
    print("Energy at estimated point: {}".format(e))

    G = gradient_energy_H(H2)
    h = derivative_H_to_h(G)
    print("Gradient matrix at estimated point:\n{}".format(G))
    print("Gradient vector at estimated point:\n{}".format(h))



def test_calculate_H_entry():
    print("\n1 --- 'calculate_H_entry'")
    print("Need to leave away the last column and row!\n")
    kList = [3, 4, 5, 8]
    for k in kList:
        print("\nk={}:".format(k))
        for i in range(k - 1):
            for j in range(k - 1):
                print('{},  '.format(calculate_H_entry(i, j, k)), end='')
            print()



def test_create_constraints():
    # create constraints, and evaluate all constraint functions for a given x vector
    print("\n1 --- 'create_constraints'")

    print("\nk=3:")
    k = 3
    r = create_constraints(k)
    v0 = np.ones(k)/k
    for i in r:
        print(i.get('fun')(v0))
    for i in r:
        print(getsource(i.get('fun')))

    print("\nk=4:")
    k = 4
    r = create_constraints(k)
    v0 = np.ones(k**2)/k
    for i in r:
        print(i.get('fun')(v0))

    print("\nk=5:")
    k = 5
    r = create_constraints(k)
    v0 = np.ones(k ** 2) / k
    for i in r:
        print(i.get('fun')(v0))

    print("\nk=6:")
    k = 6
    r = create_constraints(k)
    v0 = np.ones(k ** 2) / k
    for i in r:
        print(i.get('fun')(v0))

    print("\nk=5 (free parameters are all 0):")
    k = 5
    r = create_constraints(k)
    v0 = np.zeros(k ** 2) / k
    for i in r:
        print(i.get('fun')(v0))

    print("\nk=4 (free parameters are all different):")
    k = 4
    r = create_constraints(k)
    v0 = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 10]
    for i in r:
        print(i.get('fun')(v0))



def test_PaperExample():
    print("\n-- 'estimateH': Example graph for MHE vs LHE paper example --")
    CHOICE = 1
    if CHOICE == 1:         # graph example
        X = np.array([
            [0, 1],
            [0, 1],
            [1, 0],
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 0], ])
    elif CHOICE == 2:       # full graph
        X = np.array([
            [0, 1],
            [0, 1],
            [1, 0],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1], ])
    elif CHOICE == 3:       # no neighbors connected
        X = np.array([
            [0, 1],
            [0, 0],
            [1, 0],
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 0], ])

    Xb = to_explicit_bool_vector(X)
    X2c = to_centering_beliefs(X, ignoreZeroRows=True)  # try without
    X2cf = to_centering_beliefs(X, ignoreZeroRows=False)  # try without

    row = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3]
    col = [1, 4, 5, 2, 3, 5, 6, 4, 6, 6]
    row, col = row + col, col + row
    Ws = sparse.csr_matrix(([1]*len(row), (row, col)), shape=(7, 7))

    # _out_visualize_Graph(Ws, X, Xb=Xb, colorDisplay='explicit')

    print("W:\n{}".format(Ws.todense()))
    print("X:\n{}\n".format(X))

    start = time.time()
    H = estimateH(X, Ws, method='MHE')
    time_est = time.time() - start
    print("Estimated H (MHE):\n{}".format(H))
    print("Time :{}\n".format(time_est))

    start = time.time()
    H = estimateH(X, Ws, method='LHE')
    time_est = time.time() - start
    print("Estimated H (LHE):\n{}".format(H))
    print("Time :{}\n".format(time_est))

    start = time.time()
    H = estimateH(X, Ws, method='LHE', constraints=True)
    time_est = time.time() - start
    print("Estimated H (LHE) with constraints:\n{}".format(H))
    print("Time :{}\n".format(time_est))

    # start = time.time()
    # H = estimateH(X, Ws, method='LHEregular')
    # time_est = time.time() - start
    # print ("Estimated H (LHEregular):\n{}".format(H))
    # print ("Time :{}\n".format(time_est))
    #
    # start = time.time()
    # H = estimateH(X, Ws, method='LHE2')
    # time_est = time.time() - start
    # print ("Estimated H (LHE2):\n{}".format(H))
    # print ("Time :{}\n".format(time_est))

    print("= Variants with centered X -- ")
    start = time.time()
    H = estimateH(X2c, Ws, method='LHE')
    # print (X2c)
    time_est = time.time() - start
    print("Estimated H (LHE) with centering (while ignoring zero rows):\n{}".format(H))
    print("Time :{}\n".format(time_est))

    start = time.time()
    H = estimateH(X2cf, Ws, method='LHE')
    # print (X2cf)
    time_est = time.time() - start
    print("Estimated H (LHE) with centering (and NOT ignoring zero rows):\n{}".format(H))
    print("Time :{}\n".format(time_est))

    # start = time.time()
    # H = estimateH(X, Ws, method='LHE', epsilon=0.1)
    # time_est = time.time() - start
    # print ("Estimated H (LHE) with epsilon=0.1:\n{}".format(H))
    # print ("Time :{}\n".format(time_est))


# %%
if __name__ == '__main__':
    test_DotProduct()
    test_Diff_matrices()
    test_hToH_and_HToh()
    test_M_observed()
    test_H_observed()
    test_H_observed_EC2_variants()
    test_estimate_synthetic()
    test_derivative_H()
    test_gradient()
    test_gradient_optimization()
    test_gradient_optimization2()
    test_calculate_H_entry()
    test_create_constraints()
    test_PaperExample()

