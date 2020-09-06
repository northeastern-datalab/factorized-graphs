"""
Test class for 'sslh/utils'
Author: Wolfgang Gatterbauer
"""

import sys
sys.path.append('./../sslh')
from utils import (check_normalized_beliefs, check_centered_beliefs,
                   to_centering_beliefs, from_centering_beliefs,
                   check_dictionary_beliefs, from_dictionary_beliefs,
                   check_explicit_beliefs, to_dictionary_beliefs,
                   to_explicit_bool_vector, to_explicit_list,
                   matrix_difference, max_binary_matrix,
                   replace_fraction_of_rows,
                   row_normalize_matrix,
                   matrix_convergence_percentage,
                   degree_matrix,
                   calculate_potential_from_row_normalized,
                   create_parameterized_H, create_parameterized_alpha,
                   W_row, W_red, W_clamped, W_star,
                   approx_spectral_radius,
                   matrix_difference_classwise,
                   get_class_indices)
from graphGenerator import planted_distribution_model
from fileInteraction import load_X
from scipy.sparse import csr_matrix, kron
import numpy as np
import scipy.sparse as sps
import time



# -- Determine path to data irrespective of where the file is run from
from os.path import abspath, dirname, join
from inspect import getfile, currentframe
current_path = dirname(abspath(getfile(currentframe())))
data_directory = join(current_path, 'data/')
fig_directory = join(current_path, 'figs/')



def test_transform_beliefs():
    print("\n-- 'check_normalized_beliefs', 'to_centering_beliefs' --")
    X = np.array([[1.0001, 0, 0]])
    print("X:", X)
    assert check_normalized_beliefs(X)
    print("X centered:",to_centering_beliefs(X))

    Y = np.array([0.9999, 0, 0])
    print("Y:", Y)
    assert check_normalized_beliefs(Y)
    print("Y centered:", to_centering_beliefs(Y))

    Z = np.array([[1.001, 0, 0]])
    print("Z:", Z)
    assert not check_normalized_beliefs(Z)

    W = np.array([0.999, 0, 0])
    print("W:", W)
    assert not check_normalized_beliefs(W)

    print("\n-- 'check_centered_beliefs', 'from_centering_beliefs'")
    Xc = np.array([[1.0001, -1, 0]])
    print("Xc: ", Xc)
    assert check_centered_beliefs(Xc)
    print("Xc uncentered: ", from_centering_beliefs(Xc))

    Yc = np.array([0.9999, -1, 0])
    print("Yc: ", Yc)
    assert check_centered_beliefs(Yc)
    print("Yc uncentered: ", from_centering_beliefs(Yc))

    Zc = np.array([[1.001, -1, 0]])
    print("Zc: ", Zc)
    assert not check_centered_beliefs(Zc)

    Wc = np.array([0.999, -1, 0])
    print("Wc: ", Wc)
    assert not check_centered_beliefs(Wc)

    print("\n-- 'to_centering_beliefs', 'from_centering_beliefs' for matrices --")
    X = np.array([[1, 0, 0],
                 [0.8, 0.2, 0],
                 [1./3, 1./3, 1./3],
                 [0, 0, 1],
                 [0, 0, 1],
                 [0, 0, 0],
                 [0, 0, 0]])
    print("X original:\n", X)
    print("np.sum(X,1):\n", np.sum(X,1))
    print("X.sum(axis=1, keepdims=True):\n", X.sum(axis=1, keepdims=True))
    print("X.shape:", X.shape)
    print("len(X.shape): ", len(X.shape))

    Xc = to_centering_beliefs(X, ignoreZeroRows=True)
    print("X centered (ignoringZeroRows=True):\n", Xc)
    Y = from_centering_beliefs(Xc)
    print("X again un-centered:\n", Y)

    fileNameX = join(data_directory, 'Torus_X.csv')
    X, _, _ = load_X(fileNameX, n=8, zeroindexing=False)
    X = X.dot(0.1)
    print("\nCentered X for Torus example as input\n", X)
    Xc = from_centering_beliefs(X)
    print("X un-centered:\n", Xc)


    X = np.array(   [[1,0,0]])
    print("\nX original:\n", X)
    Xc = to_centering_beliefs(X)
    print("X centered:\n", Xc)
    Y = from_centering_beliefs(Xc)
    print("X back non-centered:\n", Y)

    X = np.array(   [1,0,0])
    print("\nX original:\n", X)
    print("np.sum(X,0):", np.sum(X,0))
    print("X.sum(axis=0, keepdims=True):", X.sum(axis=0, keepdims=True))
    print("X.shape: ", X.shape)
    print("len(X.shape): ", len(X.shape))



def test_dictionary_transform():
    print("\n-- 'check_dictionary_beliefs', 'from_dictionary_beliefs' --")
    Xd = {1: 1, 2: 2, 3: 3, 5: 1}
    print("Xd:", Xd)

    print("zeroindexing=True:")
    print("X:\n", from_dictionary_beliefs(Xd, n=None, k=None, zeroindexing=True))
    print("zeroindexing=False:")
    print("X:\n", from_dictionary_beliefs(Xd, n=None, k=None, zeroindexing=False))
    print("zeroindexing=True, n=7, k=5:")
    print("X:\n", from_dictionary_beliefs(Xd, n=7, k=5, zeroindexing=True))

    print("\nzeroindexing=False, fullBeliefs=True:")
    X1 = {1: 1, 2: 2, 3: 3, 4: 1}
    assert check_dictionary_beliefs(X1, n=None, k=None, zeroindexing=False, fullBeliefs=True)
    print("X1:", X1)

    print("zeroindexing=True, fullBeliefs=True:")
    X2 = {0: 0, 1: 1, 2: 2, 3: 0}
    assert check_dictionary_beliefs(X2, n=None, k=None, zeroindexing=True, fullBeliefs=True)
    print("X2:", X2)

    print("zeroindexing=True, fullBeliefs=False:")
    X3 = {0: 0, 1: 1, 2: 2, 4: 0}
    assert check_dictionary_beliefs(X3, n=None, k=None, zeroindexing=True, fullBeliefs=False)
    print("X3:", X3)

    print("zeroindexing=True, fullBeliefs=False:")
    X4 = {0: 1, 2: 2, 4: 0}
    assert check_dictionary_beliefs(X4, n=None, k=None, zeroindexing=True, fullBeliefs=False)
    print("X4:", X4)

    print("\nerrors:")
    X5 = {0: 0, 1: 1, 2: 3, 3: 0}
    print("X5:", X5)
    assert not check_dictionary_beliefs(X5, n=None, k=None, zeroindexing=False, fullBeliefs=True)
    X6 = {0: 1, 1: 1, 2: 2, 4: 1}
    print("X6:", X6)
    assert not check_dictionary_beliefs(X6, n=None, k=None, zeroindexing=True, fullBeliefs=True)


    print("\n-- 'check_explicit_beliefs', 'to_dictionary_beliefs' --")
    X = np.array(  [[1., 0, 0],
                    [0, 0, 0],
                    [0, 0, 1],
                    ])
    print("original X:\n", X)
    print("Stacked X:\n", np.hstack(X))
    print("List of X entires:\n", set(np.hstack(X)))
    print("Verify: ", set(np.hstack(X)) == {0, 1})
    print("Verify: ", {0., 1.} == {0, 1})
    assert check_explicit_beliefs(X)
    Y = np.array([1., 0, 0])
    assert check_explicit_beliefs(Y)
    Xd = to_dictionary_beliefs(X)
    print("Xd: ", Xd)


def test_to_explicit_vectors():
    print("\n-- 'to_explicit_bool_vector', 'to_explicit_list' --")
    fileNameX = join(data_directory, 'Torus_X.csv')
    X, _, _ = load_X(fileNameX, n=8, zeroindexing=False)
    print("Torus X:\n", X)
    Xb = to_explicit_bool_vector(X)
    print('Xb:\n', Xb)
    Xl = to_explicit_list(X)
    print('Xl:\n', Xl)

    Y = np.array(   [[0,0,0],
                     [0,0,1],
                     [0,1,1],
                     [0,0,-1],
                     [0,0,0.0001],
                     [0,0,0.001]])
    print("\nY:\n", Y)
    Yb = to_explicit_bool_vector(Y)
    print('Yb:\n', Yb)
    Yl = to_explicit_list(Y)
    print('Yl:\n', Yl)



def test_max_binary_matrix():
    print("\n-- 'max_binary_matrix' --")
    X = np.array(   [[1,0,0],
                     [10,8,5],
                     [1./3,1./3,1./3],
                     [0,0,1],
                     [0,0.9,1],
                     [0.5,0,0.5]])
    print("X original:\n", X)
    Xb = max_binary_matrix(X)
    print("X with winning classes (no tolerance):\n", Xb)
    Xb = max_binary_matrix(X, 0.2)
    print("X with winning classes (with 0.2 tolerance):\n", Xb)

    X = np.array(   [[10,9,0]])
    print("\nX original:\n", X)
    Xb = max_binary_matrix(X,2)
    print("X with winning classes (with 2 tolerance):\n", Xb)



def test_row_normalize_matrix():
    print("\n-- 'row_normalize_matrix' (l1, l2, zscores) --")
    v = np.array([1, 1, 0, 0, 0])
    print("original:\n ", v)
    print("l2:\n ", row_normalize_matrix(v, norm='l2'))
    print("l1:\n ", row_normalize_matrix(v, norm='l1'))
    print("zscores:\n ", row_normalize_matrix(v, norm='zscores'))

    v = np.array([1, 1, 1, 0, 0])
    print("\noriginal:\n ", v)
    print("l2:\n ", row_normalize_matrix(v, norm='l2'))
    print("l1 :\n ", row_normalize_matrix(v, norm='l1'))
    print("zscores:\n ", row_normalize_matrix(v, norm='zscores'))

    X = np.array(  [[1, 0, 0],
                    [0, 0, 0],
                    [1, -1, -1],
                    [1, -1, -1.1],
                    [1, -2, -3],])
    print("\noriginal:\n", X)
    print("l2:\n", row_normalize_matrix(X, norm='l2'))
    print("!!! Notice that l1 norm with negative values is counterintuitive: !!!")
    print("l1:\n", row_normalize_matrix(X, norm='l1'))
    print("zscores:\n", row_normalize_matrix(X, norm='zscores'))

    X = np.array([[0, 20, 0],
                  [21, 0, 0],
                  [0, 0, 14]])
    print("\noriginal:\n", X)
    print("l2:\n", row_normalize_matrix(X, norm='l2'))
    print("l1:\n", row_normalize_matrix(X, norm='l1'))
    print("zscores:\n", row_normalize_matrix(X, norm='zscores'))


    print("\n -- zscore and normalizing together --")
    v = np.array([1, 1, 0, 0, 0])
    print("original:\n  ", v)
    print("zscore:\n  ", row_normalize_matrix(v, norm='zscores'))
    print("normalized zscore:\n  ", \
        row_normalize_matrix(
            row_normalize_matrix(v, norm='zscores'), norm='l2'))
    print("normalized zscore normalized:\n  ", \
        row_normalize_matrix(
            row_normalize_matrix(
                row_normalize_matrix(v,norm='l2'), norm='zscores'), norm='l2'))

    X = np.array(  [[1, 0, 0],
                    [1, -1, -1],
                    [1, -1, -1.1],
                    [1, -2, -3],
                    [0, 0, 0],
                    [1,1,-1],
                    [1,1.1,-1],
                    [1,1,1]])
    print("\noriginal:\n", X)
    print("zscore:\n", row_normalize_matrix(X, norm='zscores'))
    print("normalized:\n", row_normalize_matrix(X, norm='l2'))
    print("normalized zscore:\n", \
        row_normalize_matrix(
            row_normalize_matrix(X, norm='zscores'), norm='l2'))
    print("normalized zscore normalized:\n", \
        row_normalize_matrix(
            row_normalize_matrix(
                row_normalize_matrix(X,norm='l2'), norm='zscores'), norm='l2'))
    print("zscore normalized zscore normalized:\n", \
        row_normalize_matrix(
            row_normalize_matrix(
                row_normalize_matrix(
                    row_normalize_matrix(X,norm='l2'), norm='zscores'), norm='l2'), norm='zscores'))



def test_degree_matrix():
    print("\n-- 'degree_matrix' --")
    print("- Directed case")
    row = [0, 0, 0, 1, 2, 3]
    col = [1, 2, 3, 4, 4, 4]
    weight = [2, 3, 4, 1, 2, 3]
    Ws = sps.csr_matrix((weight, (row, col)), shape=(5, 5))
    print("Ws:\n{}".format(Ws))
    print("W:\n{}".format(Ws.todense()))

    print("\nSquared=False")
    D_in = degree_matrix(Ws, indegree=True, squared=False)
    D_out = degree_matrix(Ws, indegree=False, squared=False)
    print("D_in (col sum):\n{}".format(D_in))
    print("D_in (col sum):\n{}".format(D_in.todense()))
    print("D_out (row sum):\n{}".format(D_out))

    print("\nSquared=True")
    D_in = degree_matrix(Ws, indegree=True)
    D_out = degree_matrix(Ws, indegree=False)
    print("D_in (col sum):\n{}".format(D_in))
    print("D_out (row sum):\n{}".format(D_out))

    print("\n- Undirected case (undirected=True)")
    row =    [0, 1, 0, 2, 1, 2, 2, 3,   3]
    col =    [1, 0, 2, 0, 2, 1, 3, 2,   1]
    weight = [1, 2, 2, 1, 1, 1, 1, 0.1, 0.5]
    Ws = sps.csr_matrix((weight, (row, col)), shape=(4, 4))
    print("Ws:\n{}".format(Ws))
    print("W:\n{}".format(Ws.todense()))

    print("\nSquared=False")
    print("D (undirected, in):\n{}".format(degree_matrix(Ws, undirected=True, indegree=True, squared=False).todense()))
    print("D (undirected, out):\n{}".format(degree_matrix(Ws, undirected=True, indegree=False, squared=False).todense()))

    print("\nSquared=True")
    print("D (undirected):\n{}".format(degree_matrix(Ws, undirected=True).todense()))


    print("\n- Undirected case with row-normalized matrix (undirected=True)")
    Wrow = W_row(Ws)
    print("Wrow:\n{}".format(Wrow))
    print("Wrow:\n{}".format(Wrow.todense()))
    Wrow2 = row_normalize_matrix(Ws.todense())
    print("Wrow2 (with 'row_normalize_matrix' after '.todense()':\n{}".format(Wrow2))

    print("\nSquared=False")
    print("D_row (undirected, in):\n{}".format(degree_matrix(Wrow, undirected=True, indegree=True, squared=False).todense()))
    print("D_row (undirected, out):\n{}".format(degree_matrix(Wrow, undirected=True, indegree=False, squared=False).todense()))

    print("\nSquared=True")
    print("D_row (undirected):\n{}".format(degree_matrix(Wrow, undirected=True).todense()))


    # -- Timing
    print("\nTiming with big random matrix (n=100k, d=10) and random weights")
    n = 100000
    d = 10
    row = np.random.randint(n, size=n*d)
    col = np.random.randint(n, size=n*d)
    weight = np.random.randint(1, 10, size=n*d)
    Ws = sps.csr_matrix((weight, (row, col)), shape=(n, n))
    if False:
        Ws.data[:] = [1]*len(Ws.data)                                 # faster by factor 3 if all degress have same weight 1

    start = time.time()
    D_in = degree_matrix(Ws, indegree=True)
    end = time.time()-start
    print("Time to calculate D_in:", end)



def test_matrix_difference_with_cosine_simililarity():
    print("\n-- 'matrix_difference' (cosine), 'row_normalize_matrix' --")
    print("k=3")
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([1, 1, 0])
    print("Cosine with original:\n  ", \
        matrix_difference(v1,
                          v1, similarity='cosine'))
    print("Cosine with original zscore:\n  ", \
        matrix_difference(row_normalize_matrix(v1, norm='zscores'),
                          row_normalize_matrix(v1, norm='zscores'), similarity='cosine'))
    print("Cosine with zscore :\n  ", \
        matrix_difference(v1,
                          row_normalize_matrix(v1, norm='zscores'), similarity='cosine'))
    print("Cosine with normal:\n  ", \
        matrix_difference(v1,
                          v2, similarity='cosine'))
    print("Cosine with normal after both zscore:\n  ", \
        matrix_difference(row_normalize_matrix(v1, norm='zscores'),
                          row_normalize_matrix(v2, norm='zscores'), similarity='cosine'))
    print("! Notice that average guessing leads to expectation of 0!")
    print("Cosine v1, v3:\n  ", \
        matrix_difference(v1,
                          v3, similarity='cosine'))
    print("Cosine v1, v3 after zscore:\n  ", \
        matrix_difference(row_normalize_matrix(v1, norm='zscores'),
                          row_normalize_matrix(v3, norm='zscores'), similarity='cosine'))

    print("\nk=5")
    v1 = np.array([1, 0, 0, 0, 0])
    v2 = np.array([0, 1, 0, 0, 0])
    v3 = np.array([1, 1, 0, 0, 0])
    v4 = np.array([0, 0, 0, 0, 0])
    print("Cosine with normal:\n  ", \
        matrix_difference(v1,
                          v2, similarity='cosine'))
    print("Cosine with normal after both zscore:\n  ", \
        matrix_difference(row_normalize_matrix(v1, norm='zscores'),
                          row_normalize_matrix(v2, norm='zscores'), similarity='cosine'))
    print("! Notice that average guessing leads to expectation of 0!")
    print("Cosine v1, v3:\n  ", \
        matrix_difference(v1,
                          v3, similarity='cosine'))
    print("Cosine v1, v3 after zscore:\n  ", \
        matrix_difference(row_normalize_matrix(v1, norm='zscores'),
                          row_normalize_matrix(v3, norm='zscores'), similarity='cosine'))
    print("Average Cos similarity partly zscore:\n  ", \
        matrix_difference(row_normalize_matrix(v1, norm='zscores'),
                          row_normalize_matrix(v3, norm='zscores'), similarity='cosine'))
    print("Cosine with 0-vector:\n  ", \
        matrix_difference(row_normalize_matrix(v1, norm='zscores'),
                          row_normalize_matrix(v4, norm='zscores'), similarity='cosine'))
    print()

    X = np.array([[1, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0],
                  [1, 1, 0, 0, 0]])
    Y = np.array([[1, 0, 0, 0, 0],
                  [1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [1, 1.1, 0, 0, 0]])
    print("X\n", X)
    print("Y\n", Y)
    Xs = row_normalize_matrix(X, norm='zscores')
    Ys = row_normalize_matrix(Y, norm='zscores')
    print("Xs\n", Xs)
    print("Ys\n", Ys)

    print("\nCosine original:\n  ", \
        matrix_difference(X,
                          Y, vector=True, similarity='cosine'))
    print("Cosine zscore:\n  ", \
        matrix_difference(Xs,
                          Ys, vector=True, similarity='cosine'))
    print("Average cosine zscore:\n  ", \
        matrix_difference(X,
                          Y, similarity='cosine'))



def test_matrix_difference_with_accuracy_etc():
    print("\n-- 'matrix_difference' (precision/recall/accuracy/cosine), 'max_binary_matrix' --")
    X_true = np.array([[2, 0, 0],
                       [2, 0, 2],
                       [0, 1, 0],
                       [0, 0, 3],
                       [0, 0, 3],
                       [1, 0, 2],
                       [0, 3, 3],
                       [0, 3, 3],
                       [0, 0, 3],
                       [0, 0, 3]
                       ])
    X_pred = np.array([[1, 1, 2],
                       [2, 1, 2],
                       [3, 4, 0],
                       [1, 1, 2],
                       [2, 1, 1],
                       [1, 2, 2],
                       [1, 2, 3],
                       [1, 2.99, 3],
                       [1, 2.8, 3],
                       [1, 2.99, 3],
                       ])
    X_true_b = max_binary_matrix(X_true)
    X_pred_b = max_binary_matrix(X_pred)
    X_pred_b1 = max_binary_matrix(X_pred, threshold=0.1)
    print("X_true:\n", X_true)
    print("X_pred:\n", X_pred)
    print("\nX_true binary:\n", X_true_b)
    print("X_pred binary:\n", X_pred_b)
    print("X_pred binary with threshold 0.1:\n", X_pred_b1)

    ind = list([])
    # ind = list([0, 1])
    # ind = list([1, 2, 3, 4, 5])
    # ind = list([0, 2, 3, 4, 5, 6])
    print("\nPrecision:\n", matrix_difference(X_true, X_pred_b1, ind, vector=True, similarity='precision'))
    # print("*** type:", type (matrix_difference(X_true, X_pred_b1, ind, vector=True, similarity='precision')))
    print("Recall:\n", matrix_difference(X_true, X_pred_b1, ind, vector=True, similarity='recall'))
    print("Accuracy:\n", matrix_difference(X_true, X_pred_b1, ind, vector=True, similarity='accuracy'))

    cosine_list = matrix_difference(X_true, X_pred, ind, vector=True, similarity='cosine')
    print("Cosine:\n", cosine_list)
    print("Cosine sorted:\n", sorted(cosine_list, reverse=True))

    print("\nPrecision:\n", matrix_difference(X_true, X_pred, ind, similarity='precision'))
    print("Recall:\n", matrix_difference(X_true, X_pred, ind, similarity='recall'))
    print("Accuracy:\n", matrix_difference(X_true, X_pred, ind))
    print("Cosine:\n", matrix_difference(X_true, X_pred, ind, similarity='cosine'))



def test_matrix_difference():
    print("\n-- 'matrix_difference' (cosine/cosine_ratio/l2), 'to_centering_beliefs' --")
    X0 = np.array([[2, 0, 0],
                   [2, 0, 2],
                   [0, 1, 0],
                   [0, 0, 3],
                   [0, 0, 3],
                   [1, 0, 2],
                   [0, 3, 3],
                   [0, 0, 0],
                   [0, 1, 0],
                   [0, 1, 0],
                   [9, 9, 9],
                   [9, 9, 9],
                   [100, 100, 100],])
    X1 = np.array([[1, 1, 2],
                   [2, 1, 2],
                   [3, 4, 0],
                   [1, 1, 2],
                   [2, 1, 1],
                   [1, 2, 2],
                   [1, 2, 3],
                   [0, 0, 0],
                   [1, 0, 0],
                   [0, 2, 0],
                   [9, 9, 9],
                   [8, 9, 9],
                   [100, 100, 101],])
    print("X0:\n", X0)
    print("X1:\n", X1)

    result = matrix_difference(X0, X1, similarity='cosine', vector=True)
    print("cosine:\n", result)
    result = matrix_difference(X0, X1, similarity='cosine_ratio', vector=True)
    print("cosine_ratio:\n", result)
    result = matrix_difference(X0, X1, similarity='l2', vector=True)
    print("l2:\n", result)

    X0 = np.array([[ 1.       ,   0.       ,   0.        ],
                  [ 0.30804075,  0.56206462,  0.12989463],
                  [ 0.32434628,  0.33782686,  0.33782686],
                  [ 0.30804075,  0.12989463,  0.56206462],
                  [ 0.14009173,  0.71981654,  0.14009173],
                  [ 0.32273419,  0.21860539,  0.45866042],
                  [ 0.33804084,  0.32391832,  0.33804084],
                  [ 0.45866042,  0.21860539,  0.32273419]])
    X1 = np.array([[ 1.      ,    0.      ,    0.        ],
                  [ 0.22382029,  0.45296374,  0.32321597],
                  [ 0.32434628,  0.33782686,  0.33782686],
                  [ 0.22382029,  0.32321597,  0.45296374],
                  [ 0.2466463 ,  0.5067074 ,  0.2466463 ],
                  [ 0.32273419,  0.21860539,  0.45866042],
                  [ 0.33804084,  0.32391832,  0.33804084],
                  [ 0.45866042,  0.21860539,  0.32273419]])
    print("\nX0:\n", X0)
    print("X1:\n", X1)

    result = matrix_difference(X0, X1, similarity='cosine_ratio', vector=True)
    print("cosine_ratio:\n", result)

    # X0z = row_normalize_matrix(X0, norm='zscores')
    # X1z = row_normalize_matrix(X1, norm='zscores')
    X0z = to_centering_beliefs(X0)
    X1z = to_centering_beliefs(X1)

    print("\nX0z:\n", X0z)
    print("X1z:\n", X1z)

    result = matrix_difference(X0z, X1z, similarity='cosine_ratio', vector=True)
    print("cosine_ratio zscores:\n", result)

    # actualPercentageConverged = matrix_convergence_percentage(X0z, X1z, threshold=convergenceCosineSimilarity)

    X0 = np.array([1, 0, 0])
    X1 = np.array([1, 1, 0])
    print("\nX0:\n", X0)
    print("X1:\n", X1)
    result = matrix_difference(X0, X1, similarity='cosine_ratio', vector=True)
    print("cosine_ratio zscores:\n", result)

    X0 = np.array([-30, -15, 45])
    X1 = np.array([-15, -30, 45])
    print("\nX0:\n", X0)
    print("X1:\n", X1)
    result = matrix_difference(X0, X1, similarity='cosine_ratio', vector=True)
    print("cosine_ratio zscores:\n", result)



def test_matrix_difference_classwise():
    print("\n-- 'matrix_difference' (cosine/cosine_ratio/l2), 'to_centering_beliefs' --")
    X0 = np.array([[2, 0, 0],
                   [2, 0, 2],
                   [0, 1, 0],
                   [0, 0, 3],
                   [0, 0, 3],
                   [1, 0, 2],
                   [0, 3, 3],
                   [0, 0, 0],
                   [0, 1, 0],
                   [0, 1, 0],
                   [9, 9, 9],
                   [9, 9, 9],
                   [100, 100, 100],])
    X1 = np.array([[1, 1, 2],
                   [2, 1, 2],
                   [3, 4, 0],
                   [1, 1, 2],
                   [2, 1, 1],
                   [1, 2, 2],
                   [1, 2, 3],
                   [0, 0, 0],
                   [1, 0, 0],
                   [0, 2, 0],
                   [9, 9, 9],
                   [8, 9, 9],
                   [100, 100, 101],])
    print("X0:\n", X0)
    print("X1:\n", X1)

    # result = matrix_difference(X0, X1, similarity='accuracy', vector=True)
    # print("accuracy:\n", result)
    result = matrix_difference(X0, X1, similarity='accuracy', vector=False)
    print("accuracy:\n", result)

    result = get_class_indices(X0, 0)
    print("indices for class 0 in X0:\n", result)

    result = matrix_difference_classwise(X0, X1, )
    print("accuracy classwise:\n", result)



def test_matrix_convergence_percentage():
    print("\n-- 'matrix_convergence_percentage' --")
    X0 = np.array([[2, 0, 0],
                   [2, 0, 2],
                   [0, 1, 0],
                   [0, 0, 3],
                   [0, 0, 3],
                   [1, 0, 2],
                   [0, 3, 3],
                   [0, 0, 0],
                   [9, 9, 9],
                   [100, 100, 100],])
    X1 = np.array([[1, 1, 2],
                   [2, 1, 2],
                   [3, 4, 0],
                   [1, 1, 2],
                   [2, 1, 1],
                   [1, 2, 2],
                   [1, 2, 3],
                   [0, 0, 0],
                   [8, 9, 9],
                   [100, 100, 101],])
    print("X0:\n", X0)
    print("X1:\n", X1)

    threshold = 0.5
    percentage = matrix_convergence_percentage(X0, X1, threshold)
    print("percentage converged (original):\n", percentage)

    X0z = row_normalize_matrix(X0, norm='zscores')
    X1z = row_normalize_matrix(X1, norm='zscores')
    percentage = matrix_convergence_percentage(X0z, X1z, threshold)
    print("percentage converged (after zscore):\n", percentage)



def test_replace_fraction_of_rows():
    print("\n-- 'replace_fraction_of_rows' --")
    In = np.array([[1, 0, 0],
                   [2, 0, 2],
                   [3, 3, 0],
                   [4, 0, 4],
                   [5, 0, 5],
                   [6, 0, 6],
                   [7, 0, 7],
                   [8, 8, 0],
                   [9, 9, 0],
                   [10, 10, 10],])
    f = 0.5
    Out, ind = replace_fraction_of_rows(In, f)
    print("In:\n", In)
    print("f =", f)
    print("Out:\n", Out)
    print("ind of remaining 1-f rows:\n", ind)

    f = 0.6
    Out, ind = replace_fraction_of_rows(In, f, ind_prior=ind)       # re-use a prior index list
    print("f =", f)
    print("Out:\n", Out)
    print("ind of remaining 1-f rows:\n", ind)



def test_replace_fraction_of_rows_stratified():
    print("\n-- 'replace_fraction_of_rows' --")
    In = np.array([[1, 0, 0],
                   [1, 0, 0],
                   [1, 0, 0],
                   [0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0],
                   [0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1],])
    f = 2/3
    # f = 1/2
    Out, ind = replace_fraction_of_rows(In, f, stratified=True)
    print("In:\n{}".format(In))
    print("f =", f)
    print("Out:\n{}".format(Out))
    print("ind of remaining 1-f rows:\n", ind)

    ind_prior = [0, 4, 7]
    Out, ind = replace_fraction_of_rows(In, f, stratified=True, ind_prior=ind_prior)
    print("In:\n{}".format(In))
    print("f =", f)
    print("ind_prior =", ind_prior)

    print("Out:\n{}".format(Out))
    print("ind of remaining 1-f rows:\n", ind)

    # f = 0.6
    # Out, ind = replace_fraction_of_rows(In, f, ind_prior=ind)       # re-use a prior index list
    # print("f =", f)
    # print("Out:\n", Out)
    # print("ind of remaining 1-f rows:\n", ind)



def test_construct_H_and_alpha():
    print("\n-- 'create_parameterized_H', 'create_parameterized_alpha' --")
    k = 2
    h = 9
    print("\nH (symmetric) h={}:\n{}".format(h,create_parameterized_H(k, h)))
    print("H (not symmetric) h={}:\n{}".format(h, create_parameterized_H(k, h, symmetric=False)))
    k = 3
    h = 8
    print("\nH (symmetric) h={}:\n{}".format(h, create_parameterized_H(k, h)))
    print("H (not symmetric) h={}:\n{}".format(h, create_parameterized_H(k, h, symmetric=False)))
    k = 4
    h = 7
    print("\nH (symmetric) h={}:\n{}".format(h, create_parameterized_H(k, h)))
    print("H (not symmetric) h={}:\n{}".format(h, create_parameterized_H(k, h, symmetric=False)))

    k = 2
    h = 3
    print("\nalpha h={}:\n{}".format(h, create_parameterized_alpha(k, h)))
    k = 3
    h = 1
    print("alpha h={}:\n{}".format(h, create_parameterized_alpha(k, h)))
    k = 3
    h = 3
    print("alpha h={}:\n{}".format(h, create_parameterized_alpha(k, h)))
    k = 4
    h = 4
    print("alpha h={}:\n{}".format(h, create_parameterized_alpha(k, h)))

    k = 2
    h = 3
    d = 10
    print("\ndegree h={}, d={}:\n{}".format(h, d, create_parameterized_alpha(k, h) * k * d))      # multiply with k to get average 1, then with d



def test_W_red_and_clamped():
    print("\n-- 'W_row', 'W_red', 'W_clamped' --")
    W = [[0, 1, 2, 0],
         [1, 0, 1, 0],
         [1, 1, 0, 3],
         [0, 0, 1, 0]]
    W = csr_matrix(W)
    print("W:\n", W.todense())
    print("W_row (row-normalized):\n", W_row(W).todense())
    print("W_red (symmetric and semi-normalized):\n", W_red(W).todense())

    Xl = [0]    # list of nodes with explicit beliefs [0, 2]
    print("\nRemove following rows from W:", Xl)
    Wclamped = W_clamped(W, Xl)
    print("W_clamped:\n", Wclamped.todense())

    print("\nCalculate undirected squared degree matrices:")
    print("D (for W):\n{}".format(degree_matrix(W, undirected=True, squared=True).todense()))
    print("D (for W_row):\n{}".format(degree_matrix(W_row(W), undirected=True, squared=True).todense()))
    print("D (for W_red):\n{}".format(degree_matrix(W_red(W), undirected=True, squared=True).todense()))
    print("D clamped (for W_clamped):\n{}".format(degree_matrix(Wclamped, undirected=True, squared=True).todense()))



def test_W_star():
    print("\n-- 'W_star' --")
    W = [[0, 1, 2, 0],
         [1, 0, 1, 0],
         [1, 1, 0, 3],
         [0, 0, 1, 0]]
    W = csr_matrix(W)
    print("W:\n{}".format(W.todense()))
    print("W_row (row-normalized):\n{}".format( W_row(W).todense()) )
    print("W_red (symmetric and semi-normalized):\n{}".format( W_red(W).todense()) )

    print("\nW_star (0, 0):\n{}".format( W_star(W, 0, 0).todense()) )
    print("W_star (1, 0):\n{}".format(W_star(W, 1, 0).todense()))
    print("W_star (0.5, 0.5):\n{}".format(W_star(W, 0.5, 0.5).todense()))

    Xl = [0]    # list of nodes with explicit beliefs [0, 2]
    print("\nRemove following rows from W:", Xl)
    Wclamped = W_clamped(W, Xl)
    print("W_clamped:\n{}".format( Wclamped.todense()) )

    print("\nW_star clamped (0.5, 0.5, 1):\n{}".format(W_star(W, 0.5, 0.5, 1, indices=Xl).todense()))
    print("W_star clamped (0.5, 0.5, 0):\n{}".format(W_star(W, 0.5, 0.5, 0, indices=Xl).todense()))
    print("W_star clamped (0.5, 0.5, 0.5):\n{}".format(W_star(W, 0.5, 0.5, 0.5, indices=Xl).todense()))

    # print("\nCalculate undirected squared degree matrices:")
    # print("D (for W):\n{}".format(degree_matrix(W, undirected=True, squared=True).todense()))
    # print("D (for W_row):\n{}".format(degree_matrix(W_row(W), undirected=True, squared=True).todense()))
    # print("D (for W_red):\n{}".format(degree_matrix(W_red(W), undirected=True, squared=True).todense()))
    # print("D clamped (for W_clamped):\n{}".format(degree_matrix(Wclamped, undirected=True, squared=True).todense()))



def test_transform_potentials():
    print("\n-- 'calculate_potential_from_row_normalized' --")
    H = np.array([[0.1, 0.8, 0.1],
                  [0.8, 0.1, 0.1],
                  [0.1, 0.1, 0.8]])
    alpha = [0.6, 0.2, 0.2]     # works with list or np.array
    print("H0:\n", H)
    print("alpha:\n", alpha)

    Pot = calculate_potential_from_row_normalized(H, alpha)
    print("Pot:\n", Pot)

    alpha0T = np.array([alpha]).transpose()
    print("\nalphaT:\n", alpha0T)
    print("H0 * alpha:\n", H * alpha)
    print("H0 * alphaT:\n", H * alpha0T)
    Pot2 = np.array(Pot)
    print("row sum (Pot):\n", Pot2.sum(1, keepdims=True))
    print("H(Pot):\n", row_normalize_matrix(Pot))
    print("H(H(Pot)):\n", row_normalize_matrix(row_normalize_matrix(Pot)))

    # -- Also consider outdegrees (in particular for undirected graph -> symmetric H)
    print("\n--Specify also outdegrees (so that potential becomes symmetric)")
    d_vec = [1, 3, 3]           # [1, 2, 3]
    d_vec = np.array(d_vec)     # check that np.array also works
    print("d_vec:", d_vec)
    Pot2 = calculate_potential_from_row_normalized(H, alpha, d_vec)
    print("Pot (symmetric):\n", Pot2)

    # # -- Variants of row-recentering
    # print("\n-- Calculate row-recentered residuals in 2 variants")
    # Pc1 = row_recentered_residual(Pot, paperVariant=True)
    # print("Row-recentered according to paper\n", Pc1)
    # Pc2 = row_recentered_residual(Pot, paperVariant=False)
    # print("Row-recentered according to new idea (!!! theory + experiments needed)\n", Pc2)      # TODO



def test_approx_spectral_radius():
    print("\n-- 'approx_spectral_radius' --")

    # --- Create the graph
    n = 1000
    alpha0 = [0.3334, 0.3333, 0.3333]
    h = 5
    P = np.array([[1, h, 1],
                  [h, 1, 1],
                  [1, 1, h]])
    m = 10000
    distribution = 'powerlaw'    # uniform powerlaw
    exponent = -0.3
    backEdgesAllowed = True
    sameInAsOutDegreeRanking = False
    debug = False
    start = time.time()
    W, Xd = planted_distribution_model(n, alpha=alpha0, P=P, m=m,
                                              distribution=distribution, exponent=exponent,
                                              backEdgesAllowed=backEdgesAllowed,
                                              sameInAsOutDegreeRanking=sameInAsOutDegreeRanking,
                                              debug=debug)
    print("n: {}".format(n))
    print("Time for graph generation: {}\n".format(time.time()-start))

    # --- Bigger graph with Kronecker
    M = kron(W.transpose(), P)

    # --- Time two variants
    start = time.time()
    rho1 = approx_spectral_radius(M,pyamg=True)
    print("Time for pyamg spectral radius: {}".format(time.time() - start))
    print("rho1: {}".format(rho1))

    start = time.time()
    rho2 = approx_spectral_radius(M, pyamg=False)
    print("Time for scipy spectral radius: {}".format(time.time() - start))
    print("rho2: {}".format(rho2))

    # --- 3 methods for spectral radisu, including non-sparse matrices
    H = create_parameterized_H(20, 2, symmetric=True)
    H = to_centering_beliefs(H)
    print("\nH:\n{}".format(H))

    start = time.time()
    rho1 = approx_spectral_radius(H,pyamg=True)
    print("Time for pyamg spectral radius: {}".format(time.time() - start))
    print("rho1: {}".format(rho1))

    start = time.time()
    rho2 = approx_spectral_radius(H, pyamg=False)
    print("Time for scipy spectral radius: {}".format(time.time() - start))
    print("rho2: {}".format(rho2))

    start = time.time()
    rho3 = approx_spectral_radius(H, pyamg=False, sparse=False)
    print("Time for non-sparse numpy spectral radius: {}".format(time.time() - start))
    print("rho3: {}".format(rho3))

    # --- For k=2, scipy made to default to non-sparse
    H = create_parameterized_H(2, 2, symmetric=True)
    print("\nH (k=2):\n{}".format(H))

    start = time.time()
    rho4 = approx_spectral_radius(H, pyamg=False)
    print("Time for scipy spectral radius: {}".format(time.time() - start))
    print("rho4: {}".format(rho4))



if __name__ == '__main__':
    test_transform_beliefs()
    test_dictionary_transform()
    test_to_explicit_vectors()
    test_max_binary_matrix()
    test_row_normalize_matrix()
    test_degree_matrix()
    test_matrix_difference_with_cosine_simililarity()
    test_matrix_difference_with_accuracy_etc()
    test_matrix_difference()
    test_matrix_difference_classwise()
    test_matrix_convergence_percentage()
    test_replace_fraction_of_rows()
    test_replace_fraction_of_rows_stratified()
    test_construct_H_and_alpha()
    test_W_red_and_clamped()
    test_W_star()
    test_transform_potentials()
    test_approx_spectral_radius()
