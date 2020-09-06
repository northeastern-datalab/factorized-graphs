"""Test class for 'SSLH_files' (file manipulation for SSL-H)
Author: Wolfgang Gatterbauer
"""


import sys
sys.path.append('./../sslh')
import numpy as np
import pytest
from fileInteraction import (load_W, save_W,
                             load_X, save_X,
                             load_Xd, save_Xd,
                             load_H, save_H,
                             load_csv_records,
                             save_csv_record)


# -- Determine path to data *irrespective* of where the file is run from
from os.path import abspath, dirname, join
from inspect import getfile, currentframe
current_path = dirname(abspath(getfile(currentframe())))
data_directory = join(current_path, 'data/')


def test_load_and_save_W():
    print("\n-- 'load_W', 'save_W'")

    print("- pytest example where file does not exist:")
    filename1 = 'notExistingFile.csv'
    with pytest.raises(IOError):
        W, n = load_W(join(data_directory, filename1))     # files still without zero indexing

    print("\n- Torus, zeroindexing=False")
    filename2 = 'Torus_W.csv'
    print("File directory:\n  ", join(data_directory, filename2))
    W, n = load_W(join(data_directory, filename2), zeroindexing=False)
    print("W:\n{}".format(W))
    print("Dense:\n{}".format(W.todense()))
    print("Shape: {}".format(W.shape))

    print("\n- Torus, n=9")
    W, n = load_W(join(data_directory, filename2), zeroindexing=False, n=9)
    print("W:\n{}".format(W))
    print("Dense:\n{}".format(W.todense()))
    print("Shape: {}".format(W.shape))

    print("\n- Save same data with n=9 and load without specifying n=9 during load (notices max int 8th node = 7)")
    filename3 = 'Torus_W3.csv'
    save_W(join(data_directory, filename3), W, saveWeights=True)
    W, n = load_W(join(data_directory, filename3))                    # newly written files use zeroindexing=True
    print("W3:\n{}".format(W))
    print("Dense:\n{}".format(W.todense()))
    print("Shape: {}".format(W.shape))

    print("\n- 2 columns, directed")
    filename5 = 'W2.txt'
    W, n = load_W(join(data_directory, filename5), zeroindexing=False)
    print("W2\n{}".format(W))

    print("\n- 2 columns, directed, doubleUndirected=True")
    W, n = load_W(join(data_directory, filename5), zeroindexing=False, doubleUndirected=True)
    print("W\n{}".format(W))

    print("\n- 3 columns, directed")
    W, n = load_W(join(data_directory, 'Torus_W6.csv'))
    print("W6\n{}".format(W))

    print("\n- 3 columns, directed, doubleUndirected")
    W, n = load_W(join(data_directory, 'Torus_W6.csv'), doubleUndirected=True)
    print("W6\n{}".format(W))

    print("\n- Save the same without weights, and load again")
    filename7 = 'Torus_W7.csv'
    save_W(join(data_directory, filename7), W)
    W, n = load_W(join(data_directory, filename7))
    print("W7\n{}".format(W))

    print("\n- Load with large entries, save and load again")
    filename8 = 'W8.csv'
    W, n = load_W(join(data_directory, filename8))
    print("W8\n{}".format(W))
    filename9 = 'W9.csv'
    save_W(join(data_directory, filename9), W)
    W, n = load_W(join(data_directory, filename9))
    print("W9\n{}".format(W))


def test_load_and_save_X():
    print("\n-- 'load_X', 'save_X'")

    print("- Load matrix and specify n=8")
    X, _, _ = load_X(join(data_directory, 'Torus_X.csv'), n=8, zeroindexing=False)
    print("X:\n{}".format(X))
    print("Shape: {}".format(X.shape))

    print("\n- Save and load matrix with n=8 k=3")
    filename2 = 'Torus_X2.csv'
    save_X(join(data_directory, filename2), X)
    X, _, _ = load_X(join(data_directory, filename2), n=8, k=3)
    print("X2:\n{}".format(X))

    print("\n- Input contains float: load, save and load")
    filename3 = 'Torus_X3.csv'
    X, _, _ = load_X(join(data_directory, filename3), n=8, k=3)
    print("X3:\n{}".format(X))
    filename4 = 'Torus_X4.csv'
    save_X(join(data_directory, filename4), X)
    X, _, _ = load_X(join(data_directory, filename4), n=8, k=3)
    print("X4:\n{}".format(X))

    print("\n- Input contains only 1 explicit belief (3 columns)")
    X, _, _ = load_X(join(data_directory, 'Torus_X5.csv'), n=8, k=3)
    print("X5:\n{}".format(X))

    print("\n- Input contains only two columns. Correct")
    X, _, _ = load_X(join(data_directory, 'Torus_X6.csv'), n=8, k=3)
    print("X6:\n{}".format(X))

    print("\n- Input contains only two columns. With assert error (node with several classes)")
    try:
        X, _, _ = load_X(join(data_directory, 'Torus_X_twoColumns_assertError.csv'), n=8, k=3)
    except AssertionError as e:
        print("! Assertion error:\n", e)

    print("\n- Input contains a node number > 1Mio")
    filename7 = 'Torus_X7.csv'
    X, _, _ = load_X(join(data_directory, filename7))
    print("X7:\n{}".format(X))
    filename8 = 'Torus_X8.csv'
    save_X(join(data_directory, filename8), X)
    X, _, _ = load_X(join(data_directory, filename8))
    print("X8:\n{}".format(X))


def test_load_and_save_Xd():

    # TODO: could be made faster with Pandas: make test case

    print("\n-- 'save_Xd', 'load_Xd")
    filename = 'test_load_and_save_Xd.csv'
    Xd = {0: 4, 1: 5, 8: -1, 0: 3, 1234567: 1}
    print("Original Xd:\n", Xd)
    save_Xd(join(data_directory, filename), Xd)

    Xd = load_Xd(join(data_directory, filename))
    print("Loaded Xd:\n", Xd)


def test_load_and_save_H():
    print("\n-- 'load_H', 'save_H'")

    print("- Loading H")
    H, _, _ = load_H(join(data_directory, 'Torus_H.csv'), zeroindexing=False)
    print("Loaded H: \n{}".format(H))
    print("Shape:{}".format(H.shape))

    filename2 = 'Torus_H2.csv'
    print("\n- Saving and loading does not lead to different precision anymore (because of change to Pandas)")
    save_H(join(data_directory, filename2), H)
    H, _, _ = load_H(join(data_directory, filename2))
    print("Saved and loaded H2: \n{}".format(H))
    print("Shape: {}".format(H.shape))
    save_H(join(data_directory, 'Torus_H3.csv'), H)

    print("\n- Loading keeps integers (before were transformed to float)")
    H = np.array([[1, 2, 3],
                  [4, 5, 6]])        # change 6 to 6.5
    print("Original H:\n{}".format(H))
    print("Shape: {}".format(H.shape))
    filename4 = 'H4.csv'
    save_H(join(data_directory, filename4), H)
    H, _, _ = load_H(join(data_directory, filename4))
    print("Loaded H4:\n{}".format(H))
    print("Shape: {}".format(H.shape))


def test_save_csv_record():
    print("\n-- 'save_csv_record'")
    print("Can recover int and float from strings")
    record1 = ['a', 1, 1.1]
    record2 = ('b', 2, 2.2)
    record3 = np.array(('c', 3, 3.3))
    filename = 'test_save_csv_record.csv'
    dirfilename = join(data_directory, filename)
    save_csv_record(dirfilename, record1, append=False)
    save_csv_record(dirfilename, record2, append=True)
    save_csv_record(dirfilename, record2, append=True)
    records3 = load_csv_records(dirfilename)
    print("Loaded record from csv:\n", records3)



if __name__ == '__main__':
    test_load_and_save_W()
    test_load_and_save_X()
    test_load_and_save_Xd()
    test_load_and_save_H()
    test_save_csv_record()
