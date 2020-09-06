"""
Test class for 'sslh/inference'
"""


from inference import (linBP_symmetric,
                       beliefPropagation,
                       maximumMarginals)
from utils import (to_centering_beliefs,
                   to_explicit_list,
                   eps_convergence_linbp)
from fileInteraction import load_W
import numpy as np
import matplotlib.pyplot as plt
import os



# -- Determine path to data irrespective (!) of where the file is run from
from os.path import abspath, dirname, join
from inspect import getfile, currentframe
current_path = dirname(abspath(getfile(currentframe())))
data_directory = join(current_path, 'data/')
fig_directory = join(current_path, 'figs/')



def test_linBP_symmetric_Torus():
    # Shows that with s>1 LinBP will diverge and v.v., for Torus graph
    # Interesting is that with H (instead of Hc) and echo=True, just above s=1, the oscillations can start late
    print("\n-- 'linBP_symmetric', 'eps_convergence_linbp', with Torus --")

    # -- Load W, create X and P
    W, n = load_W(join(data_directory, 'Torus_W.csv'), zeroindexing=False)
    X = np.array(
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1],
     [0, 0, 0],
     [0, 0, 0],
     [0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]])
    H = np.array(
    [[0.1, 0.8, 0.1],
     [0.8, 0.1, 0.1],
     [0.1, 0.1, 0.8]])
    print("W:\n", W.todense())
    print("X:\n", X)
    Xc = to_centering_beliefs(X, ignoreZeroRows=True)
    print("Xc:\n", Xc)
    Xl = to_explicit_list(X)
    print("Xl:\n", Xl)
    print("H:\n", H)
    Hc = to_centering_beliefs(H)
    print("Hc:\n", Hc)

    # -- Other eps_max for 3 x 2 methods
    print("\neps_max without echo and Hc:")
    print(" eps_max (W):              ", eps_convergence_linbp(Hc, W))
    print("eps_max with echo and Hc:")
    print(" eps_max (W):              ", eps_convergence_linbp(Hc, W, echo=True))
    print("eps_max with echo and compensation and Hc:")
    print(" eps_max (W):              ", eps_convergence_linbp(Hc, W, echo=True, compensation=True))

    print("\neps_max without echo and H:")
    print(" eps_max (W):              ", eps_convergence_linbp(H, W))
    print("eps_max with echo and H:")
    print(" eps_max (W):              ", eps_convergence_linbp(H, W, echo=True))
    print("eps_max with echo and compensation and H:")
    print(" eps_max (W):              ", eps_convergence_linbp(H, W, echo=True, compensation=True))

    # -- Define parameters and run LinBP
    print("\nActual run with various parameters")
    s = 1.15                         # 0.4
    numMaxIt = 200
    echo = True
    convergencePercentage = None       # 0.5
    convergenceThreshold = 0.99
    eps_max = eps_convergence_linbp(H, W, echo=echo)

    print("eps:", s)
    print("echo:", echo)

    listF, actualNumIt, listConverged = linBP_symmetric(Xc, W, H * eps_max * s,
                                                             echo=echo,
                                                             numMaxIt=numMaxIt,
                                                             convergencePercentage=convergencePercentage, convergenceThreshold=convergenceThreshold,
                                                             debug=3)

    # # -- Display BP results
    print("\nlinBP results:")
    print("Notice that we get identical results with X or Xc, and for Hc or H (except for convergence)")
    print("\nlast two F:")
    print(listF[-2])
    print(listF[-1])
    print("actualNumIt:", actualNumIt)
    print("listConverged:\n", listConverged)

    # print("\nValues for node 6 (zero indexing):"
    # print listF[:, 6, :]
    # print("all:\n", listF

    # -- Visualize BP results
    filename = join(fig_directory, 'Fig_temp_SSLH_inference.pdf')
    print("\nVisualize values for node 3 (zero indexing):")
    node = 3
    plt.plot(listF[:, node, :], lw=2)
    plt.xlabel('# iterations')
    plt.ylabel('belief')
    plt.xlim(0, numMaxIt)

    print(filename)

    plt.savefig(filename, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype='letter', format='pdf',
                transparent=True, bbox_inches='tight', pad_inches=0.1)
    os.system("chmod 744 " + filename)  # first change permissions in order to open PDF
    os.system("open " + filename)       # open PDF




def test_beliefPropagation_Torus_directedBP():
    print("\n-- 'beliefPropagation', '_out_eps_convergence_directed_linbp'--")

    # -- Load W, create X and P
    W, n = load_W(fileName='data/Torus_directed_W.csv')
    X = np.array(
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1],
     [0, 0, 0],
     [0, 0, 0],
     [0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]])
    # !!! interesting with eps=2 and node 5, notice that non-symmetric potential (not suitable for VLDB)
    # P = np.array(
    # [[0.1, 0.8, 0.1],
    #  [0.1, 0.1, 0.8],
    #  [0.8, 0.1, 0.1]])
    # P = np.array(
    # [[0.1, 0.8, 0.0001],
    #  [0.9, 0.1, 0.0001],
    #  [0.0001, 0.0001, 1]])
    P = np.array(
    [[0.1, 0.8, 0.1],
     [0.8, 0.1, 0.1],
     [0.1, 0.1, 0.8]])
    print("W (dense):\n", W.todense())
    print("X:\n", X)
    print("P:\n", P)
    print()

    # --- Other parameters
    eps = 1
    numMaxIt = 100
    convergenceThreshold = 0.9
    convergencePercentage = 0.9
    damping = 1

    eps_max = eps_convergence_linbp(P, W, echo=False)  # !!! double-check
    print("eps_max: ", eps_max)
    print("eps: ", eps)
    P2 = P**eps             # entry-wise exponentiation (notice that syntax is different from matrix exponentiation)
    print("P.^eps:\n", P2)
    print()

    # --- BP
    # print "cos(5 deg):", np.cos((np.pi / 180) * 5)
    # print "cos(1 deg):", np.cos((np.pi / 180) * 1)
    listF, actualIt, listConv = beliefPropagation(X, W, P2,
                                                  numMaxIt=numMaxIt,
                                                  convergencePercentage=convergencePercentage,
                                                  convergenceThreshold=convergenceThreshold,
                                                  debug=3, damping=damping, clamping=False)

    # --- Display BP results
    print("BP results:")
    print("actualIt:", actualIt)
    print("last two convergence cond.:", listConv[-2:])
    print("last two F:")
    print(listF[-2])
    print(listF[-1])
    print("\nValues for node 6 (zero indexing):")
    print(listF[:, 6, :])

    # --- Visualize BP results
    filename = 'figs/temp2.pdf'
    print("\nVisualize values for node 3 (zero indexing):")
    node = 3
    plt.close()
    plt.plot(listF[:, node, :], lw=2)
    plt.xlabel('# iterations')
    plt.ylabel('belief')
    plt.yticks([0, 0.5, 1])
    # plt.xticks([0, actualIt])
    plt.xlim(0, actualIt)
    plt.savefig(filename, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype='letter', format='pdf',
                transparent=True, bbox_inches='tight', pad_inches=0.1)
    os.system("chmod 744 " + filename)  # first change permissions in order to open PDF
    os.system("open " + filename)       # open PDF

    # --- Maximum Marginal
    listF2 = maximumMarginals(X, W, P2, debug=1)
    print("\nMaximum Marginals:\n", listF2)




if __name__ == '__main__':
    test_linBP_symmetric_Torus()
    test_beliefPropagation_Torus_directedBP()

