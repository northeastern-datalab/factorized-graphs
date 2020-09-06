"""
TODO: some figure numberings (CHOICE, VERSION) were changed: make sure the current numberings are consistent with original runs
TODO: replaces previous versions 161110, 171029
TODO: how to get the grid small log lines also for x-axis?
TODO: mention that Python 3.5.2 or later is required (ideally 3.8)

Plots times for graph creation, eps_max calculation, compatibility estimation and propagation
Since graph creation takes most time, especially for large graphs, saves graphs to a file format, then loads them later again.

CHOICE is a choice of parameters and is thus included in CSV file name
VARIANT is a variant that is chosen to be plotted, is included only in Figure file name

Important (CHOICE, VARIANT) combinations:
    (3,3): paper figure introduction (prop, Holdout, DCEr) with arrows
    (3,2): paper figure main experiments (all methods) with arrows
    (3,4): paper figure variant (prop)
    (3,5): paper figure variant (prop, Holdout)
    (3,6): paper figure variant (prop, Holdout, DCEr)

First version: Nov 10, 2016
This version: Jan 26, 2020
"""

import numpy as np
import datetime
import random
# import os                                   # for displaying created PDF TODO: can be removed?
import time
import sys

sys.path.append("../sslh")  # important to be able to run from command line
from fileInteraction import (save_csv_record,
                      save_W,
                      save_X,
                      load_W,
                      load_X)  # TODO: Paul, why do we need to use sslh here as part of the name but below not for estimation?
from utils import (from_dictionary_beliefs,
                   create_parameterized_H,
                   replace_fraction_of_rows,
                   to_centering_beliefs,
                   eps_convergence_linbp_parameterized,
                   showfig)
from estimation import (estimateH,
                        estimateH_baseline_serial)
from graphGenerator import planted_distribution_model
from inference import linBP_symmetric_parameterized
import matplotlib as mpl
from matplotlib.ticker import LogLocator

mpl.use('Agg')  # more common rendering
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', None)  # show all columns
pd.options.mode.chained_assignment = None  # default='warn'

# -- Determine path to data *irrespective* of where the file is run from
from os.path import abspath, dirname, join
from inspect import getfile, currentframe

current_path = dirname(abspath(getfile(currentframe())))
figure_directory = join(current_path, 'figs')
data_directory = join(current_path, 'datacache')


def run(choice, variant, create_data=False, add_data=False, create_graph=False,
        create_fig=True, show_plot=False, create_pdf=False, show_pdf=False, shorten_length=False, show_arrows=True):
    """main parameterized method to produce all figures.
    Can be run from external jupyther notebook or method to produce all figures in PDF
    """

    # -- Setup
    CHOICE = choice  # determines the CSV data file to use
    VARIANT = variant  # determines the variant of how the figures are plotted
    CREATE_DATA = create_data  # starts new CSV file and stores experimental timing results
    ADD_DATA = add_data  # adds data to existing file
    CREATE_GRAPH = create_graph  # creates the actual graph for experiments (stores W and X in CSV files)

    SHOW_PDF = show_pdf
    SHOW_PLOT = show_plot
    CREATE_FIG = create_fig
    CREATE_PDF = create_pdf
    SHORTEN_LENGTH = shorten_length  # to prune certain fraction of data to plot
    SHOW_SCALING_LABELS = True  # first entry in the legend is for the dashed line of scalability
    SHOW_TITLE = True  # show parameters in title of plot
    SHOW_DCER_WITH_BOX = True  # show DCER value in a extra box
    LABEL_FONTSIZE = 16  # size of number labels in figure
    SHOW_LINEAR = True  # show dashed line for linear scaling

    SHOW_ARROWS = show_arrows  # show extra visual comparison of speed-up

    csv_filename = 'Fig_Timing_{}.csv'.format(CHOICE)  # CSV filename includes CHOICE
    filename = 'Fig_Timing_{}-{}'.format(CHOICE, VARIANT)  # PDF filename includes CHOICE and VARIANT
    header = ['n', 'type', 'time']
    if CREATE_DATA:
        save_csv_record(join(data_directory, csv_filename), header, append=False)

    # -- Default Graph parameters
    distribution = 'powerlaw'
    exponent = -0.3
    k = 3
    a = 1  # this value was erroneously set to 5 previously!!! TODO: fix everywhere else
    # err = 0
    avoidNeighbors = False
    f = 0.1
    est_EC = True  # !!! TODO: for graph estimation
    weights = 10
    pyamg = False
    convergencePercentage_W = None
    alpha = 0
    beta = 0
    gamma = 0
    s = 0.5
    numMaxIt = 10
    xtick_lab = [0.001, 0.01, 0.1, 1]
    ytick_lab = np.arange(0, 1, 0.1)
    xmin = 1e2
    xmax = 1e8
    # xmax = 1e6
    ymin = 1e-3
    ymax = 5e3
    color_vec = ["#4C72B0", "#55A868", "#8172B2", "#C44E52", "#CCB974", 'black', 'black', "#64B5CD", "black"]
    marker_vec = ['s', '^', 'x', 'o', 'None', 'None', 'None', 'None']
    linestyle_vec = ['solid'] * 6 + ['dashed']
    linewidth_vec = [3] * 3 + [4, 3, 4] + [3] * 7
    SHOWMAXNUMBER = True
    show_num_vec = ['MHE', 'LHE', 'DHE', 'DHEr', 'Holdout', 'prop', 'eps_max']

    # %% -- Main Options
    if CHOICE == 3:
        n_vec = [100, 200, 400, 800,
                 1600, 3200, 6400,
                 12800, 25600, 51200,
                 102400, 204800, 409600, 819200,
                 1638400, 3276800, 6553600
                 ]
        # # n_vec = [1638400]  # graph:  12021 sec = 3.4h, 18600 sec = 5h, 21824 sec (34000 sec old laptop)
        # # n_vec = [3276800]  # graph:  49481 sec = 13.8h, 68145 sec (125233 sec old laptop)
        # # n_vec = [6553600]  # graph: 145020 sec = 40h
        h = 8
        d = 5

        repeat_vec_vec = [[
            50, 50, 50, 50,
            50, 50, 50,
            20, 10, 10,
            5, 5, 5, 3,
            3, 3, 3
        ],
            [
                5, 5, 5, 5,
                3, 3, 3,
                3, 3, 1,
                1
            ],
            [
                20, 20, 20, 10,
                10, 10, 10,
                10, 5, 5,
                5, 3, 3, 1,
                1, 1, 1
            ]
        ]
        method_vec_vec = [['MHE', 'DHE', 'DHEr', 'LHE'],
                          ['Holdout'],
                          ['prop']
                          ]

        if VARIANT == 1:
            method_vec_fig = ['MHE', 'LHE', 'DHE', 'DHEr', 'Holdout', 'prop']
            label_vec = ['MCE', 'LCE', 'DCE', 'DCEr', 'Holdout', 'prop']
            show_num_vec = ['MHE', 'LHE', 'DHE', 'DHEr', 'Holdout', 'prop']

        if VARIANT == 2:  # version used for main paper figure
            method_vec_fig = ['MHE', 'LHE', 'DHE', 'DHEr', 'Holdout', 'prop']
            label_vec = ['MCE', 'LCE', 'DCE', 'DCEr', 'Holdout', 'prop']
            linestyle_vec = ['solid'] * 5 + ['dashed']
            SHOW_ARROWS = False

        if VARIANT == 3:  # version used for main paper figure
            method_vec_fig = ['DHEr', 'Holdout', 'prop']
            label_vec = ['DCEr', 'Holdout', 'Propagation', '$\epsilon_{\mathrm{max}}$']
            linestyle_vec = ['solid'] * 2 + ['dashed']
            color_vec = ["#C44E52", "#CCB974", 'black', 'black', "#64B5CD", "black"]
            marker_vec = ['o', 'x', 'None', 'None', 'None']
            linestyle_vec = ['solid'] * 3 + ['dashed']
            linewidth_vec = [4, 3, 4] + [3] * 7
            ymin = 1e-2
            SHOW_ARROWS = True

        if VARIANT == 4:  # figure used in slides
            method_vec_fig = ['prop']
            label_vec = ['Propagation']
            color_vec = ['black']
            marker_vec = ['None']
            linestyle_vec = ['solid'] * 1
            linewidth_vec = [2]
            ymin = 1e-2
            SHOW_ARROWS = False
            SHOW_SCALING_LABELS = False
            SHOW_TITLE = False
            SHOW_DCER_WITH_BOX = False
            LABEL_FONTSIZE = 20
            SHOW_LINEAR = False

        if VARIANT == 5:  # figure used in slides
            method_vec_fig = ['prop', 'Holdout']
            label_vec = ['Propagation', 'Baseline']
            color_vec = ['black', "#CCB974"]
            marker_vec = ['None', '^']
            linestyle_vec = ['solid'] * 2
            linewidth_vec = [2, 4]
            ymin = 1e-2
            SHOW_ARROWS = True
            SHOW_SCALING_LABELS = False
            SHOW_TITLE = False
            SHOW_DCER_WITH_BOX = False
            LABEL_FONTSIZE = 20
            SHOW_LINEAR = False

        if VARIANT == 6:  # figure used in slides
            method_vec_fig = ['prop', 'Holdout', 'DHEr']
            label_vec = ['Propagation', 'Baseline', 'Our method']
            color_vec = ['black', "#CCB974", "#C44E52"]
            marker_vec = ['None', '^', 'o', 'None', 'None']
            linestyle_vec = ['solid'] + ['solid'] * 2
            linewidth_vec = [2, 4, 4]
            ymin = 1e-2
            SHOW_ARROWS = True
            SHOW_SCALING_LABELS = False
            SHOW_TITLE = True
            SHOW_DCER_WITH_BOX = False
            LABEL_FONTSIZE = 20
            SHOW_LINEAR = False

        graph_cvs = 'Fig_Timing_SSLH_1'  # re-use existing large graphs


    elif CHOICE == 4:
        n_vec = [200, 400, 800,
                 1600, 3200, 6400,
                 12800, 25600, 51200,
                 102400, 204800, 409600, 819200,
                 ]
        # n_vec = [819200]    # graph: 47905 sec = 13.3h. 90562 sec = 25h (180527 sec old laptop)
        h = 3
        d = 25
        repeat_vec_vec = [[
            50, 50, 50,
            50, 50, 50,
            20, 10, 10,
            5, 3, 3, 3,
        ],
            [
                5, 5, 5,
                3, 1, 1,
                1, 1, 1
            ],
            [
                20, 20, 10,
                10, 10, 10,
                10, 5, 5,
                5, 1, 1, 1,
            ]
        ]
        method_vec_vec = [['MHE', 'DHE', 'DHEr', 'LHE'],
                          ['Holdout'],
                          ['prop']
                          ]

        VARIANT = 2

        if VARIANT == 1:
            method_vec_fig = ['MHE', 'LHE', 'DHE', 'DHEr', 'Holdout', 'prop', 'eps_max']
            label_vec = ['MCE', 'LCE', 'DCE', 'DCEr', 'Holdout', 'prop', '$\epsilon_{\mathrm{max}}$']
            show_num_vec = ['MHE', 'LHE', 'DHE', 'DHEr', 'Holdout', 'prop', 'eps_max']

        if VARIANT == 2:
            method_vec_fig = ['MHE', 'LHE', 'DHE', 'DHEr', 'Holdout', 'prop']
            label_vec = ['MCE', 'LCE', 'DCE', 'DCEr', 'Holdout', 'prop']
            linestyle_vec = ['solid'] * 5 + ['dashed']

        if VARIANT == 3:
            method_vec_fig = ['DHEr', 'Holdout', 'prop']

            label_vec = ['DCEr', 'Holdout', 'Propagation', '$\epsilon_{\mathrm{max}}$']
            linestyle_vec = ['solid'] * 2 + ['dashed']
            color_vec = ["#C44E52", "#CCB974", 'black', 'black', "#64B5CD", "black"]
            marker_vec = ['o', 'x', 'None', 'None', 'None']
            linestyle_vec = ['solid'] * 3 + ['dashed']
            linewidth_vec = [4, 3, 4] + [3] * 7
            ymin = 1e-2

        graph_cvs = 'Fig_Timing_SSLH_2'  # re-use existing large graphs
        xmin = 1e3
        xmax = 5e7
        ymax = 1e3


    elif CHOICE == 2:
        # rep_Estimation = 10
        # n_vec = [200, 400, 800, 1600, 3200, 6400, 12800,
        #          25600, 51200, 102400, 204800, 409600, 819200]
        # repeat_vec = [20, 20, 20, 20, 20, 10, 10,
        #               10, 10, 10, 5, 5, 1]
        # n_vec = [819200]    # graph: 47905 sec = 13.3h. 90562 sec = 25h (180527 sec old laptop)
        n_vec = [1638400]  # !!! not done yet
        repeat_vec = [1]
        h = 3
        d = 25
        xmax = 5e7
        graph_cvs = 'Fig_Timing_SSLH_2'

    elif CHOICE == 10:  # same as 3 but with difference bars
        n_vec = [100, 200, 400, 800,
                 1600, 3200, 6400,
                 12800, 25600, 51200,
                 102400, 204800, 409600, 819200,
                 1638400, 3276800, 6553600
                 ]
        # # n_vec = [1638400]  # graph:  12021 sec = 3.4h, 18600 sec = 5h, 21824 sec (34000 sec old laptop)
        # # n_vec = [3276800]  # graph:  49481 sec = 13.8h, 68145 sec (125233 sec old laptop)
        # # n_vec = [6553600]  # graph: 145020 sec = 40h
        h = 8
        d = 5

        repeat_vec_vec = [[
            50, 50, 50, 50,
            50, 50, 50,
            20, 10, 10,
            5, 5, 5, 3,
            3, 3, 3
        ],
            [
                5, 5, 5, 5,
                3, 3, 3,
                3, 3, 1,
                1
            ],
            [
                20, 20, 20, 10,
                10, 10, 10,
                10, 5, 5,
                5, 3, 3, 1,
                1, 1, 1
            ]
        ]
        method_vec_vec = [['MHE', 'DHE', 'DHEr', 'LHE'],
                          ['Holdout'],
                          ['prop']
                          ]

        method_vec_fig = ['DHEr', 'Holdout', 'prop']
        label_vec = ['DCEr', 'Holdout', 'Propagation', '$\epsilon_{\mathrm{max}}$']
        linestyle_vec = ['solid'] * 2 + ['dashed']
        color_vec = ["#C44E52", "#CCB974", 'black', 'black', "#64B5CD", "black"]
        marker_vec = ['o', 'x', 'None', 'None', 'None']
        linestyle_vec = ['solid'] * 3 + ['dashed']
        linewidth_vec = [4, 3, 4] + [3] * 7
        ymin = 1e-2

        graph_cvs = 'Fig_Timing_SSLH_1'  # re-use existing large graphs

    else:
        raise Warning("Incorrect choice!")

    # %% -- Common options

    alpha0 = np.array([a, 1., 1.])
    alpha0 = alpha0 / np.sum(alpha0)
    H0 = create_parameterized_H(k, h, symmetric=True)
    H0c = to_centering_beliefs(H0)
    RANDOMSEED = None  # For repeatability
    random.seed(RANDOMSEED)  # seeds some other python random generator
    np.random.seed(seed=RANDOMSEED)  # seeds the actually used numpy random generator; both are used and thus needed

    # print("CHOICE: {}".format(CHOICE))

    def save_tuple(n, label, time):
        tuple = [str(datetime.datetime.now())]
        text = [n, label, time]
        tuple.extend(text)
        print("time potential {}: {}".format(label, time))
        save_csv_record(join(data_directory, csv_filename), tuple)

    # %% -- Create data
    if CREATE_DATA or ADD_DATA:

        for repeat_vec, method_vec in zip(repeat_vec_vec, method_vec_vec):

            for n, repeat in zip(n_vec, repeat_vec):
                print("\nn: {}".format(n))
                # repeat = repeat_vec[j]

                # -- Graph
                if CREATE_GRAPH:
                    start = time.time()
                    W, Xd = planted_distribution_model(n, alpha=alpha0, P=H0, m=d * n,
                                                       distribution=distribution,
                                                       exponent=exponent,
                                                       directed=False,
                                                       debug=False)
                    X0 = from_dictionary_beliefs(Xd)
                    time_graph = time.time() - start

                    save_W(join(data_directory, '{}_{}_W.csv'.format(graph_cvs, n)), W, saveWeights=False)
                    save_X(join(data_directory, '{}_{}_X.csv'.format(graph_cvs, n)), X0)
                    save_tuple(n, 'graph', time_graph)

                else:
                    W, _ = load_W(join(data_directory, '{}_{}_W.csv'.format(graph_cvs, n)), skiprows=1, zeroindexing=True, n=None,
                                  doubleUndirected=False)
                    X0, _, _ = load_X(join(data_directory, '{}_{}_X.csv'.format(graph_cvs, n)), n=None, k=None, skiprows=1, zeroindexing=True)

                # -- Repeat loop
                for i in range(repeat):
                    print("\n  repeat: {}".format(i))
                    X2, ind = replace_fraction_of_rows(X0, 1 - f, avoidNeighbors=avoidNeighbors, W=W)

                    for method in method_vec:

                        if method == 'DHE':
                            start = time.time()
                            H2 = estimateH(X2, W, method='DHE', variant=1, distance=5, EC=est_EC, weights=weights)
                            time_est = time.time() - start
                            save_tuple(n, 'DHE', time_est)

                        elif method == 'DHEr':
                            start = time.time()
                            H2 = estimateH(X2, W, method='DHE', variant=1, distance=5, EC=est_EC, weights=weights, randomize=True)
                            time_est = time.time() - start
                            save_tuple(n, 'DHEr', time_est)

                        elif method == 'MHE':
                            start = time.time()
                            H2 = estimateH(X2, W, method='MHE', variant=1, distance=1, EC=est_EC, weights=None)
                            time_est = time.time() - start
                            save_tuple(n, 'MHE', time_est)

                        elif method == 'LHE':
                            start = time.time()
                            H2 = estimateH(X2, W, method='LHE', variant=1, distance=1, EC=est_EC, weights=None)
                            time_est = time.time() - start
                            save_tuple(n, 'LHE', time_est)

                        elif method == 'Holdout':
                            start = time.time()
                            H2 = estimateH_baseline_serial(X2, ind, W, numMax=numMaxIt,
                                                           numberOfSplits=1,
                                                           # EC=EC,
                                                           # weights=weight,
                                                           alpha=alpha, beta=beta, gamma=gamma)
                            time_est = time.time() - start
                            save_tuple(n, 'Holdout', time_est)

                        elif method == 'prop':
                            H2c = to_centering_beliefs(H0)
                            X2c = to_centering_beliefs(X2, ignoreZeroRows=True)  # try without
                            start = time.time()
                            eps_max = eps_convergence_linbp_parameterized(H2c, W, method='noecho', alpha=alpha, beta=beta, gamma=gamma, X=X2,
                                                                          pyamg=pyamg)
                            time_eps_max = time.time() - start
                            save_tuple(n, 'eps_max', time_eps_max)

                            # -- Propagate
                            eps = s * eps_max
                            try:
                                start = time.time()
                                F, actualIt, actualPercentageConverged = \
                                    linBP_symmetric_parameterized(X2, W, H2c * eps,
                                                                  method='noecho',
                                                                  alpha=alpha, beta=beta, gamma=gamma,
                                                                  numMaxIt=numMaxIt,
                                                                  convergencePercentage=convergencePercentage_W,
                                                                  debug=2)
                                time_prop = time.time() - start
                            except ValueError as e:
                                print(
                                    "ERROR: {}: d={}, h={}".format(e, d, h))
                            else:
                                save_tuple(n, 'prop', time_prop)

                        else:
                            raise Warning("Incorrect choice!")

    # %% -- Read, aggregate, and pivot data for all options
    df1 = pd.read_csv(join(data_directory, csv_filename))
    # print("\n-- df1: (length {}):\n{}".format(len(df1.index), df1.head(50)))

    # Aggregate repetitions
    df2 = df1.groupby(['n', 'type']).agg \
        ({'time': [np.mean, np.median, np.std, np.size],  # Multiple Aggregates
          })
    df2.columns = ['_'.join(col).strip() for col in df2.columns.values]  # flatten the column hierarchy
    df2.reset_index(inplace=True)  # remove the index hierarchy
    df2.rename(columns={'time_size': 'count'}, inplace=True)
    # print("\n-- df2 (length {}):\n{}".format(len(df2.index), df2.head(15)))

    # Pivot table
    df3 = pd.pivot_table(df2, index=['n'], columns=['type'], values=['time_mean', 'time_median'])  # Pivot
    # df3 = pd.pivot_table(df2, index=['n'], columns=['type'], values=['time_mean', 'time_median', 'time_std'] )  # Pivot
    # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(30)))
    df3.columns = ['_'.join(col).strip() for col in df3.columns.values]  # flatten the column hierarchy
    df3.reset_index(inplace=True)  # remove the index hierarchy
    # df2.rename(columns={'time_size': 'count'}, inplace=True)
    # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(30)))

    # Extract values
    X = df3['n'].values  # plot x values
    X = X * d / 2  # calculate edges (!!! notice dividing by 2 as one edge appears twice in symmetric adjacency matrix)
    Y = {}
    for method in method_vec_fig:
        # Y[method] = df3['time_mean_{}'.format(method)].values
        Y[method] = df3['time_median_{}'.format(method)].values

    if SHORTEN_LENGTH:
        SHORT_FACTOR = 4  ## KEEP EVERY Nth ELEMENT
        X = np.copy(X[list(range(0, len(X), SHORT_FACTOR)),])
        for method in method_vec_fig:
            Y[method] = np.copy(Y[method][list(range(0, len(Y[method]), SHORT_FACTOR)),])

    # %% -- Figure
    if CREATE_FIG:
        fig_filename = '{}.pdf'.format(filename)  # TODO: repeat pattern in other files
        mpl.rcParams['backend'] = 'agg'
        mpl.rcParams['lines.linewidth'] = 3
        mpl.rcParams['font.size'] = LABEL_FONTSIZE
        mpl.rcParams['axes.labelsize'] = 20
        mpl.rcParams['axes.titlesize'] = 16
        mpl.rcParams['xtick.labelsize'] = 16
        mpl.rcParams['ytick.labelsize'] = 16
        mpl.rcParams['legend.fontsize'] = 12
        mpl.rcParams['axes.edgecolor'] = '111111'  # axes edge color
        mpl.rcParams['grid.color'] = '777777'  # grid color
        mpl.rcParams['figure.figsize'] = [4, 4]
        mpl.rcParams['xtick.major.pad'] = 4  # padding of tick labels: default = 4
        mpl.rcParams['ytick.major.pad'] = 4  # padding of tick labels: default = 4
        fig = plt.figure()
        ax = fig.add_axes([0.13, 0.17, 0.8, 0.8])

        # -- Draw the plots
        if SHOW_LINEAR:
            ax.plot([1, 1e8], [1e-5, 1e3], linewidth=1, color='gray', linestyle='dashed', label='1sec/100k edges', clip_on=True, zorder=3)
        for i, (method, color, marker, linewidth, linestyle) in enumerate(zip(method_vec_fig, color_vec, marker_vec, linewidth_vec, linestyle_vec)):
            ax.plot(X, Y[method], linewidth=linewidth, color=color, linestyle=linestyle, label=label_vec[i], clip_on=True, marker=marker,
                    markersize=6, markeredgewidth=1, markeredgecolor='black', zorder=4)

            # for choice, (option, label, color, linewidth, clip_on, linestyle, marker, markersize) in \
            #         enumerate(zip(option_vec, labels, facecolor_vec, linewidth_vec, clip_on_vec, linestyle_vec, marker_vec, markersize_vec)):
            #     P = ax.plot(X_f, Y[choice], linewidth=linewidth, color=color, linestyle=linestyle, label=label, zorder=4, marker=marker,
            #                 markersize=markersize, markeredgewidth=1, markeredgecolor='black', clip_on=clip_on)

            if SHOWMAXNUMBER and method in show_num_vec:
                if method == 'DHEr' and SHOW_DCER_WITH_BOX:
                    j = np.argmax(np.ma.masked_invalid(Y[method]))  # mask nan, then get index of max element
                    ax.annotate(int(np.round(Y[method][j])), xy=(X[j] * 1.5, Y[method][j]), color=color, va='center',
                                bbox=dict(boxstyle="round,pad=0.3", fc="w"), annotation_clip=False, zorder=5)
                else:
                    j = np.argmax(np.ma.masked_invalid(Y[method]))  # mask nan, then get index of max element
                    ax.annotate(int(np.round(Y[method][j])), xy=(X[j] * 1.5, Y[method][j]), color=color, va='center', annotation_clip=False, zorder=5)

        if SHOW_ARROWS:
            dce_opt = 'DHEr'
            holdout_opt = 'Holdout'
            prop_opt = 'prop'

            j_holdout = np.argmax(np.ma.masked_invalid(Y[holdout_opt]))

            if dce_opt in Y:
                j_dce = np.argmax(np.ma.masked_invalid(Y[dce_opt]))
                ax.annotate(s='', xy=(X[j_dce], Y[prop_opt][j_dce]),
                            xytext=(X[j_dce], Y[dce_opt][j_dce]),
                            arrowprops=dict(arrowstyle='<->'))
                ax.annotate(str(int(np.round(Y[prop_opt][j_dce] / Y[dce_opt][j_dce]))) + 'x',
                            xy=(X[j_dce], int(Y[prop_opt][j_dce] + Y[dce_opt][j_dce]) / 6),
                            color='black', va='center', fontsize=14,
                            # bbox = dict(boxstyle="round,pad=0.3", fc="w"),
                            annotation_clip=False, zorder=5)

                ax.annotate(s='', xy=(X[j_holdout], Y[holdout_opt][j_holdout]),
                            xytext=(X[j_holdout], Y[dce_opt][j_holdout]),
                            arrowprops=dict(arrowstyle='<->'))
                ax.annotate(str(int(np.round(Y[holdout_opt][j_holdout] / Y[dce_opt][j_holdout]))) + 'x',
                            xy=(X[j_holdout], int(Y[holdout_opt][j_holdout] + Y[dce_opt][j_holdout]) / 8),
                            color='black', va='center', fontsize=14,
                            # bbox = dict(boxstyle="round,pad=0.3", fc="w"),
                            annotation_clip=False, zorder=5)

            else:  # in case dce_opt not shown, then show arrow as compared to prop method
                ax.annotate(s='', xy=(X[j_holdout], Y[holdout_opt][j_holdout]),
                            xytext=(X[j_holdout], Y[prop_opt][j_holdout]),
                            arrowprops=dict(arrowstyle='<->'))
                ax.annotate(str(int(np.round(Y[holdout_opt][j_holdout] / Y[prop_opt][j_holdout]))) + 'x',
                            xy=(X[j_holdout], int(Y[holdout_opt][j_holdout] + Y[prop_opt][j_holdout]) / 8),
                            color='black', va='center', fontsize=14,
                            # bbox = dict(boxstyle="round,pad=0.3", fc="w"),
                            annotation_clip=False, zorder=5)

        if SHOW_TITLE:
            plt.title(r'$\!\!\!d\!=\!{}, h\!=\!{}$'.format(d, h))

        handles, labels = ax.get_legend_handles_labels()
        if not SHOW_SCALING_LABELS and SHOW_LINEAR:
            handles = handles[1:]
            labels = labels[1:]

        legend = plt.legend(handles, labels,
                            loc='upper left',  # 'upper right'
                            handlelength=2,
                            labelspacing=0,  # distance between label entries
                            handletextpad=0.3,  # distance between label and the line representation
                            borderaxespad=0.2,  # distance between legend and the outer axes
                            borderpad=0.3,  # padding inside legend box
                            numpoints=1,  # put the marker only once
                            )
        legend.set_zorder(3)
        frame = legend.get_frame()
        frame.set_linewidth(0.0)
        frame.set_alpha(0.2)  # 0.8

        # -- Figure settings and save
        plt.minorticks_on()
        plt.xscale('log')
        plt.yscale('log')
        minorLocator = LogLocator(base=10, subs=[0.1 * n for n in range(1, 10)], numticks=40)   # TODO: discuss with Paul trick that helped with grid lines last time; necessary in order to create the log locators (otherwise does now show the wanted ticks
#         ax.xaxis.set_minor_locator(minorLocator)
        plt.xticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
        plt.grid(True, which='both', axis='both', alpha=0.2, linestyle='-', linewidth=1, zorder=1)  # linestyle='dashed', which='minor', axis='y',
        # grid(b=True, which='minor', axis='x', alpha=0.2, linestyle='solid', linewidth=0.5)  # linestyle='dashed', which='minor', axis='y',
        plt.xlabel(r'Number of edges ($m$)', labelpad=0)  # labelpad=0
        plt.ylabel(r'Time [sec]', labelpad=0)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        # print(ax.get_xaxis().get_minor_locator())

        if CREATE_PDF:
            plt.savefig(join(figure_directory, fig_filename), format='pdf',
                        dpi=None,
                        edgecolor='w',
                        orientation='portrait',
                        transparent=False,
                        bbox_inches='tight',
                        pad_inches=0.05,
                        # frameon=None
                        )
        if SHOW_PDF:
            showfig(join(figure_directory, fig_filename))  # shows actually created PDF
        if SHOW_PLOT:
            plt.show()


if __name__ == "__main__":
    run(3, 2, create_pdf=True, show_pdf=True)
