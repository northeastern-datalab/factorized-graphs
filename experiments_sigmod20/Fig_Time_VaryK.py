
"""
Plots Time for labeling for various choices of k
"""

import numpy as np
import datetime
import time
import random
import os                       # for displaying created PDF
import sys
sys.path.append("./../sslh")    # add utils to path
from fileInteraction import save_csv_record
from utils import (from_dictionary_beliefs,
                              create_parameterized_H,
                              replace_fraction_of_rows,
                              to_centering_beliefs,
                              introduce_errors,
                              showfig)
from estimation import (estimateH, estimateH_baseline_serial)
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

from matplotlib.pyplot import (figure, xlabel, ylabel, savefig, show, xlim, ylim, xticks, grid, title)
import pandas as pd
pd.set_option('display.max_columns', None)      # show all columns from pandas
pd.options.mode.chained_assignment = None       # default='warn'
from graphGenerator import planted_distribution_model_H
from collections import defaultdict


# -- Determine path to data *irrespective* of where the file is run from
from os.path import abspath, dirname, join
from inspect import getfile, currentframe
current_path = dirname(abspath(getfile(currentframe())))
figure_directory = join(current_path, 'figs')
data_directory = join(current_path, 'datacache')


def run(choice, create_data=False, add_data=False, show_plot=False, create_pdf=False, show_pdf=False, shorten_length=False):
# -- Setup
    CHOICE = choice
    CREATE_DATA = create_data
    ADD_DATA = add_data
    SHOW_PLOT = show_plot
    SHOW_PDF = show_pdf
    CREATE_PDF = create_pdf
    SHOW_ARROWS = False
    STD_FILL = False

    CALCULATE_DATA_STATISTICS = False
    csv_filename = 'Fig_timing_VaryK_{}.csv'.format(CHOICE)
    header = ['currenttime',
              'option',
              'k',
              'f',
              'time']
    if CREATE_DATA:
        save_csv_record(join(data_directory, csv_filename), header, append=False)


    # -- Default Graph parameters
    rep_SameGraph = 2       # iterations on same graph
    initial_h0 = None           # initial vector to start finding optimal H
    distribution = 'powerlaw'
    exponent = -0.3
    length = 5
    variant = 1
    EC = True                   # Non-backtracking for learning
    ymin = 0.0
    ymax = 1
    xmin = 2
    xmax = 7.5
    xtick_lab = [2,3,4,5,6,7,8]
    xtick_labels = ['2', '3', '4', '5', '6', '7', '8']
    ytick_lab = [1e-3, 1e-2, 1e-1, 1, 10 ,50]
    ytick_labels = [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$1$', r'$10$', r'$50$']
    f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(21)]
    k_vec = [3, 4, 5 ]
    rep_DifferentGraphs = 1000   # iterations on different graphs
    err = 0
    avoidNeighbors = False
    gradient = False
    convergencePercentage_W = None
    stratified = True
    label_vec = ['*'] * 10
    clip_on_vec = [True] * 15
    draw_std_vec = range(10)
    numberOfSplits = 1
    linestyle_vec = ['solid'] * 15
    linewidth_vec = [3, 2, 4, 2, 3, 2] + [3] * 15
    marker_vec = ['^', 's', 'o', 'x', 'o', '+', 's'] * 3
    markersize_vec = [8, 7, 8, 10, 7, 6] + [10] * 10
    facecolor_vec = ["#CCB974", "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#64B5CD"]
    legend_location = 'upper right'


    # -- Options with propagation variants
    if CHOICE == 600:     ## 1k nodes
        n = 1000
        h = 8
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4']
        learning_method_vec = ['GT', 'MHE', 'DHE', 'Holdout']
        weight_vec = [10] * 4
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 4 + [True]
        xmin = 3.
        xmax = 10.
        ymin = 0.
        ymax = 50.
        label_vec = ['GT', 'MCE', 'DCE', 'Holdout']
        facecolor_vec = ['black'] + ["#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974"] * 4
        f_vec = [0.03, 0.01, 0.001]
        k_vec = [3, 4, 5, 6]
        ytick_lab = [0,1e-3, 1e-2, 1e-1, 1, 10, 50]
        ytick_labels = [r'$0$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$1$', r'$10$', r'$50$']


    elif CHOICE == 601:        ## 10k nodes
        n = 10000
        h = 8
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3',  'opt4']
        learning_method_vec = ['GT', 'MHE', 'DHE', 'Holdout']
        weight_vec = [10] * 4
        alpha_vec = [0] * 20
        beta_vec = [0] * 20
        gamma_vec = [0] * 20
        s_vec = [0.5] * 20
        numMaxIt_vec = [10] * 20
        randomize_vec = [False] * 15 + [True]
        xmin = 3.
        xmax = 8.
        ymin = 0.
        ymax = 500.
        label_vec = ['GT', 'MCE', 'DCE', 'Holdout']
        facecolor_vec = ['black'] + ["#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974"] * 4
        f_vec = [0.03, 0.01, 0.001]
        k_vec = [3, 4, 5]
        ytick_lab = [0, 1e-3, 1e-2, 1e-1, 1, 10, 100, 300]
        ytick_labels = [r'$0$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$1$', r'$10$', r'$100$', r'$300$']


    elif CHOICE == 602:        ## 10k nodes
        n = 10000
        h = 8
        d = 25
        weight_vec = [10] * 20
        alpha_vec = [0] * 20
        beta_vec = [0] * 20
        gamma_vec = [0] * 20
        s_vec = [0.5] * 20
        numMaxIt_vec = [10] * 20
        randomize_vec = [False] * 3 + [True] + [False]
        ymin = 0.01
        ymax = 500
        label_vec = ['Holdout', 'LCE', 'MCE', 'DCE', 'DHEr']
        facecolor_vec = ["#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974"] * 4
        f_vec = [0.01]
        k_vec = [3, 4, 5]
        ytick_lab = [1e-3, 1e-2, 1e-1, 1, 10, 100, 500]
        ytick_labels = [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$1$', r'$10$', r'$100$', r'$500$']

        option_vec = ['opt5', 'opt6', 'opt2', 'opt3', 'opt4']
        learning_method_vec = ['Holdout', 'LHE', 'MHE', 'DHE', 'DHE']
        k_vec = [2, 3, 4, 5, 6, 7, 8]

        # option_vec = ['opt2', 'opt3', 'opt6']
        # learning_method_vec = ['MHE', 'DHE', 'LHE']
        # k_vec = [2, 3, 4, 5]


    elif CHOICE == 603:        ## 10k nodes



        n = 10000
        h = 3
        d = 25
        weight_vec = [10] * 20
        alpha_vec = [0] * 20
        beta_vec = [0] * 20
        gamma_vec = [0] * 20
        s_vec = [0.5] * 20
        numMaxIt_vec = [10] * 20
        randomize_vec = [False] * 4 + [True]

        xmin = 1.8
        xmax = 8.2
        ymin = 0.01
        ymax = 500
        label_vec = ['Holdout', 'LCE', 'MCE', 'DCE', 'DCEr']
        facecolor_vec = ["#CCB974", "#55A868", "#4C72B0", "#8172B2", "#C44E52"] * 4
        f_vec = [0.01]
        k_vec = [3, 4, 5]
        ytick_lab = [1e-3, 1e-2, 1e-1, 1, 10, 100, 500]
        ytick_labels = [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$1$', r'$10$', r'$100$', r'$500$']

        option_vec = ['opt5', 'opt6', 'opt2', 'opt3', 'opt4']
        learning_method_vec = ['Holdout', 'LHE', 'MHE', 'DHE', 'DHE']
        k_vec = [2, 3, 4, 5, 6, 7, 8]

        legend_location = 'upper right'

        # option_vec = ['opt2', 'opt3', 'opt6']
        # learning_method_vec = ['MHE', 'DHE', 'LHE']
        # k_vec = [2, 3, 4, 5]

        # option_vec = ['opt4', 'opt3']
        # learning_method_vec = ['MHE', 'MHE']
        # randomize_vec = [True, False]
        # k_vec = [2, 3, 4, 5]



    elif CHOICE == 604:        ## 10k nodes with Gradient
        n = 10000
        h = 3
        d = 25
        weight_vec = [10] * 20
        alpha_vec = [0] * 20
        beta_vec = [0] * 20
        gamma_vec = [0] * 20
        s_vec = [0.5] * 20
        numMaxIt_vec = [10] * 20
        randomize_vec = [False] * 4 + [True]
        ymin = 0.00
        ymax = 800
        label_vec = ['Holdout', 'LCE', 'MCE', 'DCE', 'DCEr']
        facecolor_vec = ["#CCB974", "#55A868", "#4C72B0", "#8172B2", "#C44E52"] * 4
        f_vec = [0.01]
        k_vec = [3, 4, 5]
        ytick_lab = [1e-3, 1e-2, 1e-1, 1, 10, 100, 500]
        ytick_labels = [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$1$', r'$10$', r'$100$', r'$500$']

        option_vec = ['opt5', 'opt6', 'opt2', 'opt3', 'opt4']
        learning_method_vec = ['Holdout', 'LHE', 'MHE', 'DHE', 'DHE']
        k_vec = [2, 3, 4, 5, 6, 7, 8]
        # k_vec = [7, 8]
        gradient = True
        legend_location = 'center right'




    elif CHOICE == 605:        ## 10k nodes with Gradient   with f = 0.005
        n = 10000
        h = 3
        d = 25
        weight_vec = [10] * 20
        alpha_vec = [0] * 20
        beta_vec = [0] * 20
        gamma_vec = [0] * 20
        s_vec = [0.5] * 20
        numMaxIt_vec = [10] * 20
        randomize_vec = [False] * 4 + [True]
        ymin = 0.00
        ymax = 800
        label_vec = ['Holdout', 'LCE', 'MCE', 'DCE', 'DCEr']
        facecolor_vec = ["#CCB974", "#55A868", "#4C72B0", "#8172B2", "#C44E52"] * 4
        f_vec = [0.005]
        k_vec = [3, 4, 5]
        ytick_lab = [1e-3, 1e-2, 1e-1, 1, 10, 100, 500]
        ytick_labels = [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$1$', r'$10$', r'$100$', r'$500$']

        option_vec = ['opt5', 'opt6', 'opt2', 'opt3', 'opt4']
        learning_method_vec = ['Holdout', 'LHE', 'MHE', 'DHE', 'DHE']
        k_vec = [2, 3, 4, 5, 6, 7]
        # k_vec = [7, 8]
        gradient = True
        legend_location = 'center right'



    elif CHOICE == 606:        ## 10k nodes with Gradient   with f = 0.005 and Gradient and PruneRandom
        n = 10000
        h = 3
        d = 25
        weight_vec = [10] * 20
        alpha_vec = [0] * 20
        beta_vec = [0] * 20
        gamma_vec = [0] * 20
        s_vec = [0.5] * 20
        numMaxIt_vec = [10] * 20
        randomize_vec = [False] * 4 + [True]

        xmin = 1.8
        xmax = 7.2
        ymin = 0.01
        ymax = 800
        label_vec = ['Holdout', 'LCE', 'MCE', 'DCE', 'DCEr']
        facecolor_vec = ["#CCB974", "#55A868", "#4C72B0", "#8172B2", "#C44E52"] * 4
        f_vec = [0.005]
        k_vec = [3, 4, 5]
        ytick_lab = [1e-3, 1e-2, 1e-1, 1, 10, 100, 500]
        ytick_labels = [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$1$', r'$10$', r'$100$', r'$500$']

        option_vec = ['opt5', 'opt6', 'opt2', 'opt3', 'opt4']
        learning_method_vec = ['Holdout', 'LHE', 'MHE', 'DHE', 'DHE']
        k_vec = [2, 3, 4, 5, 6, 7]

        gradient = True
        pruneRandom = True
        legend_location = 'upper right'


    elif CHOICE == 607:  ## 10k nodes   with gradient and PruneRandom
        n = 10000
        h = 3
        d = 25
        option_vec = ['opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        learning_method_vec = ['LHE', 'MHE', 'DHE', 'DHE', 'Holdout']
        weight_vec = [10] * 10
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 3 + [True] + [False]

        xmin = 1.8
        xmax = 7.
        ymin = 0.01
        ymax = 800
        label_vec = ['LCE', 'MCE', 'DCE', 'DCEr', 'Holdout']
        facecolor_vec = ["#55A868", "#4C72B0", "#8172B2", "#C44E52","#CCB974"] * 4
        legend_location = 'upper left'
        marker_vec = [None, 's', 'x', 'o', '^', '+'] * 3
        markersize_vec = [8, 7, 10, 8, 7, 6] + [10] * 10
        f_vec = [0.01]
        k_vec = [2, 3, 4, 5, 6, 7, 8]
        clip_on_vec = [True] * 10
        gradient = True
        pruneRandom = True
        ytick_lab = [1e-3, 1e-2, 1e-1, 1, 10, 100, 500]
        ytick_labels = [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$1$', r'$10$', r'$100$', r'$500$']

    elif CHOICE == 608:  ## 10k nodes   with gradient and PruneRandom
        n = 10000
        h = 3
        d = 25
        option_vec = ['opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        learning_method_vec = ['LHE', 'MHE', 'DHE', 'DHE', 'Holdout']
        weight_vec = [10] * 10
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 3 + [True] + [False]

        xmin = 1.8
        xmax = 7.2
        ymin = 0.01
        ymax = 800
        label_vec = ['LCE', 'MCE', 'DCE', 'DCEr', 'Holdout']
        facecolor_vec = ["#55A868", "#4C72B0", "#8172B2", "#C44E52","#CCB974"] * 4
        legend_location = 'upper left'
        marker_vec = [None, 's', 'x', 'o', '^', '+'] * 3
        markersize_vec = [8, 7, 10, 8, 7, 6] + [10] * 10
        f_vec = [0.01]
        k_vec = [2, 3, 4, 5, 6, 7, 8]
        clip_on_vec = [True] * 10
        gradient = True
        pruneRandom = True
        ytick_lab = [1e-3, 1e-2, 1e-1, 1, 10, 100, 500]
        ytick_labels = [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$1$', r'$10$', r'$100$', r'$500$']
        rep_DifferentGraphs = 10


    else:
        raise Warning("Incorrect choice!")


    RANDOMSEED = None  # For repeatability
    random.seed(RANDOMSEED)  # seeds some other python random generator
    np.random.seed(seed=RANDOMSEED)  # seeds the actually used numpy random generator; both are used and thus needed
    # print("CHOICE: {}".format(CHOICE))


    # -- Create data
    if CREATE_DATA or ADD_DATA:
        for i in range(rep_DifferentGraphs):  # create several graphs with same parameters
            # print("\ni: {}".format(i))

            for k in k_vec:
                # print("\nk: {}".format(k))

                H0 = create_parameterized_H(k, h, symmetric=True)
                H0c = to_centering_beliefs(H0)

                a = [1.] * k
                alpha0 = np.array(a)
                alpha0 = alpha0 / np.sum(alpha0)

                W, Xd = planted_distribution_model_H(n, alpha=alpha0, H=H0, d_out=d,
                                                          distribution=distribution,
                                                          exponent=exponent,
                                                          directed=False,
                                                          debug=False)
                X0 = from_dictionary_beliefs(Xd)

                for j in range(rep_SameGraph):  # repeat several times for same graph
                    # print("j: {}".format(j))

                    ind = None
                    for f in f_vec:             # Remove fraction (1-f) of rows from X0 (notice that different from first implementation)
                        X1, ind = replace_fraction_of_rows(X0, 1-f, avoidNeighbors=avoidNeighbors, W=W, ind_prior=ind, stratified=stratified)
                        X2 = introduce_errors(X1, ind, err)

                        for option_index, (learning_method, alpha, beta, gamma, s, numMaxIt, weights, randomize) in \
                                enumerate(zip(learning_method_vec, alpha_vec, beta_vec, gamma_vec, s_vec, numMaxIt_vec, weight_vec, randomize_vec)):

                            # -- Learning
                            if learning_method == 'GT':
                                timeTaken = 0.0


                            elif learning_method == 'Holdout':

                                prev_time = time.time()
                                H2 = estimateH_baseline_serial(X2, ind, W, numMax=numMaxIt,
                                                               numberOfSplits=numberOfSplits,
                                                               EC=EC,
                                                               alpha=alpha, beta=beta, gamma=gamma)
                                timeTaken = time.time() - prev_time

                            else:
                                prev_time = time.time()
                                if gradient and pruneRandom:
                                    H2 = estimateH(X2, W, method=learning_method, variant=1, distance=length, EC=EC, weights=weights, randomize=randomize, gradient=gradient)
                                else:
                                    H2 = estimateH(X2, W, method=learning_method, variant=1, distance=length, EC=EC, weights=weights, randomize=randomize)
                                timeTaken = time.time() - prev_time


                            tuple = [str(datetime.datetime.now())]
                            text = [option_vec[option_index],
                                    k,
                                    f,
                                    timeTaken]
                            tuple.extend(text)
                            # print("option: {}, f: {}, timeTaken: {}".format(option_vec[option_index], f, timeTaken))
                            save_csv_record(join(data_directory, csv_filename), tuple)


    # -- Read, aggregate, and pivot data for all options
    df1 = pd.read_csv(join(data_directory, csv_filename))
    # print("\n-- df1: (length {}):\n{}".format(len(df1.index), df1.head(15)))

    # -- Aggregate repetitions
    df2 = df1.groupby(['option', 'k', 'f']).agg \
        ({'time': [np.mean, np.std, np.size, np.median],  # Multiple Aggregates
          })
    df2.columns = ['_'.join(col).strip() for col in df2.columns.values]  # flatten the column hierarchy
    df2.reset_index(inplace=True)  # remove the index hierarchy
    df2.rename(columns={'time_size': 'count'}, inplace=True)
    # print("\n-- df2 (length {}):\n{}".format(len(df2.index), df2.head(15)))

    # -- Pivot table
    df3 = pd.pivot_table(df2, index=['f', 'k'], columns=['option'], values=[ 'time_mean', 'time_std', 'time_median'] )  # Pivot
    # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(30)))
    df3.columns = ['_'.join(col).strip() for col in df3.columns.values]  # flatten the column hierarchy
    df3.reset_index(inplace=True)  # remove the index hierarchy
    # df2.rename(columns={'time_size': 'count'}, inplace=True)
    # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(100)))



    # X_f = k_vec
    X_f = df3['k'].values            # read k from values instead

    Y_hash = defaultdict(dict)
    Y_hash_std = defaultdict(dict)

    for f in f_vec:
        for option in option_vec:
            Y_hash[f][option] = list()
            Y_hash_std[f][option] = list()

    for f in f_vec:
        for option in option_vec:
            Y_hash[f][option] = df3.loc[df3['f'] == f]['time_mean_{}'.format(option)].values            # mean
            # Y_hash[f][option] = df3.loc[df3['f'] == f]['time_median_{}'.format(option)].values          # median
            Y_hash_std[f][option] = df3.loc[df3['f'] == f]['time_std_{}'.format(option)].values




    if SHOW_PLOT or SHOW_PDF or CREATE_PDF:

        # -- Setup figure
        fig_filename = 'Fig_Time_varyK_{}.pdf'.format(CHOICE)
        mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': [u'Arial', u'Liberation Sans']})
        mpl.rcParams['axes.labelsize'] = 20
        mpl.rcParams['xtick.labelsize'] = 16
        mpl.rcParams['ytick.labelsize'] = 16
        mpl.rcParams['legend.fontsize'] = 14
        mpl.rcParams['grid.color'] = '777777'  # grid color
        mpl.rcParams['xtick.major.pad'] = 2  # padding of tick labels: default = 4
        mpl.rcParams['ytick.major.pad'] = 1  # padding of tick labels: default = 4
        mpl.rcParams['xtick.direction'] = 'out'  # default: 'in'
        mpl.rcParams['ytick.direction'] = 'out'  # default: 'in'
        mpl.rcParams['axes.titlesize'] = 16
        mpl.rcParams['figure.figsize'] = [4, 4]
        fig = figure()
        ax = fig.add_axes([0.13, 0.17, 0.8, 0.8])


        opt_f_vecs = [(option, f) for option in option_vec for f in f_vec]

        for ((option, f), color, linewidth, clip_on, linestyle, marker, markersize) in \
            zip(opt_f_vecs, facecolor_vec, linewidth_vec, clip_on_vec, linestyle_vec, marker_vec, markersize_vec):

            label = label_vec[option_vec.index(option)]
            # label = label + " " + str(f)

            if STD_FILL:
                ax.fill_between(X_f, Y_hash[f][option] + Y_hash_std[f][option], Y_hash[f][option] - Y_hash_std[f][option],
                                facecolor=color, alpha=0.2, edgecolor=None, linewidth=0)
                ax.plot(X_f, Y_hash[f][option] + Y_hash_std[f][option], linewidth=0.5, color='0.8', linestyle='solid')
                ax.plot(X_f, Y_hash[f][option] - Y_hash_std[f][option], linewidth=0.5, color='0.8', linestyle='solid')

            ax.plot(X_f, Y_hash[f][option], linewidth=linewidth, color=color, linestyle=linestyle, label=label, zorder=4, marker=marker,
                markersize=markersize, markeredgecolor='black', markeredgewidth=1, clip_on=clip_on)


        if SHOW_ARROWS:
            for indx in [2,3]:
                ax.annotate(s='', xy=(X_f[indx]-0.05, Y_hash[f]['opt4'][indx]), xytext=(X_f[indx]-0.05, Y_hash[f]['opt5'][indx]), arrowprops=dict(facecolor='blue',  arrowstyle='<->'))
                ax.annotate(str(int(np.round(Y_hash[f]['opt5'][indx]/Y_hash[f]['opt4'][indx])))+'x', xy=(X_f[indx]-0.4, (Y_hash[f]['opt5'][indx]+Y_hash[f]['opt4'][indx])/10 ), color='black', va='center', annotation_clip=False, zorder=5)


        # -- Title and legend
        if distribution == 'uniform':
            distribution_label = ',$uniform'
        else:
            distribution_label = '$'
        if n < 1000:
            n_label='{}'.format(n)
        else:
            n_label = '{}k'.format(int(n / 1000))


        title(r'$\!\!\!n\!=\!{}, d\!=\!{}, h\!=\!{}, f\!=\!{}{}'.format(n_label, d, h, f, distribution_label))
        handles, label_vec = ax.get_legend_handles_labels()
        legend = plt.legend(handles, label_vec,
                            loc=legend_location,  # 'upper right'
                            handlelength=2,
                            labelspacing=0,  # distance between label entries
                            handletextpad=0.3,  # distance between label and the line representation
                            borderaxespad=0.2,  # distance between legend and the outer axes
                            borderpad=0.3,  # padding inside legend box
                            numpoints=1,  # put the marker only once
                            )
        # # legend.set_zorder(1)
        frame = legend.get_frame()
        frame.set_linewidth(0.0)
        frame.set_alpha(0.9)  # 0.8


        # -- Figure settings and save
        plt.yscale('log')
        plt.xticks(xtick_lab, xtick_labels)
        plt.yticks(ytick_lab, ytick_lab)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        grid(b=True, which='major', axis='both', alpha=0.2, linestyle='solid', linewidth=0.5)  # linestyle='dashed', which='minor', axis='y',
        grid(b=True, which='minor', axis='both', alpha=0.2, linestyle='solid', linewidth=0.5)  # linestyle='dashed', which='minor', axis='y',
        xlabel(r'Number of Classes $(k)$', labelpad=0)      # labelpad=0
        ylabel(r'Time [sec]', labelpad=0)



        if CREATE_PDF:
            savefig(join(figure_directory, fig_filename), format='pdf',
                    dpi=None,
                    edgecolor='w',
                    orientation='portrait',
                    transparent=False,
                    bbox_inches='tight',
                    pad_inches=0.05,
                    frameon=None)

        if SHOW_PLOT:
            plt.show()

        if SHOW_PDF:
            showfig(join(figure_directory, fig_filename))  # shows actually created PDF

if __name__ == "__main__":
    run(607, show_plot=True)
