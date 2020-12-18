import numpy as np
import datetime
import random
import os                       # for displaying created PDF
import sys
sys.path.append('./../sslh')
from fileInteraction import save_csv_record
from utils import (from_dictionary_beliefs,
                   create_parameterized_H,
                   replace_fraction_of_rows,
                   to_centering_beliefs,
                   eps_convergence_linbp_parameterized,
                   matrix_difference,
                   introduce_errors,
                   showfig)
from estimation import (estimateH,
                        estimateH_baseline_serial)
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, xlabel, ylabel, savefig, show, xlim, ylim, xticks, grid, title)
import pandas as pd
pd.set_option('display.max_columns', None)      # show all columns from pandas
pd.options.mode.chained_assignment = None       # default='warn'
from graphGenerator import planted_distribution_model_H
from inference import linBP_symmetric_parameterized
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
    SHOW_PDF = show_pdf
    SHOW_PLOT = show_plot
    CREATE_PDF = create_pdf
    STD_FILL = True


    csv_filename = 'Fig_End-to-End_accuracy_VaryK_{}.csv'.format(CHOICE)
    header = ['currenttime',
              'option',
              'k',
              'f',
              'accuracy']
    if CREATE_DATA:
        save_csv_record(join(data_directory, csv_filename), header, append=False)


    # -- Default Graph parameters
    rep_SameGraph = 10       # iterations on same graph
    initial_h0 = None           # initial vector to start finding optimal H
    distribution = 'powerlaw'
    exponent = -0.3
    length = 5
    variant = 1
    EC = True                   # Non-backtracking for learning
    ymin = 0.3
    ymax = 1
    xmax = 8
    xtick_lab = [2,3,4,5,6,7, 8]
    xtick_labels = ['2', '3', '4', '5', '6', '7', '8']
    ytick_lab = np.arange(0, 1.1, 0.1)
    f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(21)]
    k_vec = [3, 4, 5 ]
    rep_DifferentGraphs = 10   # iterations on different graphs
    err = 0
    avoidNeighbors = False
    gradient = False
    pruneRandom = False
    convergencePercentage_W = None
    stratified = True
    label_vec = ['*'] * 10
    clip_on_vec = [False] * 10
    draw_std_vec = range(10)
    numberOfSplits = 1
    linestyle_vec = ['dashed'] + ['solid'] * 10
    linewidth_vec = [5, 4, 3, 3] + [3] * 10
    marker_vec = [None, None, 'o', 'x', 'o', '^', 'o', 'x', 'o', '^', 'o', 'x', 'o', '^']
    markersize_vec = [0, 0, 4, 8] + [6] * 10
    facecolor_vec = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]


    # -- Options with propagation variants
    if CHOICE == 500:     ## 1k nodes
        n = 1000
        h = 8
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3']
        learning_method_vec = ['GS', 'MHE', 'DHE']
        weight_vec = [10] * 3
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 2 + [True]
        xmin = 3.
        ymin = 0.
        ymax = 1.
        label_vec = ['GS', 'MCE', 'DCEr']
        facecolor_vec = ['black'] + ["#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974"] * 3
        f_vec = [0.03, 0.01, 0.001]
        k_vec = [3, 4, 5, 6]

    elif CHOICE == 501:        ## 10k nodes
        n = 10000
        h = 8
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3']
        learning_method_vec = ['GT', 'MHE', 'DHE']
        weight_vec = [10] * 3
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 2 + [True]
        xmin = 2.
        ymin = 0.
        ymax = 1.
        label_vec = ['GT', 'MCE', 'DCEr']
        facecolor_vec = ['black'] + ["#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974"] * 3
        f_vec = [0.03, 0.01, 0.001]
        k_vec = [2, 3, 4, 5]


    elif CHOICE == 502:        ## 10k nodes
        n = 10000
        h = 8
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE', 'Holdout']
        weight_vec = [10] * 10
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 4 + [True] + [False]
        xmin = 2
        ymin = 0.6
        ymax = 1.
        label_vec = ['GT', 'LCE', 'MCE', 'DCE', 'DCEr', 'Holdout']
        facecolor_vec = ['black'] + ["#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974"] * 3
        f_vec = [0.01]
        k_vec = [2, 3, 4, 5, 6, 7, 8]

        # option_vec = ['opt1', 'opt2', 'opt3', 'opt4']
        # learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE']
        # k_vec = [2, 3, 4, 5]



    elif CHOICE == 503:        ## 10k nodes
        n = 10000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE', 'Holdout']
        weight_vec = [10] * 10
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 4 + [True] + [False]
        xmin = 2
        ymin = 0.3
        ymax = 0.9
        label_vec = ['GT', 'LCE', 'MCE', 'DCE', 'DCEr', 'Holdout']
        facecolor_vec = ['black'] + ["#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974"] * 3
        f_vec = [0.01]
        k_vec = [2, 3, 4, 5, 6, 7, 8]
        # k_vec = [6, 7, 8]
        clip_on_vec = [True] * 10

        # option_vec = ['opt1', 'opt2', 'opt3', 'opt4']
        # learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE']
        # k_vec = [2, 3, 4, 5]



    elif CHOICE == 504:        ## 10k nodes
        n = 10000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE', 'Holdout']
        weight_vec = [10] * 10
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 4 + [True] + [False]
        xmin = 2
        xmax = 7
        ymin = 0.2
        ymax = 0.9
        label_vec = ['GT', 'LCE', 'MCE', 'DCE', 'DCEr', 'Holdout']
        facecolor_vec = ['black'] + ["#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974"] * 3
        f_vec = [0.01]
        # k_vec = [2, 3, 4, 5, 6, 7, 8]
        k_vec = [7]
        clip_on_vec = [True] * 10




    elif CHOICE == 505:        ## 10k nodes    with f = 0.005
        n = 10000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE', 'Holdout']
        weight_vec = [10] * 10
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 4 + [True] + [False]
        xmin = 2
        xmax = 7
        ymin = 0.2
        ymax = 0.9
        label_vec = ['GT', 'LCE', 'MCE', 'DCE', 'DCEr', 'Holdout']
        facecolor_vec = ['black'] + ["#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974"] * 3
        f_vec = [0.005]
        k_vec = [2, 3, 4, 5, 6, 7]
        # k_vec = [7]
        clip_on_vec = [True] * 10

    # elif CHOICE == 506:        ## 10k nodes    with f = 0.005
    #     n = 10000
    #     h = 3
    #     d = 25
    #     option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5']
    #     learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE']
    #     weight_vec = [10] * 10
    #     alpha_vec = [0] * 10
    #     beta_vec = [0] * 10
    #     gamma_vec = [0] * 10
    #     s_vec = [0.5] * 10
    #     numMaxIt_vec = [10] * 10
    #     randomize_vec = [False] * 4 + [True] + [False]
    #     xmin = 2
    #     xmax = 7
    #     ymin = 0.2
    #     ymax = 0.9
    #     label_vec = ['GT', 'LCE', 'MCE', 'DCE', 'DCEr']
    #     facecolor_vec = ['black'] + ["#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974"] * 3
    #     f_vec = [0.005]
    #     k_vec = [2,3,4,5,6,7]
    #     # k_vec = [7]
    #     clip_on_vec = [True] * 10




    elif CHOICE == 506:        ## 10k nodes
        n = 10000
        h = 8
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5']
        learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE', 'Holdout']
        learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE']
        weight_vec = [10] * 10
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 4 + [True] + [False]
        xmin = 2
        xmax = 7
        ymin = 0.2
        ymax = 0.9
        label_vec = ['GT', 'LCE', 'MCE', 'DCE', 'DCEr', 'Holdout']
        facecolor_vec = ['black'] + ["#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974"] * 3
        f_vec = [0.005]
        k_vec = [2, 3, 4, 5, 6, 7, 8]
        # k_vec = [5]
        clip_on_vec = [True] * 10

        rep_SameGraph = 1       # iterations on same graph
        rep_DifferentGraphs = 1  # iterations on same graph

    elif CHOICE == 507:  ## 10k nodes   with gradient and PruneRandom
        n = 10000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        learning_method_vec = ['GS', 'LHE', 'MHE', 'DHE', 'DHE', 'Holdout']
        weight_vec = [10] * 10
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 4 + [True] + [False]
        xmin = 2
        ymin = 0.1
        ymax = 0.9
        label_vec = ['GS', 'LCE', 'MCE', 'DCE', 'DCEr', 'Holdout']
        facecolor_vec = ['black'] + ["#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974"] * 3
        f_vec = [0.01]
        k_vec = [2, 3, 4, 5, 6, 7, 8]
        # k_vec = [6, 7, 8]
        clip_on_vec = [True] * 10

        # option_vec = ['opt1', 'opt2', 'opt3', 'opt4']
        # learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE']
        # k_vec = [2, 3, 4, 5]

        gradient = True
        pruneRandom = True


    elif CHOICE == 508:  ## 10k nodes   with gradient and PruneRandom
        n = 1000
        h = 3
        d = 10
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        learning_method_vec = ['GS', 'LHE', 'MHE', 'DHE', 'DHE', 'Holdout']
        weight_vec = [10] * 10
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 4 + [True] + [False]
        xmin = 2
        ymin = 0.1
        ymax = 0.9
        label_vec = ['GS', 'LCE', 'MCE', 'DCE', 'DCEr', 'Holdout']
        facecolor_vec = ['black'] + ["#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974"] * 3
        f_vec = [0.01]
        k_vec = [2, 3, 4, 5, 6, 7, 8]
        # k_vec = [6, 7, 8]
        clip_on_vec = [True] * 10

        # option_vec = ['opt1', 'opt2', 'opt3', 'opt4']
        # learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE']
        # k_vec = [2, 3, 4, 5]

        gradient = True
        pruneRandom = True
        rep_DifferentGraphs = 1
        rep_SameGraph = 1



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
                                H2c = H0c


                            elif learning_method == 'Holdout':


                                H2 = estimateH_baseline_serial(X2, ind, W, numMax=numMaxIt,
                                                               # ignore_rows=ind,
                                                               numberOfSplits=numberOfSplits,
                                                               # method=learning_method, variant=1, distance=length,
                                                               EC=EC,
                                                               alpha=alpha, beta=beta, gamma=gamma)
                                H2c = to_centering_beliefs(H2)

                            elif learning_method != 'DHE':
                                H2 = estimateH(X2, W, method=learning_method, variant=1, distance=length, EC=EC, weights=weights, randomize=randomize)
                                H2c = to_centering_beliefs(H2)

                            else:
                                H2 = estimateH(X2, W, method=learning_method, variant=1, distance=length, EC=EC, weights=weights, randomize=randomize, gradient=gradient, randomrestarts=pruneRandom)
                                H2c = to_centering_beliefs(H2)


                            # -- Propagation
                            X2c = to_centering_beliefs(X2, ignoreZeroRows=True)       # try without
                            eps_max = eps_convergence_linbp_parameterized(H2c, W,
                                                                          method='noecho',
                                                                          alpha=alpha, beta=beta, gamma=gamma,
                                                                          X=X2)
                            eps = s * eps_max
                            try:
                                F, actualIt, actualPercentageConverged = \
                                    linBP_symmetric_parameterized(X2, W, H2c * eps,
                                                                  method='noecho',
                                                                  alpha=alpha, beta=beta, gamma=gamma,
                                                                  numMaxIt=numMaxIt,
                                                                  convergencePercentage=convergencePercentage_W,
                                                                  debug=2)
                            except ValueError as e:
                                print (
                                "ERROR: {} with {}: d={}, h={}".format(e, learning_method, d, h))

                            else:
                                accuracy_X = matrix_difference(X0, F, ignore_rows=ind)

                                tuple = [str(datetime.datetime.now())]
                                text = [option_vec[option_index],
                                        k,
                                        f,
                                        accuracy_X]
                                # text = ['' if v is None else v for v in text]       # TODO: test with vocabularies
                                # text = np.asarray(text)         # without np, entries get ugly format
                                tuple.extend(text)
                                # print("option: {}, f: {}, actualIt: {}, accuracy: {}".format(option_vec[option_index], f, actualIt, accuracy_X))
                                save_csv_record(join(data_directory, csv_filename), tuple)


    # -- Read, aggregate, and pivot data for all options
    df1 = pd.read_csv(join(data_directory, csv_filename))
    # print("\n-- df1: (length {}):\n{}".format(len(df1.index), df1.head(15)))

    # -- Aggregate repetitions
    df2 = df1.groupby(['option', 'k', 'f']).agg \
        ({'accuracy': [np.mean, np.std, np.size, np.median],  # Multiple Aggregates
          })
    df2.columns = ['_'.join(col).strip() for col in df2.columns.values]  # flatten the column hierarchy
    df2.reset_index(inplace=True)  # remove the index hierarchy
    df2.rename(columns={'accuracy_size': 'count'}, inplace=True)
    # print("\n-- df2 (length {}):\n{}".format(len(df2.index), df2.head(15)))

    # -- Pivot table
    df3 = pd.pivot_table(df2, index=['f', 'k'], columns=['option'], values=[ 'accuracy_mean', 'accuracy_std'] )  # Pivot
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
            Y_hash[f][option] = df3.loc[df3['f'] == f]['accuracy_mean_{}'.format(option)].values
            Y_hash_std[f][option] = df3.loc[df3['f'] == f]['accuracy_std_{}'.format(option)].values




    if CREATE_PDF or SHOW_PLOT or SHOW_PDF:

        # -- Setup figure
        fig_filename = 'Fig_End-to-End_accuracy_varyK_{}.pdf'.format(CHOICE)
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

            # label = learning_method_vec[option_vec.index(option)]
            label = label_vec[option_vec.index(option)]
            # label = label + " " + str(f)

            if STD_FILL:


                # print((X_f))
                # print(Y_hash[f][option])


                ax.fill_between(X_f, Y_hash[f][option] + Y_hash_std[f][option], Y_hash[f][option] - Y_hash_std[f][option],
                                facecolor=color, alpha=0.2, edgecolor=None, linewidth=0)
                ax.plot(X_f, Y_hash[f][option] + Y_hash_std[f][option], linewidth=0.5, color='0.8', linestyle='solid')
                ax.plot(X_f, Y_hash[f][option] - Y_hash_std[f][option], linewidth=0.5, color='0.8', linestyle='solid')

            ax.plot(X_f, Y_hash[f][option], linewidth=linewidth, color=color, linestyle=linestyle, label=label, zorder=4, marker=marker,
                markersize=markersize, markeredgewidth=1, markeredgecolor='black', clip_on=clip_on)

        if CHOICE==507:
            Y_f = [1/float(i) for i in X_f]

            ax.plot(X_f, Y_f, linewidth=2, color='black', linestyle='dashed',
                    label='Random', zorder=4, marker='x',
                markersize=8, markeredgewidth=1, markeredgecolor='black', clip_on=clip_on)

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
                            loc='upper right',  # 'upper right'
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
        plt.xticks(xtick_lab, xtick_labels)
        plt.yticks(ytick_lab, ytick_lab)
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        grid(b=True, which='major', axis='both', alpha=0.2, linestyle='solid', linewidth=0.5)  # linestyle='dashed', which='minor', axis='y',
        grid(b=True, which='minor', axis='both', alpha=0.2, linestyle='solid', linewidth=0.5)  # linestyle='dashed', which='minor', axis='y',
        xlabel(r'Number of Classes $(k)$', labelpad=0)      # labelpad=0
        ylabel(r'Accuracy', labelpad=0)

        xlim(xmin, xmax)
        ylim(ymin, ymax)

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
            showfig(join(figure_directory, fig_filename))

if __name__ == "__main__":
    run(507, show_plot=True)
