"""
Plot time for LCE, MCE, DCE and Holdout set

"""

import numpy as np
import datetime
import time
import random
import os      # for displaying created PDF
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
from estimation import  estimateH, estimateH_baseline_serial, estimateH_baseline_parallel
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import collections      # remove error bars
from matplotlib.pyplot import figure, xlabel, ylabel, savefig, show, xlim, ylim, xticks, grid, title
import pandas as pd
pd.set_option('display.max_columns', None)      # show all columns from pandas
pd.options.mode.chained_assignment = None       # default='warn'
from graphGenerator import planted_distribution_model_H, planted_distribution_model_H
from inference import linBP_symmetric_parameterized


# -- Determine path to data *irrespective* of where the file is run from
from os.path import abspath, dirname, join
from inspect import getfile, currentframe
current_path = dirname(abspath(getfile(currentframe())))
figure_directory = join(current_path, 'figs')
data_directory = join(current_path, 'datacache')

def run(choice, create_data=False, add_data=False, show_plot=False, create_pdf=False, show_pdf=False, shorten_length=False,
        show_arrows=False):
    # -- Setup
    CHOICE = choice
    CREATE_DATA = create_data
    ADD_DATA = add_data
    SHOW_PLOT = show_plot
    SHOW_PDF = show_pdf
    CREATE_PDF = create_pdf

    SHOW_STD = True         ## FALSE for just scatter plot points
    SHOW_ARROWS = show_arrows


    # -- Default Graph parameters
    rep_SameGraph = 1       # iterations on same graph
    distribution = 'powerlaw'
    exponent = -0.3
    length = 5
    variant = 1
    EC = False
    numberOfSplits = 1
    scaling_vec = [None]*10
    ymin = 0.3
    ymax = 1
    xmin = 1e-3
    xmax = 1e3
    xtick_lab = [1e-3, 0.01, 0.1, 1, 10, 100, 1000]
    xtick_labels = [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$1$', r'$10$', r'$10^{2}$', r'$10^{3}$']
    ytick_lab = np.arange(0, 1.1, 0.1)
    k = 3
    a = 1
    rep_DifferentGraphs = 1   # iterations on different graphs
    err = 0
    avoidNeighbors = False
    convergencePercentage_W = 0.99
    facecolor_vec = ["#4C72B0", "#55A868", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
    label_vec = ['MCE', 'LCE', 'DCE', 'Holdout']
    linewidth_vec = [4, 3, 1, 2, 2, 1]
    # clip_ons = [True, True, True, True, True, True]
    FILEZNAME = 'Fig_timing_accuracy_learning'
    marker_vec = ['s', '^', 'v', 'o', 'x', '+', 'None']   #'^'
    length_vec = [5]
    stratified = True
    f = 0.01
    numMaxIt_vec = [10]*7
    alpha_vec = [0] * 7
    beta_vec = [0] * 7  # TODO: LinBP does not use beta. Also SSLH uses alpha, but not beta for W^row! Now fixed
    gamma_vec = [0] * 7
    s_vec = [0.5] * 7


    # -- Main Options
    if CHOICE == 1:         # Main graph
        n = 1000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        learning_method_vec = ['MHE']  + ['LHE'] + ['DHE'] + ['DHE'] + ['Holdout'] + ['GS']
        label_vec = ['MCE', 'LCE', 'DCE', 'DCE r', 'Holdout', 'GS']
        randomize_vec = [False]*3 + [True] + [None]*2
        scaling_vec = [None]*2 + [10, 100] + [None]*2
        splits_vec = [1, 2, 4, 8]


    elif CHOICE == 2:
        n = 1000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5']
        learning_method_vec = ['MHE']  + ['LHE'] + ['DHE'] + ['DHE'] + ['GS']
        label_vec = ['MCE', 'LCE', 'DCE', 'DCE r', 'GS']
        randomize_vec = [False]*3 + [True] + [None]
        scaling_vec = [None]*2 + [10, 100] + [None]


    elif CHOICE == 3:
        n = 1000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5']
        learning_method_vec = ['MHE']  + ['LHE'] + ['DHE'] + ['DHE'] + ['GS']
        label_vec = ['MCE', 'LCE', 'DCE', 'DCE r', 'GS']
        randomize_vec = [False]*3 + [True] + [None]
        scaling_vec = [None]*2 + [10, 100] + [None]
        f = 0.02


    elif CHOICE == 4:         # TODO: Overnight Wolfgang
        n = 1000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        learning_method_vec = ['MHE']  + ['LHE'] + ['DHE'] + ['DHE'] + ['Holdout'] + ['GS']
        label_vec = ['MCE', 'LCE', 'DCE', 'DCE r', 'Holdout', 'GS']
        randomize_vec = [False]*3 + [True] + [None]*2
        scaling_vec = [None]*2 + [10, 100] + [None]*2
        splits_vec = [1, 2, 4, 8, 16]


    elif CHOICE == 5:         # Toy graph with 100 nodes
        n = 100
        h = 3
        d = 8
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        learning_method_vec = ['MHE']  + ['LHE'] + ['DHE'] + ['DHE'] + ['Holdout'] + ['GS']
        label_vec = ['MCE', 'LCE', 'DCE', 'DCE r', 'Holdout', 'GS']
        randomize_vec = [False]*3 + [True] + [None]*2
        scaling_vec = [None]*2 + [10, 100] + [None]*2
        splits_vec = [1, 2, 4, 8]
        f=0.05


    elif CHOICE == 6:         # To be run by Prakhar on Cluster
        n = 10000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        learning_method_vec = ['MHE']  + ['LHE'] + ['DHE'] + ['DHE'] + ['Holdout'] + ['GS']
        label_vec = ['MCE', 'LCE', 'DCE', 'DCEr', 'Holdout', 'GS']
        randomize_vec = [False]*3 + [True] + [None]*2
        scaling_vec = [None]*2 + [10, 100] + [None]*2
        splits_vec = [1, 2, 4, 8]
        f=0.003
        xmin = 1e-2
        # ymax = 0.9
        ymin = 0.2
        ymax = 0.9
        xmin = 1e-2
        xmax = 1e3



    elif CHOICE == 7:
        n = 1000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        learning_method_vec = ['MHE']  + ['LHE'] + ['DHE'] + ['DHE'] + ['Holdout'] + ['GS']
        label_vec = ['MCE', 'LCE', 'DCE', 'DCE r', 'Holdout', 'GS']
        randomize_vec = [False]*3 + [True] + [None]*2
        scaling_vec = [None]*2 + [10, 100] + [None]*2
        splits_vec = [1, 2, 4, 8, 16]
        f=0.009

    # elif CHOICE == 8:       # not working well
    #     n = 1000
    #     h = 3
    #     d = 25
    #     option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
    #     learning_method_vec = ['MHE']  + ['LHE'] + ['DHE'] + ['DHE'] + ['Holdout'] + ['GS']
    #     label_vec = ['MCE', 'LCE', 'DCE', 'DCE r', 'Holdout', 'GS']
    #     randomize_vec = [False]*3 + [True] + [None]*2
    #     scaling_vec = [None]*2 + [10, 100] + [None]*2
    #     splits_vec = [1, 2, 4, 8, 16]
    #     f=0.005



    else:
        raise Warning("Incorrect choice!")



    csv_filename = '{}_{}.csv'.format(FILEZNAME, CHOICE)
    header = ['currenttime',
              'option',
              'lensplit',
              'f',
              'accuracy',
              'timetaken']
    if CREATE_DATA:
        save_csv_record(join(data_directory, csv_filename), header, append=False)

    alpha0 = np.array([a, 1., 1.])
    alpha0 = alpha0 / np.sum(alpha0)
    H0 = create_parameterized_H(k, h, symmetric=True)
    H0c = to_centering_beliefs(H0)


    RANDOMSEED = None  # For repeatability
    random.seed(RANDOMSEED)  # seeds some other python random generator
    np.random.seed(seed=RANDOMSEED)  # seeds the actually used numpy random generator; both are used and thus needed
    # print("CHOICE: {}".format(CHOICE))


    # -- Create data
    if CREATE_DATA or ADD_DATA:
        for i in range(rep_DifferentGraphs):  # create several graphs with same parameters
            # print("\ni: {}".format(i))

            W, Xd = planted_distribution_model_H(n, alpha=alpha0, H=H0, d_out=d,
                                                      distribution=distribution,
                                                      exponent=exponent,
                                                      directed=False,
                                                      debug=False)
            X0 = from_dictionary_beliefs(Xd)

            for j in range(rep_SameGraph):  # repeat several times for same graph
                # print("j: {}".format(j))

                ind = None
                X1, ind = replace_fraction_of_rows(X0, 1-f, avoidNeighbors=avoidNeighbors, W=W, ind_prior=ind, stratified = stratified)     # TODO: stratified sampling option = True
                X2 = introduce_errors(X1, ind, err)

                for option_index, (learning_method, alpha, beta, gamma, s, numMaxIt, weight, randomize, option) in \
                        enumerate(zip(learning_method_vec, alpha_vec, beta_vec, gamma_vec, s_vec, numMaxIt_vec, scaling_vec, randomize_vec, option_vec)):

                    # weight = np.array([np.power(scaling, i) for i in range(5)])       # TODO: now enough to specify weight as a scalar!
                    H_est_dict = {}
                    timeTaken_dict = {}

                    # -- Learning
                    if learning_method == 'Holdout' :
                        for numberOfSplits in splits_vec:
                            prev_time = time.time()
                            H_est_dict[numberOfSplits] = estimateH_baseline_serial(X2, ind, W, numMax=numMaxIt,
                                                                                   # ignore_rows=ind,
                                                                                   numberOfSplits=numberOfSplits,
                                                                                   # method=learning_method, variant=1, distance=length,
                                                                                   EC=EC,
                                                                                   weights=weight, alpha=alpha, beta=beta, gamma=gamma)
                            timeTaken = time.time() - prev_time
                            timeTaken_dict[numberOfSplits] = timeTaken

                    elif learning_method in ['LHE', 'MHE', 'DHE']:      # TODO: no smartInit, just randomization as option
                        for length in length_vec:
                            prev_time = time.time()
                            H_est_dict[length] = estimateH(X2, W, method=learning_method, variant=1, randomize=randomize, distance=length, EC=EC, weights=weight)
                            timeTaken = time.time() - prev_time
                            timeTaken_dict[length] = timeTaken

                    elif learning_method == 'GS':
                        H_est_dict['GS'] = H0

                    for key in H_est_dict:
                        H_est = H_est_dict[key]
                        H2c = to_centering_beliefs(H_est)
                        # print("H_estimated by {} is \n".format(learning_method), H_est)
                        # print("H0 is \n", H0)
                        # print("randomize was: ", randomize)

                        # Propagation
                        X2c = to_centering_beliefs(X2, ignoreZeroRows=True)  # try without
                        eps_max = eps_convergence_linbp_parameterized(H2c, W,
                                                                      method='noecho',
                                                                      alpha=alpha, beta=beta, gamma=gamma,
                                                                      X=X2)

                        eps = s * eps_max

                        # print("Max Eps ", eps_max)

                        try:
                            F, actualIt, actualPercentageConverged = \
                                linBP_symmetric_parameterized(X2, W, H2c * eps,
                                                              method='noecho',
                                                              alpha=alpha, beta=beta, gamma=gamma,
                                                              numMaxIt=numMaxIt,
                                                              convergencePercentage=convergencePercentage_W,
                                                              convergenceThreshold=0.99,
                                                              debug=2)

                        except ValueError as e:
                            print(
                                "ERROR: {} with {}: d={}, h={}".format(e, learning_method, d, h))

                        else:
                            accuracy_X = matrix_difference(X0, F, ignore_rows=ind)

                            tuple = [str(datetime.datetime.now())]
                            if learning_method == 'Holdout':
                                text = [option,"split{}".format(key), f, accuracy_X, timeTaken_dict[key]]
                            elif learning_method in ['MHE', 'DHE', 'LHE']:
                                text = [option, "len{}".format(key), f, accuracy_X, timeTaken_dict[key]]
                            elif learning_method == 'GS':
                                text = [option, 0, f, accuracy_X, 0]

                            tuple.extend(text)
                            # print("option: {}, f: {}, actualIt: {}, accuracy: {}".format(option, f, actualIt, accuracy_X))
                            save_csv_record(join(data_directory, csv_filename), tuple)





    # -- Read, aggregate, and pivot data for all options
    df1 = pd.read_csv(join(data_directory, csv_filename))
    # print("\n-- df1: (length {}):\n{}".format(len(df1.index), df1.head(15)))

    # Aggregate repetitions
    df2 = df1.groupby(['option', 'lensplit', 'f']).agg \
        ({'accuracy': [np.mean, np.std, np.size],  # Multiple Aggregates
          })
    df2.columns = ['_'.join(col).strip() for col in df2.columns.values]  # flatten the column hierarchy
    df2.reset_index(inplace=True)  # remove the index hierarchy
    df2.rename(columns={'accuracy_size': 'count'}, inplace=True)
    # print("\n-- df2 (length {}):\n{}".format(len(df2.index), df2.head(15)))

    df3 = df1.groupby(['option', 'lensplit', 'f']).agg({'timetaken': [np.median] })
    df3.columns = ['_'.join(col).strip() for col in df3.columns.values]  # flatten the column hierarchy
    df3.reset_index(inplace=True)  # remove the index hierarchy
    # resultdf3 = df3.sort(['timetaken'], ascending=1)
    # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(15)))

    X_time_median_dict = {}
    Y_acc_dict = {}
    Y_std_dict = {}

    for option in option_vec:
        Y_acc_dict[option] = df2.loc[(df2['option'] == option), "accuracy_mean"].values
        Y_std_dict[option] = df2.loc[(df2['option'] == option), "accuracy_std"].values
        X_time_median_dict[option] = df3.loc[(df3['option'] == option), "timetaken_median"].values

        # print("option: ", option)
        # print("Y_acc_dict[option]: ", Y_acc_dict[option])
        # print("Y_std_dict[option]: ", Y_std_dict[option])
        # print("X_time_median_dict[option]: ", X_time_median_dict[option])



    # -- Setup figure
    fig_filename = '{}_{}.pdf'.format(FILEZNAME, CHOICE)
    mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': [u'Arial', u'Liberation Sans']})
    mpl.rcParams['axes.labelsize'] = 18
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['legend.fontsize'] = 14
    mpl.rcParams['grid.color'] = '777777'  # grid color
    mpl.rcParams['xtick.major.pad'] = 2  # padding of tick labels: default = 4
    mpl.rcParams['ytick.major.pad'] = 1  # padding of tick labels: default = 4
    mpl.rcParams['xtick.direction'] = 'out'  # default: 'in'
    mpl.rcParams['ytick.direction'] = 'out'  # default: 'in'
    mpl.rcParams['figure.figsize'] = [4, 4]
    fig = figure()
    ax = fig.add_axes([0.13, 0.17, 0.8, 0.8])


    SHOW_ARROWS = True

    for choice, color, learning_method, label, linewidth, marker in \
            zip(option_vec, facecolor_vec, learning_method_vec, label_vec, linewidth_vec, marker_vec):

        if learning_method == 'Holdout':
            # Draw std
            X1 = X_time_median_dict[choice]
            s = X1.argsort()
            X1 = X1[s]
            Y1 = Y_acc_dict[choice][s]
            Y2 = Y_std_dict[choice][s]

            if SHOW_STD:
                ax.fill_between(X1, Y1 + Y2, Y1 - Y2, facecolor=color, alpha=0.2, edgecolor=None, linewidth=0)
                ax.plot(X1, Y1 + Y2, linewidth=0.5, color='0.8', linestyle='solid')
                ax.plot(X1, Y1 - Y2, linewidth=0.5, color='0.8', linestyle='solid')
                ax.set_ylim(bottom=ymin)

                ax.plot(X1, Y1, linewidth=linewidth, color=color, linestyle='solid', label=label, zorder=20, marker='x', markersize=linewidth + 5, markeredgewidth=1)
                ax.annotate(np.round(X1[1], decimals=1), xy=(X1[1], Y1[1] - 0.05), color=color, va='center', annotation_clip=False, zorder=5)

            else:
                ax.scatter(list(X1), list(Y1),
                           color=color, label=label, marker='x', s=42)


        elif learning_method == 'GS':
            ax.plot([1e-4, 1e4], [Y_acc_dict[choice], Y_acc_dict[choice]],
                    linewidth=1, color='black',
                    linestyle='dashed', zorder=0,
                    marker=None,
                    label=label,
                    )

        else:       # For all other
            if SHOW_STD:
                ax.errorbar(list(X_time_median_dict[choice]), list(Y_acc_dict[choice]), yerr=Y_std_dict[choice],
                            fmt='-o', linewidth=2, color=color,
                            label=label, marker=marker, markersize=8)
                ax.annotate(np.round(X_time_median_dict[choice], decimals=2), xy=(X_time_median_dict[choice], Y_acc_dict[choice]-0.05), color=color, va='center',
                            annotation_clip=False, zorder=5)

            else:
                ax.scatter(list(X_time_median_dict[choice]), list(Y_acc_dict[choice]),
                           color=color, label=label, marker=marker, s=42)

        if SHOW_ARROWS:
            dce_opt = 'opt4'
            holdout_opt = 'opt5'

            ax.annotate(s='', xy=(X_time_median_dict[dce_opt], Y_acc_dict[dce_opt]-0.3), xytext=(X_time_median_dict[holdout_opt][2]+0.02, Y_acc_dict[dce_opt]-0.3), arrowprops=dict(arrowstyle='<->'))
            ax.annotate(str(int(np.round(X_time_median_dict[holdout_opt][2] / X_time_median_dict[dce_opt]))) + 'x', xy=((X_time_median_dict[dce_opt] + X_time_median_dict[holdout_opt][2])/100, Y_acc_dict[dce_opt]-0.28),
                        color='black', va='center',
                        # bbox = dict(boxstyle="round,pad=0.3", fc="w"),
                        annotation_clip=False, zorder=5)






    # -- Title and legend
    title(r'$\!\!\!n\!=\!{}\mathrm{{k}}, d\!=\!{}, h\!=\!{}, f\!=\!{}$'.format(int(n / 1000), d, h, f))
    handles, label_vec = ax.get_legend_handles_labels()
    for i, (h, learning_method) in enumerate(zip(handles, learning_method_vec)):        # remove error bars in legend
        if isinstance(handles[i], collections.Container):
            handles[i] = handles[i][0]

    # plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))

    SHOW_STD = False


    legend = plt.legend(handles, label_vec,
                        loc='upper right',  # 'upper right'
                        handlelength=2,
                        fontsize=12,
                        labelspacing=0.2,  # distance between label entries
                        handletextpad=0.3,  # distance between label and the line representation
                        borderaxespad=0.2,  # distance between legend and the outer axes
                        borderpad=0.3,  # padding inside legend box
                        numpoints=1,  # put the marker only once
                        )
    if not(SHOW_STD):
        legend = plt.legend(handles, label_vec,
                        loc='upper right',  # 'upper right'
                        handlelength=2,
                        fontsize=10,
                        labelspacing=0.2,  # distance between label entries
                        handletextpad=0.3,  # distance between label and the line representation
                        borderaxespad=0.2,  # distance between legend and the outer axes
                        borderpad=0.3,  # padding inside legend box
                        numpoints=1,  # put the marker only once
                        scatterpoints=1  # display only one-scatter point in legend
                        )

    # # legend.set_zorder(1)
    frame = legend.get_frame()
    frame.set_linewidth(0.0)
    frame.set_alpha(0.9)  # 0.8


    # -- Figure settings and save
    plt.xscale('log')
    plt.xticks(xtick_lab, xtick_labels)
    plt.yticks(ytick_lab, ytick_lab)
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_ylim(bottom=ymin)

    grid(b=True, which='major', axis='both', alpha=0.2, linestyle='solid', linewidth=0.5)  # linestyle='dashed', which='minor', axis='y',
    grid(b=True, which='minor', axis='both', alpha=0.2, linestyle='solid', linewidth=0.5)  # linestyle='dashed', which='minor', axis='y',

    xlim(xmin, xmax)
    ylim(ymin, ymax)


    xlabel(r'Time Median (sec)', labelpad=0)      # labelpad=0
    ylabel(r'Accuracy', labelpad=0)
    if CREATE_PDF:
        savefig(join(figure_directory, fig_filename), format='pdf',
                dpi=None,
                edgecolor='w',
                orientation='portrait',
                transparent=False,
                bbox_inches='tight',
                pad_inches=0.05,
                frameon=None)

    if SHOW_PDF:
        showfig(join(figure_directory, fig_filename))

    if SHOW_PLOT:
        plt.show()




if __name__ == "__main__":
    run(6, show_plot=True)
