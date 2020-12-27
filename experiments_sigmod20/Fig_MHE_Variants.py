"""
Creates two versions of small figures:
  1. Fig 1: accuracy of various H_est methods (1-3) and length (1-5) by calculating L2 norm against GT as bar diagram
     Also compares the relative estimates against each other (e.g., 1-2 compares L2 norm between estimate by variant 1 and variant 2)
  2. Fig 2: x-axis is weight scaling factor

Compares and stores a number of options at the same time (e.g., with or without EC, and various weight vectors).

First version: Nov 5, 2016
This version: Nov 8, 2016
Author: Wolfgang Gatterbauer
"""

import numpy as np
from numpy import linalg as LA
import time
import datetime
import random
import os                       # for displaying created PDF
import sys
sys.path.append('./../sslh')
from fileInteraction import save_csv_record
from utils import (from_dictionary_beliefs,
                              create_parameterized_H,
                              replace_fraction_of_rows)
from graphGenerator import planted_distribution_model_H
from estimation import (estimateH,
                        transform_hToH)
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)      # show all columns
pd.set_option('display.max_rows', 500)
open_cmd = {'linux' : 'xdg-open', 'linux2' : 'xdg-open', 'darwin' : 'open', 'win32' : 'start'}

# -- Determine path to data *irrespective* of where the file is run from
from os.path import abspath, dirname, join
from inspect import getfile, currentframe
current_path = dirname(abspath(getfile(currentframe())))
figure_directory = join(current_path, 'figs')
data_directory = join(current_path, 'datacache')


def run(choice, create_data=False, add_data=False, show_plot=False, create_pdf=False, show_pdf=False, show_fig=True):
    # -- Setup
    CHOICE = choice
    CREATE_DATA = create_data
    ADD_DATA = add_data
    CREATE_PDF = create_pdf
    SHOW_PDF=show_pdf
    SHOW_FIG1 = show_fig        # bar diagram
    SHOW_FIG2 = False      # curve

    csv_filename = 'Fig_MHE_Variants_{}.csv'.format(CHOICE)
    header = ['currenttime',
              'option',     # one option corresponds to one choice of weight vector. In practice, one choice of scaling factor (for weight vector)
              'variant',    # 1, 2, 3 (against GT), and 1-2, 1-3, 2-3 (against each other)
              'length',
              'diff',
              'time']       # L2 norm between H and estimate
    if CREATE_DATA:
        save_csv_record(join(data_directory, csv_filename), header, append=False)


    # Default Graph parameters and options
    n = 10000
    f = 0.1
    h = 8
    distribution = 'uniform'
    randomize = False
    initial_h0 = None           # initial vector to start finding optimal H
    initial_H0 = None
    exponent = -0.3
    length = 5
    rep = 10       #
    EC = [False] + [True] * 31
    scaling_vec = [1, 0.1, 0.14, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.4, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    num_options = len(scaling_vec)
    scaling_vec = np.array(scaling_vec)
    weight = np.array([np.power(scaling_vec, i) for i in range(5)])
    weight = weight.transpose()
    ymin1 = None
    ymax1 = None
    xmin2 = None
    xmax2 = None
    ymin2 = None
    ymax2 = None
    # fig1_index = [0, 11, 16, 21, 23, 24, 25, 26]         # which index of scaling options to display if CHOICE_FIG_BAR_VARIANT==True
    fig1_index = [21]
    smartInit = False
    smartInitRandomize = False
    delta = 0.1
    variant_vec = [1,2, 3]      # for figure 1
    variant_vec = [1]           # for figure 2, to speed up calculations

    if CHOICE == 1:     # ok
        n = 1000
        d = 10
        ymax2 = 0.24

    elif CHOICE == 2:   # ok
        n = 1000
        d = 10
        distribution = 'powerlaw'
        ymax2 = 0.24

    elif CHOICE == 3:   # ok
        n = 1000
        d = 5
        distribution = 'powerlaw'
        ymax2 = 0.4

    elif CHOICE == 4:   # ok
        n = 1000
        d = 25
        distribution = 'powerlaw'
        ymax2 = 0.16


    elif CHOICE == 10:  # ok
        d = 10
        ymax2 = 0.1

    elif CHOICE == 11:  # (selection)
        d = 10
        distribution = 'powerlaw'
        exponent = -0.5
        ymax2 = 0.1
        ymax1 = 0.14

    elif CHOICE == 12:
        d = 3
        ymax2 = 0.19
        ymax1 = 0.2

    elif CHOICE == 13:
        d = 25
        ymax2 = 0.05

    elif CHOICE == 14:  # selection (for comparison)
        d = 25
        distribution = 'powerlaw'
        ymax2 = 0.046
        ymax1 = 0.08

    elif CHOICE == 15:   # selection
        d = 25
        f = 0.05
        distribution = 'powerlaw'
        ymax1 = 0.15
        ymax2 = 0.095



    elif CHOICE == 16:   # selection TODO !!!
        d = 25
        f = 0.01
        distribution = 'powerlaw'
        ymax1 = 0.15
        ymax2 = 0.4
        ymin2 = 0

    elif CHOICE == 17:   # selection TODO !!!
        d = 25
        f = 0.003
        distribution = 'powerlaw'
        ymax1 = 0.15
        ymax2 = 1.
        ymin2 = 0

    elif CHOICE == 18:   # selection TODO !!!
        d = 25
        f = 0.001
        distribution = 'powerlaw'
        ymax1 = 0.15
        ymax2 = 1.
        ymin2 = 0



    elif CHOICE == 20:
        h = 3
        d = 10
        ymax1 = 0.12

    elif CHOICE == 21:      # selection (for comparison against f=0.05: 26)
        h = 3
        d = 10
        distribution = 'powerlaw'
        exponent = -0.5
        ymax1 = 0.15
        ymax2 = 0.099

    elif CHOICE == 22:      # selection (for comparison with start from GT: 44)
        h = 3
        d = 3
        ymax1 = 0.25
        ymax2 = 0.39

    elif CHOICE == 23:      # ok
        h = 3
        d = 25
        ymax1 = 0.1
        ymax2 = 0.12

    elif CHOICE == 24:
        h = 3
        d = 25
        distribution = 'powerlaw'
        ymax1 = 0.08

    elif CHOICE == 25:      # main selection
        h = 3
        d = 25
        f = 0.05
        distribution = 'powerlaw'
        ymax1 = 0.15
        ymax2 = 0.125

    elif CHOICE == 26:      # selection, #=200
        h = 3
        d = 10
        f = 0.05
        distribution = 'powerlaw'
        ymax1 = 0.21
        ymax2 = 0.26

    elif CHOICE == 27:      # selection, #=200
        d = 10
        f = 0.05
        distribution = 'powerlaw'
        ymax1 = 0.15
        ymax2 = 0.21

    elif CHOICE == 60:   # ??? #=50 !!!, 50 more
        d = 25
        f = 0.01
        distribution = 'powerlaw'
        ymax1 = 0.15
        ymax2 = 0.6

    elif CHOICE == 61:   # ??? #=50, 100 more
        d = 25
        f = 0.005
        distribution = 'powerlaw'
        ymax1 = 0.99
        ymax2 = 0.99

    elif CHOICE == 62:   # ??? #=50, 150 more
        h = 3
        d = 25
        f = 0.01
        distribution = 'powerlaw'
        ymax1 = 0.6
        ymax2 = 0.6

    elif CHOICE == 63:   # ??? #=50, 150 more
        h = 3
        d = 25
        f = 0.005
        distribution = 'powerlaw'
        ymax1 = 1.2
        ymax2 = 1.0




    # --- Randomization ---
    # randomized 22
    elif CHOICE == 32:
        randomize = True
        h = 3
        d = 3
        ymax1 = 0.25
        ymax2 = 0.4


    # --- GT ---
    # version of 22 where GT is supplied to start optimiziation
    # just to check if the global optimum of the energy function actually corresponds to the GT
    elif CHOICE == 42:      # selection, #=200 (for comparison with start from GT)
        initial_h0 = [0.2, 0.6, 0.2]        # start optimization at optimal point
        h = 3
        d = 3
        ymax1 = 0.25
        ymax2 = 0.39

    # version of 15 where GT is supplied to start optimiziation
    elif CHOICE == 43:      # selection, #=200
        initial_h0 = [0.1, 0.8, 0.1]  # start optimization at optimal point
        h = 8
        d = 25
        f = 0.05
        distribution = 'powerlaw'
        ymax1 = 0.15
        ymax2 = 0.095

    # version of 12 where GT is supplied to start optimiziation
    elif CHOICE == 44:      # selection, #=200 (for comparison)
        initial_h0 = [0.1, 0.8, 0.1]  # start optimization at optimal point
        h = 8
        d = 3
        ymax1 = 0.2
        ymax2 = 0.19

    # version of 25 where GT is supplied to start optimiziation
    elif CHOICE == 45:      # selection, #=200
        h = 3
        d = 25
        f = 0.05
        distribution = 'powerlaw'
        ymax1 = 0.15
        ymax2 = 0.125


    # version of 12 where GT is supplied to start optimiziation
    elif CHOICE == 46:   # selection TODO !!!
        initial_h0 = [0.1, 0.8, 0.1]  # start optimization at optimal point
        d = 25
        f = 0.01
        distribution = 'powerlaw'
        ymax1 = 0.15
        ymax2 = 0.4
        ymin2 = 0.0

    # version of 12 where GT is supplied to start optimiziation
    elif CHOICE == 47:   # selection TODO !!!
        initial_h0 = [0.1, 0.8, 0.1]  # start optimization at optimal point
        d = 25
        f = 0.003
        distribution = 'powerlaw'
        ymax1 = 0.15
        ymax2 = 1.
        ymin2 = 0.0


    # version of 12 with smart init
    elif CHOICE == 48:   # selection TODO !!!
        d = 25
        f = 0.003
        distribution = 'powerlaw'
        ymax1 = 0.15
        ymax2 = 1.
        ymin2 = 0.0
        smartInit = True


    # version of 12 with smart init
    elif CHOICE == 49:   # selection TODO !!!
        d = 25
        f = 0.003
        distribution = 'powerlaw'
        ymax1 = 0.15
        ymax2 = 1.
        ymin2 = 0.0
        smartInit = True
        smartInitRandomize = True       # initialize optimization at several random points for smart init only


    elif CHOICE == 50:   # selection TODO !!!
        initial_h0 = [0.1, 0.8, 0.1]  # start optimization at optimal point
        d = 25
        f = 0.001
        distribution = 'powerlaw'
        ymax1 = 0.15
        ymax2 = 1.
        ymin2 = 0.0

    elif CHOICE == 51:   # selection TODO !!!
        d = 25
        f = 0.001
        distribution = 'powerlaw'
        ymax1 = 0.15
        ymax2 = 1.
        ymin2 = 0.0
        randomize = True        # start optimization at several random points
        delta = 0.1

    elif CHOICE == 52:   # selection TODO !!!
        d = 25
        f = 0.001
        distribution = 'powerlaw'
        ymax1 = 0.15
        ymax2 = 1.
        ymin2 = 0.0
        randomize = True        # start optimization at several random points
        delta = 0.2

    elif CHOICE == 53:   # selection TODO !!!
        d = 25
        f = 0.001
        distribution = 'powerlaw'
        ymax1 = 0.15
        ymax2 = 1.
        ymin2 = 0.0
        randomize = True        # start optimization at several random points
        delta = 0.3




    else:
        raise Warning("Incorrect choice!")


    k = 3
    a = 1
    alpha0 = np.array([a, 1., 1.])
    alpha0 = alpha0 / np.sum(alpha0)
    H0 = create_parameterized_H(k, h, symmetric=True)
    RANDOMSEED = None  # For repeatability
    random.seed(RANDOMSEED)  # seeds some other python random generator
    np.random.seed(seed=RANDOMSEED)  # seeds the actually used numpy random generator; both are used and thus needed
    # print("CHOICE: {}".format(CHOICE))


    # -- Create data
    if CREATE_DATA or ADD_DATA:
        for r in range(1, rep+1):
            # print('Repetition {}'.format(r))

            # -- Create graph
            W, Xd = planted_distribution_model_H(n, alpha=alpha0, H=H0, d_out=d,
                                                      distribution=distribution,
                                                      exponent=exponent,
                                                      directed=False,
                                                      debug=False)
            X0 = from_dictionary_beliefs(Xd)
            X1, ind = replace_fraction_of_rows(X0, 1 - f)

            # -- Create estimates and compare against GT, or against each other
            H_est={}
            for length in range(1, length + 1):
                for option in range(num_options):
                    for variant in variant_vec:
                        start = time.time()

                        if smartInit:
                            startWeight = 0.2
                            initial_H0 = estimateH(X1, W, method='DHE', variant=variant,
                                                   distance=5,
                                                   EC=EC[option], weights=startWeight,
                                                   randomize=smartInitRandomize)

                        H_est[variant] = estimateH(X1, W, method='DHE', variant=variant,
                                                   distance=length, EC=EC[option], weights=weight[option],
                                                   randomize=randomize,
                                                   initial_h0=initial_h0,
                                                   initial_H0=initial_H0,
                                                   delta = delta
                                                   )
                        time_est = time.time() - start
                        diff = LA.norm(H_est[variant] - H0)

                        tuple = [str(datetime.datetime.now())]
                        text = [option, variant, length, diff, time_est]
                        # text = np.asarray(text)  # (without np, entries get ugly format) not used here because it transforms integers to float !!
                        tuple.extend(text)
                        save_csv_record(join(data_directory, csv_filename), tuple)

                    # -- Compare against each other
                    for variant1 in variant_vec:
                        for variant2 in variant_vec:
                            if variant1 < variant2:
                                diff = LA.norm(H_est[variant1] - H_est[variant2])

                                tuple = [str(datetime.datetime.now())]
                                text = [option, "{}-{}".format(variant1, variant2), length, diff, time_est]
                                tuple.extend(text)
                                save_csv_record(join(data_directory, csv_filename), tuple)

    if SHOW_FIG1:
        # -- Read, aggregate, and pivot data for all options
        df1 = pd.read_csv(join(data_directory, csv_filename))
        # print("\n-- df1 (length {}):\n{}".format(len(df1.index), df1.head(15)))
        df2 = df1.groupby(['option', 'variant', 'length']).agg \
            ({'diff': [np.mean, np.std, np.size],  # Multiple Aggregates
              'time': [np.mean, np.std],
              })
        df2.columns = ['_'.join(col).strip() for col in df2.columns.values]  # flatten the column hierarchy
        df2.reset_index(inplace=True)  # remove the index hierarchy
        df2.rename(columns={'diff_size': 'count'}, inplace=True)
        # print("\n-- df2 (length {}):\n{}".format(len(df2.index), df2.head(90)))


        # -- Create one separate figure for each option
        for option in range(num_options):
            if option not in fig1_index:
                continue
            scaling = scaling_vec[option]

            fig_filename = 'Fig_MHE_Variants_{}_{}.pdf'.format(CHOICE, option)

            df3 = df2.query('option==@option')  # Query
            df3 = pd.pivot_table(df3, index=['length'], columns=['variant'], values=['diff_mean', 'diff_std'])  # Pivot
            # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(30)))

            df3.columns = ['_'.join(col).strip() for col in df3.columns.values]     # flatten the column hierarchy
            df3.reset_index(level=0, inplace=True)                                  # get length into columns
            # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(30)))
            df3.drop(['diff_std_1-2', 'diff_std_1-3', 'diff_std_2-3', ], axis=1, inplace=True)
            # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(30)))


            # -- Setup figure
            mpl.rcParams['backend'] = 'pdf'
            mpl.rcParams['lines.linewidth'] = 3
            mpl.rcParams['font.size'] = 16
            mpl.rcParams['axes.labelsize'] = 20
            mpl.rcParams['axes.titlesize'] = 16
            mpl.rcParams['xtick.labelsize'] = 16
            mpl.rcParams['ytick.labelsize'] = 16
            mpl.rcParams['legend.fontsize'] = 14
            mpl.rcParams['axes.edgecolor'] = '111111'   # axes edge color
            mpl.rcParams['grid.color'] = '777777'   # grid color
            mpl.rcParams['figure.figsize'] = [4, 4]
            mpl.rcParams['xtick.major.pad'] = 4     # padding of tick labels: default = 4
            mpl.rcParams['ytick.major.pad'] = 4         # padding of tick labels: default = 4
            fig = plt.figure()
            ax = fig.add_axes([0.13, 0.17, 0.8, 0.8])


            # -- Extract values into columns (plotting dataframew with bars plus error lines and lines gave troubles)
            l_vec = df3['length'].values                   # .tolist() does not work with bar plot, requires np.array
            diff_mean_1 = df3['diff_mean_1'].values
            diff_mean_2 = df3['diff_mean_2'].values
            diff_mean_3 = df3['diff_mean_3'].values
            diff_std_1 = df3['diff_std_1'].values
            diff_std_3 = df3['diff_std_2'].values
            diff_std_2 = df3['diff_std_3'].values


            # -- Draw the bar plots
            width = 0.2       # the width of the bars
            bar1 = ax.bar(l_vec-1.5*width, diff_mean_1, width, color='blue',
                          yerr=diff_std_1, error_kw={'ecolor':'black', 'linewidth':2},    # error-bars colour
                          label=r'1')
            bar2 = ax.bar(l_vec-0.5*width, diff_mean_2, width, color='darkorange',
                          yerr=diff_std_2, error_kw={'ecolor':'black', 'linewidth':2},  # error-bars colour
                          label=r'2')
            bar3 = ax.bar(l_vec+0.5*width, diff_mean_3, width, color='green',
                          yerr=diff_std_1, error_kw={'ecolor':'black', 'linewidth':2},    # error-bars colour
                          label=r'3')

            if CHOICE == 15 and option == 0:
                ax.annotate(np.round(diff_mean_1[1], 2), xy=(1.6, 0.15), xytext=(0.8, 0.122),
                            arrowprops=dict(facecolor='black', arrowstyle="->"), )


            # -- Legend
            handles, labels = ax.get_legend_handles_labels()
            # print("labels: {}".format(labels))
            legend = plt.legend(handles, labels,
                                loc='upper right',
                                handlelength=2,
                                labelspacing=0,             # distance between label entries
                                handletextpad=0.3,          # distance between label and the line representation
                                title='Variants',
                                borderaxespad=0.3,        # distance between legend and the outer axes
                                borderpad=0.1,                # padding inside legend box
                                )
            frame = legend.get_frame()
            frame.set_linewidth(0.0)
            frame.set_alpha(0.8)        # 0.8


            # -- Title and figure settings
            if distribution == 'uniform':
                distribution_label = ',$uniform'
            else:
                distribution_label = '$'
            plt.title(r'$\!\!\!\!n\!=\!{}\mathrm{{k}}, d\!=\!{}, h\!=\!{}, f\!=\!{}{}'.format(int(n / 1000), d, h, f, distribution_label))
            # ax.set_xticks(range(10))
            plt.grid(b=True, which='both', alpha=0.2, linestyle='solid', axis='y', linewidth=0.5)       # linestyle='dashed', which='minor'
            plt.xlabel(r'Max path length ($\ell_{{\mathrm{{max}}}})$', labelpad=0)
            plt.ylabel(r'L2 norm', labelpad=0)

            if ymin1 is None:
                ymin1 = plt.ylim()[0]
                ymin1 = max(ymin1, 0)
            if ymax1 is None:
                ymax1 = plt.ylim()[1]
            plt.ylim(ymin1, ymax1)
            plt.xlim(0.5,5.5)
            plt.xticks([1, 2, 3, 4, 5])
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                # labelbottom='off',    # labels along the bottom edge are off
            )
            plt.annotate(r'$\lambda={:g}$'.format(float(scaling)), xycoords = 'axes fraction', xy=(0.5, 0.9), ha="center", va="center")

            if CREATE_PDF:
                plt.savefig(join(figure_directory, fig_filename), format='pdf',
                        dpi=None,
                        edgecolor='w',
                        orientation='portrait',
                        transparent=False,
                        bbox_inches='tight',
                        pad_inches=0.05,
                        frameon=None)
            if SHOW_FIG1:
                plt.show()
            if SHOW_PDF:
                os.system('{} "'.format(open_cmd[sys.platform]) + join(figure_directory, fig_filename) + '"')       # shows actually created PDF

    if SHOW_FIG2:
        # -- Read, aggregate, and pivot data for all options
        df1 = pd.read_csv(join(data_directory, csv_filename))
        # print("\n-- df1 (length {}):\n{}".format(len(df1.index), df1.head(15)))
        df2 = df1.groupby(['option', 'variant', 'length']).agg \
            ({'diff': [np.mean, np.std, np.size],  # Multiple Aggregates
              'time': [np.mean, np.std],
              })
        df2.columns = ['_'.join(col).strip() for col in df2.columns.values]  # flatten the column hierarchy
        df2.reset_index(inplace=True)  # remove the index hierarchy
        df2.rename(columns={'diff_size': 'count'}, inplace=True)
        # print("\n-- df2 (length {}):\n{}".format(len(df2.index), df2.head(90)))

        df2['length'] = df2['length'].astype(str)               # transform numbers into string for later join: '.join(col).strip()'
        df3 = df2.query('variant=="1"')  # We only focus on variant 1 (as close to row stochastic matrix as possible)
        # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(n=20)))

        df4 = pd.pivot_table(df3, index=['option'], columns=['length'], values=['diff_mean', 'diff_std'])  # Pivot
        # print("\n-- df4 (length {}):\n{}".format(len(df4.index), df4.head(30)))
        df4.columns = ['_'.join(col).strip() for col in df4.columns.values]     # flatten the column hierarchy, requires to have only strings
        df4.reset_index(level=0, inplace=True)  # get length into columns
        # print("\n-- df4 (length {}):\n{}".format(len(df4.index), df4.head(30)))

        # Add scaling factor for each row
        option = df4['option'].values       # extract the values from dataframe
        scaling = scaling_vec[option]       # look up the scaling factor in original list
        scaling = pd.Series(scaling)
        # print("scaling:\n{}".format(scaling))
        df5 = df4.assign(scaling=scaling.values)
        # print("\n-- df5 (length {}):\n{}".format(len(df5.index), df5.head(30)))

        # Filter rows
        select_rows = [i for i in range(num_options) if EC[i]]      # only those values for EC being tru
        df6 = df5[df5['option'].isin(select_rows)]
        # print("\n-- df6 (length {}):\n{}".format(len(df6.index), df6.head(30)))



        fig_filename = 'Fig_MHE_ScalingFactor_{}.pdf'.format(CHOICE)

        # -- Setup figure
        mpl.rcParams['backend'] = 'pdf'
        mpl.rcParams['lines.linewidth'] = 3
        mpl.rcParams['font.size'] = 14
        mpl.rcParams['axes.labelsize'] = 20
        mpl.rcParams['axes.titlesize'] = 16
        mpl.rcParams['xtick.labelsize'] = 16
        mpl.rcParams['ytick.labelsize'] = 16
        mpl.rcParams['legend.fontsize'] = 14
        mpl.rcParams['axes.edgecolor'] = '111111'  # axes edge color
        mpl.rcParams['grid.color'] = '777777'  # grid color
        mpl.rcParams['figure.figsize'] = [4, 4]
        mpl.rcParams['xtick.major.pad'] = 4  # padding of tick labels: default = 4
        mpl.rcParams['ytick.major.pad'] = 4  # padding of tick labels: default = 4
        fig = plt.figure()
        ax = fig.add_axes([0.13, 0.17, 0.8, 0.8])

        # -- Extract values into columns (plotting dataframew with bars plus error lines and lines gave troubles)
        scaling = df6['scaling'].values  # .tolist() does not work with bar plot, requires np.array
        diff_mean_1 = df6['diff_mean_1'].values
        diff_mean_2 = df6['diff_mean_2'].values
        diff_mean_3 = df6['diff_mean_3'].values
        diff_mean_4 = df6['diff_mean_4'].values
        diff_mean_5 = df6['diff_mean_5'].values
        diff_std_5 = df6['diff_std_5'].values

        # -- Draw the plots
        p1 = ax.plot(scaling, diff_mean_1, color='black', linewidth=1, linestyle='--', label=r'$\ell_\mathrm{max} = 1$')
        p2 = ax.plot(scaling, diff_mean_2, color='orange', label=r'$\ell_\mathrm{max} = 2$')
        p3 = ax.plot(scaling, diff_mean_3, color='blue', label=r'$\ell_\mathrm{max} = 3$')
        p4 = ax.plot(scaling, diff_mean_4, color='green', label=r'$\ell_\mathrm{max} = 4$')
        p5 = ax.plot(scaling, diff_mean_5, color='red', marker='o', label=r'$\ell_\mathrm{max} = 5$')
        plt.xscale('log')

        upper = diff_mean_5 + diff_std_5
        lower = diff_mean_5 - diff_std_5
        ax.fill_between(scaling, upper, lower, facecolor='red', alpha=0.2, edgecolor='none')


        # -- Title and legend
        if distribution == 'uniform':
            distribution_label = ',$uniform'
        else:
            distribution_label = '$'
        plt.title(r'$\!\!\!\!n\!=\!{}\mathrm{{k}}, d\!=\!{}, h\!=\!{}, f\!=\!{}{}'.format(int(n / 1000), d, h, f, distribution_label))
        handles, labels = ax.get_legend_handles_labels()
        # print("labels: {}".format(labels))
        legend = plt.legend(handles, labels,
                            loc='upper center',     # 'upper right'
                            handlelength=2,
                            labelspacing=0,  # distance between label entries
                            handletextpad=0.3,  # distance between label and the line representation
                            # title='Variants',
                            borderaxespad=0.3,  # distance between legend and the outer axes
                            borderpad=0.1,  # padding inside legend box
                            )
        frame = legend.get_frame()
        frame.set_linewidth(0.0)
        frame.set_alpha(0.9)  # 0.8

        # -- Figure settings
        # ax.set_xticks(range(10))
        plt.grid(b=True, which='both', alpha=0.2, linestyle='solid', linewidth=0.5)  # linestyle='dashed', which='minor', axis='y',
        plt.xlabel(r'$\lambda$', labelpad=0)
        plt.ylabel(r'L$^2$ norm', labelpad=0)

        if xmin2 is None:
            xmin2 = plt.xlim()[0]
        if xmax2 is None:
            xmax2 = plt.xlim()[1]
        if ymin2 is None:
            ymin2 = plt.ylim()[0]
            ymin2 = max(ymin2, 0)
        if ymax2 is None:
            ymax2 = plt.ylim()[1]
        plt.xlim(xmin2, xmax2)
        plt.ylim(ymin2, ymax2)
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            # bottom='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            right='off',  # ticks along the top edge are off
            # labelbottom='off',    # labels along the bottom edge are off
        )

        if CREATE_PDF:
            plt.savefig(join(figure_directory, fig_filename), format='pdf',
                    dpi=None,
                    edgecolor='w',
                    orientation='portrait',
                    transparent=False,
                    bbox_inches='tight',
                    pad_inches=0.05,
                    frameon=None)
        if SHOW_FIG2:
            plt.show()
        if SHOW_PDF:
            os.system('{} "'.format(open_cmd[sys.platform]) + join(figure_directory, fig_filename) + '"')  # shows actually created PDF


if __name__ == "__main__":
    run(15, create_pdf=True)