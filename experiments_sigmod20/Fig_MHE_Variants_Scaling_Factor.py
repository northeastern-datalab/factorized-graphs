"""
Variant of prior version that runs several methods on the identical data example
2. Fig 2: x-axis is weight scaling factor

Compares and stores a number of options at the same time (e.g., with or without EC, and various weight vectors).

First version: Nov 5, 2016
This version: Jun 26, 2017
Author: Wolfgang Gatterbauer
"""

# from __future__ import division             # allow integer division
# from __future__ import print_function
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
                              replace_fraction_of_rows,
                              showfig)
from graphGenerator import planted_distribution_model_H
from estimation import (estimateH,
                        transform_hToH)
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)      # show all columns
pd.set_option('display.max_rows', 500)

# -- Determine path to data *irrespective* of where the file is run from
from os.path import abspath, dirname, join
from inspect import getfile, currentframe
current_path = dirname(abspath(getfile(currentframe())))
figure_directory = join(current_path, 'figs')
data_directory = join(current_path, 'datacache')


def run(option, create_data=False, add_data=False, show_plot=False, create_pdf=False, show_pdf=False, show_fig=True):

    # -- Setup
    OPTION = option
    CREATE_DATA = create_data
    ADD_DATA = add_data
    SHOW_PLOT=show_plot
    CREATE_PDF=create_pdf
    SHOW_PDF=show_pdf
    SHOW_FIG2 = show_fig     # curve

    RANDOMSEED = None  # For repeatability
    random.seed(RANDOMSEED)  # seeds some other python random generator
    np.random.seed(seed=RANDOMSEED)  # seeds the actually used numpy random generator; both are used and thus needed

    header = ['currenttime',
              'option',     # one option corresponds to one choice of weight vector. In practice, one choice of scaling factor (for weight vector)
              'variant',    # 1, 2, 3 (against GT), and 1-2, 1-3, 2-3 (against each other)
              'length',
              'diff',
              'time']       # L2 norm between H and estimate


    # Default Graph parameters and options
    n = 10000
    d = 25
    h = 8
    distribution = 'powerlaw'
    randomize = False
    initial_h0 = None           # initial vector to start finding optimal H
    initial_H0 = None
    exponent = -0.3
    length = 5
    rep_differentGraphs = 1
    rep = 10       #
    EC = [False] + [True] * 35
    # scaling_vec = [1, 0.1, 0.14, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.4, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    scaling_vec = [1] + [round(0.1 * pow(10, 1/8)**x, 4) for x in range(33)]
    num_options = len(scaling_vec)
    scaling_vec = np.array(scaling_vec)
    # weight = np.array([np.power(scaling_vec, i) for i in range(5)])
    # weight = weight.transpose()
    # ymin1 = None
    # ymax1 = None
    xmin2 = 0.1
    xmax2 = 1000
    ymax2 = 1.
    ymin2 = 0
    stratified = False
    xtick_lab = [0.1, 1, 10, 100, 1000]
    # ytick_lab = [0.05, 0.1, 0.5, 1]
    # fig1_index = [0, 11, 16, 21, 23, 24, 25, 26]         # which index of scaling options to display if CHOICE_FIG_BAR_VARIANT==True
    smartInit = False
    smartInitRandomize = False
    delta = 0.1
    variant = 1           # for figure 2, to speed up calculations
    logarithm = False

    if OPTION == 1:
        CHOICE_vec = [18, 50, 51, 52, 53, 54]
        initial_H0_vec = [None] + [create_parameterized_H(3, h)] + [None] * 4
        randomize_vec = [False]*2 + [True]*4
        delta_vec = [None]*2 + [0.1, 0.2, 0.3] + [0.1]
        constraints_vec = [False]*5 + [True]

    # elif OPTION == 0:
    #     CHOICE_vec = [54]
    #     initial_H0_vec = [None]
    #     randomize_vec = [True]
    #     delta_vec = [0.1]
    #     constraints_vec = [True]
    #
    # elif OPTION == 2:
    #     f = 0.003
    #     CHOICE_vec = [101, 102, 103, 104, 105]
    #     initial_H0_vec = [None] + [create_parameterized_H(3, h)] + [None] * 3
    #     randomize_vec = [False]*2 + [True]*3
    #     delta_vec = [None]*2 + [0.1, 0.3] + [0.1]
    #     constraints_vec = [False]*4 + [True]
    #
    # elif OPTION == 3:
    #     f = 0.003
    #     h = 3
    #     CHOICE_vec = [111, 112, 113, 114, 115]
    #     initial_H0_vec = [None] + [create_parameterized_H(3, h)] + [None] * 3
    #     randomize_vec = [False]*2 + [True]*3
    #     delta_vec = [None]*2 + [0.1, 0.3] + [0.1]
    #     constraints_vec = [False]*4 + [True]

    # elif OPTION == 4:
    #     f = 0.001
    #     h = 8
    #     CHOICE_vec = [121, 122, 123, 124]
    #     initial_H0_vec = [None] + [create_parameterized_H(3, h)] + [None] * 2
    #     randomize_vec = [False]*2 + [True]*2
    #     delta_vec = [None]*2 + [0.1, 0.3]
    #     constraints_vec = [False]*4

    elif OPTION == 5:
        f = 0.001
        h = 8
        ymax2 = 2
        ymin2 = 4e-2
        CHOICE_vec = [131, 132, 133]
        initial_H0_vec = [None] + [create_parameterized_H(3, h)] + [None] * 1
        randomize_vec = [False]*2 + [True]*1
        delta_vec = [None]*2 + [0.1]
        constraints_vec = [False]*3
        stratified = True
        # CHOICE_vec = [131, 132, 133, 134]
        # initial_H0_vec = [None] + [create_parameterized_H(3, h)] + [None] * 2
        # randomize_vec = [False]*2 + [True]*2
        # delta_vec = [None]*2 + [0.1, 0.3]
        # constraints_vec = [False]*4
        # stratified = True


    # elif OPTION == 6:
    #     f = 0.003
    #     h = 8
    #     CHOICE_vec = [141, 142, 143]
    #     initial_H0_vec = [None] + [create_parameterized_H(3, h)] + [None] * 1
    #     randomize_vec = [False]*2 + [True]*1
    #     delta_vec = [None]*2 + [0.1]
    #     constraints_vec = [False]*3
    #     # CHOICE_vec = [141, 142, 143, 144]
    #     # initial_H0_vec = [None] + [create_parameterized_H(3, h)] + [None] * 2
    #     # randomize_vec = [False]*2 + [True]*2
    #     # delta_vec = [None]*2 + [0.1, 0.3]
    #     # constraints_vec = [False]*4

    elif OPTION == 7:
        f = 0.003
        h = 8
        CHOICE_vec = [151, 152, 153]
        initial_H0_vec = [None] + [create_parameterized_H(3, h)] + [None] * 1
        randomize_vec = [False]*2 + [True]*1
        delta_vec = [None]*2 + [0.1]
        constraints_vec = [False]*3
        stratified = True
        # CHOICE_vec = [151, 152, 153, 154]
        # initial_H0_vec = [None] + [create_parameterized_H(3, h)] + [None] * 2
        # randomize_vec = [False]*2 + [True]*2
        # delta_vec = [None]*2 + [0.1, 0.3]
        # constraints_vec = [False]*4
        # stratified = True

    # elif OPTION == 8:
    #     f = 0.001
    #     h = 3
    #     CHOICE_vec = [161, 162, 163]
    #     initial_H0_vec = [None] + [create_parameterized_H(3, h)] + [None] * 1
    #     randomize_vec = [False]*2 + [True]*1
    #     delta_vec = [None]*2 + [0.1]
    #     constraints_vec = [False]*3
    #     # CHOICE_vec = [161, 162, 163, 164]
    #     # initial_H0_vec = [None] + [create_parameterized_H(3, h)] + [None] * 2
    #     # randomize_vec = [False]*2 + [True]*2
    #     # delta_vec = [None]*2 + [0.1, 0.3]
    #     # constraints_vec = [False]*4

    elif OPTION == 9:
        f = 0.001
        h = 3
        CHOICE_vec = [171, 172, 173]
        initial_H0_vec = [None] + [create_parameterized_H(3, h)] + [None] * 1
        randomize_vec = [False]*2 + [True]*1
        delta_vec = [None]*2 + [0.1]
        constraints_vec = [False]*3
        stratified = True
        ymin2 = 6e-2
        ymax2 = 1
        # CHOICE_vec = [171, 172, 173, 174]
        # initial_H0_vec = [None] + [create_parameterized_H(3, h)] + [None] * 2
        # randomize_vec = [False]*2 + [True]*2
        # delta_vec = [None]*2 + [0.1, 0.3]
        # constraints_vec = [False]*4


    elif OPTION == 10:
        f = 0.001
        h = 3
        d = 10
        CHOICE_vec = [181, 182, 183]
        initial_H0_vec = [None] + [create_parameterized_H(3, h)] + [None] * 1
        randomize_vec = [False] * 2 + [True] * 1
        delta_vec = [None] * 2 + [0.1]
        constraints_vec = [False] * 3
        stratified = True


    elif OPTION == 11:
        f = 0.05
        h = 8
        d = 25
        ymax2 = 0.08
        CHOICE_vec = [191, 192, 193]
        initial_H0_vec = [None] + [create_parameterized_H(3, h)] + [None] * 1
        randomize_vec = [False] * 2 + [True] * 1
        delta_vec = [None] * 2 + [0.1]
        constraints_vec = [False] * 3
        stratified = True

    elif OPTION == 12:
        f = 0.05
        h = 3
        d = 25
        ymax2 = 0.08
        CHOICE_vec = [201, 202, 203]
        initial_H0_vec = [None] + [create_parameterized_H(3, h)] + [None] * 1
        randomize_vec = [False] * 2 + [True] * 1
        delta_vec = [None] * 2 + [0.1]
        constraints_vec = [False] * 3
        stratified = True

    elif OPTION == 13:
        n=1000
        f = 0.01
        h = 3
        CHOICE_vec = [211, 212, 213]
        initial_H0_vec = [None] + [create_parameterized_H(3, h)] + [None] * 1
        randomize_vec = [False]*2 + [True]*1
        delta_vec = [None]*2 + [0.1]
        constraints_vec = [False]*3
        stratified = True
        ymin2 = 6e-2
        ymax2 = 1


    elif OPTION == 15:
        n=100000
        f = 0.01
        h = 3
        CHOICE_vec = [221, 222, 223]
        initial_H0_vec = [None] + [create_parameterized_H(3, h)] + [None] * 1
        randomize_vec = [False]*2 + [True]*1
        delta_vec = [None]*2 + [0.1]
        constraints_vec = [False]*3
        stratified = True
        ymin2 = 5e-3
        ymax2 = 2e-1

    elif OPTION == 16:      # variant on 13 with logarithm
        n=1000
        f = 0.01
        h = 3
        CHOICE_vec = [231, 232, 233]
        initial_H0_vec = [None] + [create_parameterized_H(3, h)] + [None] * 1
        randomize_vec = [False]*2 + [True]*1
        delta_vec = [None]*2 + [0.1]
        constraints_vec = [False]*3
        stratified = True
        ymin2 = 6e-2
        ymax2 = 1
        logarithm = True

    elif OPTION == 17:      
        f = 0.001
        h = 8
        ymax2 = 2
        ymin2 = 4e-2
        CHOICE_vec = [133]
        initial_H0_vec = [None] * 1
        randomize_vec = [True]*1
        delta_vec = [0.1]
        constraints_vec = [False]*3
        stratified = True

    else:
        raise Warning("Incorrect choice!")

    k = 3
    a = 1
    alpha0 = np.array([a, 1., 1.])
    alpha0 = alpha0 / np.sum(alpha0)
    H0 = create_parameterized_H(k, h, symmetric=True)



    if CREATE_DATA:
        for CHOICE in CHOICE_vec:
            csv_filename = 'Fig_MHE_Variants_{}.csv'.format(CHOICE)
            save_csv_record(join(data_directory, csv_filename), header, append=False)

    # print("OPTION: {}".format(OPTION))



    # -- Create data
    if CREATE_DATA or ADD_DATA:
        for rs in range(1, rep_differentGraphs+1):
            # print('Graph {}'.format(rs))

            # -- Create graph
            W, Xd = planted_distribution_model_H(n, alpha=alpha0, H=H0, d_out=d,
                                                      distribution=distribution,
                                                      exponent=exponent,
                                                      directed=False,
                                                      debug=False)
            X0 = from_dictionary_beliefs(Xd)

            for r in range(1, rep + 1):
                # print('Repetition {}'.format(r))

                X1, ind = replace_fraction_of_rows(X0, 1 - f, stratified=stratified)

                for CHOICE, initial_H0, randomize, delta, constraints in zip(CHOICE_vec, initial_H0_vec, randomize_vec, delta_vec, constraints_vec):
                    csv_filename = 'Fig_MHE_Variants_{}.csv'.format(CHOICE)

                    # -- Create estimates and compare against GT, or against each other
                    for length in range(1, length + 1):
                        for option in range(num_options):
                            start = time.time()

                            if smartInit:
                                startWeight = 0.2
                                initial_H0 = estimateH(X1, W, method='DHE', variant=variant,
                                                       distance=5,
                                                       EC=EC[option], weights=startWeight,
                                                       randomize=smartInitRandomize,
                                                       logarithm=logarithm)

                            # print(option)
                            # print(scaling_vec)
                            # print(scaling_vec[option])


                            H_est = estimateH(X1, W, method='DHE', variant=variant,
                                           distance=length, EC=EC[option], weights=scaling_vec[option],
                                           randomize=randomize,
                                           initial_H0=initial_H0,
                                              constraints = constraints,
                                           delta = delta
                                                       )
                            time_est = time.time() - start
                            diff = LA.norm(H_est - H0)

                            # if np.amin(H_est) < 0:
                            # if True:
                            #     print("\nCHOICE: {}, weight: {}".format(CHOICE, scaling_vec[option]))
                            #     print("length:{}".format(length))
                            #     print("H_est:\n{}".format(H_est))
                            #     print("diff: {}".format(diff))

                            tuple = [str(datetime.datetime.now())]
                            text = [option, variant, length, diff, time_est]
                            # text = np.asarray(text)  # (without np, entries get ugly format) not used here because it transforms integers to float !!
                            tuple.extend(text)
                            save_csv_record(join(data_directory, csv_filename), tuple)




    if SHOW_FIG2:

        for CHOICE, initial_h0, randomize, delta in zip(CHOICE_vec, initial_H0_vec, randomize_vec, delta_vec):
            csv_filename = 'Fig_MHE_Variants_{}.csv'.format(CHOICE)


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
            # print("\n-- df2 (length {}):\n{}".format(len(df2.index), df2.head(30)))

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
            plt.yscale('log')

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
            plt.xlabel(r'Scaling factor $(\lambda)$', labelpad=0)
            plt.ylabel(r'L2 norm', labelpad=0)

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

            plt.xticks(xtick_lab)
            # plt.yticks(ytick_lab, ytick_lab)

            if SHOW_PLOT:
                plt.show()
            if CREATE_PDF:
                plt.savefig(join(figure_directory, fig_filename), format='pdf',
                    dpi=None,
                    edgecolor='w',
                    orientation='portrait',
                    transparent=False,
                    bbox_inches='tight',
                    pad_inches=0.05,
                    frameon=None)
            if SHOW_PDF:
                showfig(join(figure_directory, fig_filename))


if __name__ == "__main__":
    run(15, create_pdf=True, show_pdf=True)
