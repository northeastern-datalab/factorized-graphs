"""
Similar to '161108 - Fig_MHE_Optimal_ScalingFactor_d', just changing f instead of d
1: diff
2: lambda

First version: Nov 8, 2016
This version: Nov 9, 2016
Author: Wolfgang Gatterbauer <gatt@cmu.edu>
"""

import numpy as np
from numpy import linalg as LA
import datetime
import random
import os                       # for displaying created PDF
from fileInteraction import save_csv_record
from utils import (from_dictionary_beliefs,
                        create_parameterized_H,
                        replace_fraction_of_rows,
                        showfig)
from graphGenerator import planted_distribution_model_H
from estimation import estimateH
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)      # show all columns
pd.options.mode.chained_assignment = None       # default='warn'


# -- Determine path to data *irrespective* of where the file is run from
from os.path import abspath, dirname, join
from inspect import getfile, currentframe
current_path = dirname(abspath(getfile(currentframe())))
figure_directory = join(current_path, 'figs')
data_directory = join(current_path, 'datacache')


def run(choice, create_data=False, add_data=False, create_fig=True, show_plot=False, create_pdf=False, show_pdf=False, shorten_length=False, show_arrows=True):
    # -- Setup
    CHOICE = choice
    CREATE_DATA = create_data
    ADD_DATA = add_data
    SHOW_PLOT = show_plot
    SHOW_PDF = show_pdf
    CREATE_PDF = create_pdf

    csv_filename = 'Fig_MHE_Optimal_ScalingFactor_f_{}.csv'.format(CHOICE)
    header = ['currenttime',
              'option',     # one option corresponds to one choice of weight vector. In practice, one choice of scaling factor (for weight vector)
              'f',
              'scaling',
              'diff']       # L2 norm between H and estimate
    if CREATE_DATA:
        save_csv_record(join(data_directory, csv_filename), header, append=False)


    # -- Default Graph parameters
    randomize = False
    initial_h0 = None           # initial vector to start finding optimal H
    distribution = 'powerlaw'
    exponent = -0.3
    length = 5
    variant = 1
    rep = 50
    EC = True
    scaling_vec = [0] + [0.1 * pow(10, 1/8)**x for x in range(33)]
    num_options = len(scaling_vec)
    scaling_vec = np.array(scaling_vec)
    weight = np.array([np.power(scaling_vec, i) for i in range(5)])
    weight = weight.transpose()
    f_vec = [0.9 * pow(0.1, 1/12)**x for x in range(25)]
    # print(f_vec)
    fraction_of_minimum = 1.1           # scaling parameters that lead to optimum except for this scaling factor are included
    ymin2 = 0.28
    ymax2 = 500
    xmin = 0.01
    xmax = None
    xtick_lab = [0.01, 0.03, 0.1, 0.3, 1]
    # ytick_lab1 = np.arange(0, 1, 0.1)
    ytick_lab2 = [0.3, 1, 10, 100, 1000]
    ymax1 = 1.2
    ymin1 = 0.001
    # ytick_lab1 = [0.001, 0.01, 0.1, 1]
    k = 3
    a = 1


    # -- Options
    if CHOICE == 1:         # #=50
        n = 1000
        h = 8
        d = 10
        distribution = 'uniform'
        ymin1 = 0.005

    elif CHOICE == 2:       # selection, #=100
        n = 10000
        h = 8
        d = 10

    elif CHOICE == 3:       # selection, #=100
        n = 10000
        h = 8
        d = 25

    elif CHOICE == 4:       # selection, #=150
        n = 10000
        h = 3
        d = 10

    elif CHOICE == 5:       # selection, #=100
        n = 10000
        h = 3
        d = 25

    elif CHOICE == 6:       # #=50
        n = 10000
        h = 8
        d = 25
        distribution = 'uniform'

    elif CHOICE == 7:       # #=50
        n = 10000
        h = 3
        d = 25
        distribution = 'uniform'

    elif CHOICE == 8:       # #=55
        n = 10000
        h = 8
        d = 3

    elif CHOICE == 9:       # selection, #=100
        n = 10000
        h = 8
        d = 25
        rep = 5

    else:
        raise Warning("Incorrect choice!")

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
            for f in f_vec:
                # print('f: {:1.4f}'.format(f))

                # -- Create graph
                W, Xd = planted_distribution_model_H(n, alpha=alpha0, H=H0, d_out=d,
                                                          distribution=distribution,
                                                          exponent=exponent,
                                                          directed=False,
                                                          debug=False)
                X0 = from_dictionary_beliefs(Xd)
                X1, ind = replace_fraction_of_rows(X0, 1 - f)

                # -- Create estimates and compare against GT
                for option in range(num_options):
                    H_est = estimateH(X1, W, method='MHE', variant=variant, distance=length, EC=EC, weights=weight[option], randomize=randomize, initial_h0=initial_h0)
                    diff = LA.norm(H_est - H0)

                    tuple = [str(datetime.datetime.now())]
                    text = [option, f, scaling_vec[option], diff]
                    tuple.extend(text)
                    save_csv_record(join(data_directory, csv_filename), tuple)


    # -- Read, aggregate, and pivot data for all options
    df1 = pd.read_csv(join(data_directory, csv_filename))
    # print("\n-- df1: (length {}):\n{}".format(len(df1.index), df1.head(15)))

    # Aggregate repetitions
    df2 = df1.groupby(['f', 'scaling']).agg \
        ({'diff': [np.mean, np.std, np.size],  # Multiple Aggregates
          })
    df2.columns = ['_'.join(col).strip() for col in df2.columns.values]  # flatten the column hierarchy
    df2.reset_index(inplace=True)  # remove the index hierarchy
    df2.rename(columns={'diff_size': 'count'}, inplace=True)
    # print("\n-- df2 (length {}):\n{}".format(len(df2.index), df2.head(15)))

    # find minimum diff for each f, then join it back into df2
    df3 = df2.groupby(['f']).agg \
        ({'diff_mean': [np.min],  # Multiple Aggregates
          })
    df3.columns = ['_'.join(col).strip() for col in df3.columns.values]  # flatten the column hierarchy
    # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(90)))
    df4= pd.merge(df2, df3, left_on='f', right_index=True)      # ! join df2 and df3 on column "d" from df2, and index (=d) from df3
    # df4 = df4.drop(['index'], axis=1)     # does not work
    # print("\n-- df4 (length {}):\n{}".format(len(df4.index), df4.head(25)))

    # Select columns for energy comparison plot: H0
    df5 = df4.query('scaling==0')
    # print("\n-- df5 (length {}):\n{}".format(len(df5.index), df5.head(90)))
    # df5.drop('option', axis=1, inplace=True)  # gives warning
    df5 = df5.drop(['diff_mean_amin'], axis=1)
    # print("\n-- df5: scaling==0 (length {}):\n{}".format(len(df5.index), df5.head(90)))
    X_f = df5['f'].values                     # plot value
    Y_diff0 = df5['diff_mean'].values         # plot value
    Y_diff0_std = df5['diff_std'].values      # plot value

    # Select columns for energy comparison plot: H5 optimal
    df6 = df4.copy()
    # print("\n-- df6 (length {}):\n{}".format(len(df6.index), df6.head(90)))
    df6['cond'] = np.where((df6['diff_mean'] == df6['diff_mean_amin']), True, False)
    df6 = df6.query('cond==True')
    df6.drop(['cond', ], axis=1, inplace=True)
    # print("\n-- df6: best scaling (length {}):\n{}".format(len(df6.index), df6.head(90)))
    Y_diff1 = df6['diff_mean'].values         # plot value
    Y_diff1_std = df6['diff_std'].values      # plot value
    Y_scaling = df6['scaling'].values          # plot value

    # Select all (d, scaling) combinations that are close to optimal
    df4['cond'] = np.where((df4['diff_mean'] <= fraction_of_minimum*df4['diff_mean_amin']), True, False)
    df7 = df4.query('cond==True')
    df7.drop(['cond', ], axis=1, inplace=True)
    # print("\n-- df7: all good data points(length {}):\n{}".format(len(df7.index), df7.head(90)))
    X_points = df7['f'].values                  # plot value
    Y_points = df7['scaling'].values            # plot value

    # Select average (and lower and upper bound) on good data points
    df8 = df7.groupby(['f']).agg \
        ({'scaling': [np.mean, np.amin, np.amax, ],  # Multiple Aggregates
          })
    df8.columns = ['_'.join(col).strip() for col in df8.columns.values]  # flatten the column hierarchy
    df8.reset_index(inplace=True)  # remove the index hierarchy
    # print("\n-- df8: input for moving average (length {}):\n{}".format(len(df8.index), df8.head(15)))
    Y_point_mean = df8['scaling_mean'].values                  # plot value




    if CREATE_PDF or SHOW_PDF or SHOW_PLOT:
        # -- Setup figure
        fig_filename = 'Fig_MHE_Optimal_ScalingFactor_diff_f_{}.pdf'.format(CHOICE)
        mpl.rcParams['backend'] = 'pdf'
        mpl.rcParams['lines.linewidth'] = 3
        mpl.rcParams['font.size'] = 14
        mpl.rcParams['axes.labelsize'] = 20
        mpl.rcParams['axes.titlesize'] = 16
        mpl.rcParams['xtick.labelsize'] = 16
        mpl.rcParams['ytick.labelsize'] = 16
        mpl.rcParams['legend.fontsize'] = 16
        mpl.rcParams['axes.edgecolor'] = '111111'  # axes edge color
        mpl.rcParams['grid.color'] = '777777'  # grid color
        mpl.rcParams['figure.figsize'] = [4, 4]
        mpl.rcParams['xtick.major.pad'] = 4  # padding of tick labels: default = 4
        mpl.rcParams['ytick.major.pad'] = 4  # padding of tick labels: default = 4
        fig = plt.figure()
        ax = fig.add_axes([0.13, 0.17, 0.8, 0.8])

        # -- Draw the plots
        p1 = ax.plot(X_f, Y_diff0, color='blue', linewidth=2)
        ax.fill_between(X_f, Y_diff0 + Y_diff0_std, Y_diff0 - Y_diff0_std, facecolor='blue', alpha=0.2, edgecolor='none', label=r'$\tilde {\mathbf{H}}$')
        p2 = ax.plot(X_f, Y_diff1, color='red', linewidth=2)
        ax.fill_between(X_f, Y_diff1 + Y_diff1_std, Y_diff1 - Y_diff1_std, facecolor='red', alpha=0.2, edgecolor='none', label=r'$\tilde {\mathbf{H}}^{\ell}_{\mathrm{EC}}$')
        plt.xscale('log')
        plt.yscale('log')

        # -- Title and legend
        if distribution == 'uniform':
            distribution_label = ',$uniform'
        else:
            distribution_label = '$'
        plt.title(r'$\!\!\!n\!=\!{}\mathrm{{k}}, h\!=\!{}, d\!=\!{}{}'.format(int(n / 1000), h, d, distribution_label))
        handles, labels = ax.get_legend_handles_labels()
        legend = plt.legend(handles, labels,
                            loc='upper right',     # 'upper right'
                            handlelength=1.5,
                            labelspacing=0,  # distance between label entries
                            handletextpad=0.3,  # distance between label and the line representation
                            # title='Variants',
                            borderaxespad=0.2,  # distance between legend and the outer axes
                            borderpad=0.3,  # padding inside legend box
                            )
        frame = legend.get_frame()
        # frame.set_linewidth(0.0)
        frame.set_alpha(0.9)  # 0.8

        # -- Figure settings and save
        plt.xticks(xtick_lab, xtick_lab)
        # plt.yticks(ytick_lab1, ytick_lab1)
        plt.grid(b=True, which='minor', axis='both', alpha=0.2, linestyle='solid', linewidth=0.5)  # linestyle='dashed', which='minor', axis='y',
        plt.grid(b=True, which='major', axis='y', alpha=0.2, linestyle='solid', linewidth=0.5)  # linestyle='dashed', which='minor', axis='y',
        plt.xlabel(r'$f$', labelpad=0)      # labelpad=0
        plt.ylabel(r'L$^2$ norm', labelpad=-5)

        if xmin is None:
            xmin = plt.xlim()[0]
        if xmax is None:
            xmax = plt.xlim()[1]
        if ymin1 is None:
            ymin1 = plt.ylim()[1]
        if ymax1 is None:
            ymax1 = plt.ylim()[1]
        plt.xlim(xmin, xmax)
        plt.ylim(ymin1, ymax1)
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
        if SHOW_PLOT:
            plt.show()



    if CREATE_PDF or SHOW_PDF or SHOW_PLOT:
        # -- Setup figure
        fig_filename = 'Fig_MHE_Optimal_ScalingFactor_lambda_f_{}.pdf'.format(CHOICE)
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


        # # masking points outside plot area (not needed in the end)
        # print(Y_points)
        # Y_points_masked = np.ma.masked_less_equal(Y_points, ymin2)
        # print(Y_points_masked)
        # print(Y_points_masked.mask)


        # -- Draw the plots
        p1 = ax.plot(X_points, Y_points, color='0.8', linewidth=0, marker='o',  markeredgewidth=0.0,
                     clip_on=True,  # cut off data points outside of plot area
                     zorder=9,
                     label=r'$\!\leq${} Opt'.format(fraction_of_minimum))
        p2 = ax.plot(X_f, Y_scaling, color='red', linewidth=0, marker='o',
                     clip_on=True,
                     zorder=10,
                     label=r'Opt$(\lambda|d)$')
        plt.xscale('log')
        plt.yscale('log')

        # Draw the moving average from Y_point_mean
        def movingaverage(interval, window_size):
            window = np.ones(int(window_size))/float(window_size)
            return np.convolve(interval, window, 'same')

        Y_point_mean_window = movingaverage(Y_point_mean, 3)
        p5 = ax.plot(X_f, Y_point_mean_window, color='red', linewidth=1, marker=None)

        # -- Title and legend
        if distribution == 'uniform':
            distribution_label = ',$uniform'
        else:
            distribution_label = '$'
        plt.title(r'$\!\!\!n\!=\!{}\mathrm{{k}}, h\!=\!{}, d\!=\!{}{}'.format(int(n / 1000), h, d, distribution_label))
        handles, labels = ax.get_legend_handles_labels()
        legend = plt.legend(handles[::-1], labels[::-1],
                            loc='upper right',     # 'upper right'
                            handlelength=1,
                            labelspacing=0,  # distance between label entries
                            handletextpad=0.3,  # distance between label and the line representation
                            borderaxespad=0.3,  # distance between legend and the outer axes
                            borderpad=0.1,  # padding inside legend box
                            numpoints=1,    # put the marker only once
                            )
        frame = legend.get_frame()
        # frame.set_linewidth(0.0)
        frame.set_alpha(0.9)  # 0.8

        # -- Figure settings and save
        plt.xticks(xtick_lab, xtick_lab)
        plt.yticks(ytick_lab2, ytick_lab2)
        plt.grid(b=True, which='minor', alpha=0.2, linestyle='solid', linewidth=0.5)  # linestyle='dashed', which='minor', axis='y',
        plt.xlabel(r'$f$', labelpad=0)      # labelpad=0
        plt.ylabel(r'$\lambda$', labelpad=0, rotation=0)

        if xmin is None:
            xmin = plt.xlim()[0]
        if xmax is None:
            xmax = plt.xlim()[1]
        if ymin2 is None:
            ymin2 = plt.ylim()[0]
        if ymax2 is None:
            ymax2 = plt.ylim()[1]
        plt.xlim(xmin, xmax)
        plt.ylim(ymin2, ymax2)

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
            showfig(join(figure_directory, fig_filename))  # shows actually created PDF
        if SHOW_PLOT:
            plt.show()

if __name__ == "__main__":
    run(3, show_plot=True)
