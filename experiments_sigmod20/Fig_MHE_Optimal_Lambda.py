"""
Similar to '161108 - Fig_MHE_Optimal_ScalingFactor_d', just changing f instead of d
1: diff
2: lambda

First version: Nov 8, 2016
This version: Nov 9, 2016
Author: Wolfgang Gatterbauer
"""

# from __future__ import division             # allow integer division
# from __future__ import print_function
import numpy as np
from numpy import linalg as LA
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
                        M_observed)
import matplotlib as mpl
mpl.use('Agg')
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

open_cmd = {'linux' : 'xdg-open','linux2' : 'xdg-open', 'darwin' : 'open', 'win32' : 'start'}

def run(choice, create_data=False, add_data=False, show_plot=False, create_pdf=False, show_pdf=False, show_fig=True):
    # -- Setup
    CHOICE = choice
    CREATE_DATA = create_data
    ADD_DATA = add_data
    SHOW_PLOT = show_plot
    CREATE_PDF = create_pdf
    SHOW_PDF=show_pdf
    SHOW_FIG1 = show_fig
    SHOW_FIG2 = show_fig

    csv_filename = 'Fig_MHE_Optimal_ScalingFactor_f_lambda10_{}.csv'.format(CHOICE)
    header = ['currenttime',
              'option',     # one option corresponds to one choice of weight vector. In practice, one choice of scaling factor (for weight vector)
              'f',
              'scaling',
              'diff']       # L2 norm between H and estimate
    if CREATE_DATA:
        save_csv_record(join(data_directory, csv_filename), header, append=False)


    # -- Default Graph parameters
    rep = 100
    randomize = False
    initial_h0 = None           # initial vector to start finding optimal H
    distribution = 'powerlaw'
    exponent = -0.3
    rep_differentGraphs = 1
    EC = True
    f_vec = [0.9 * pow(0.1, 1/12)**x for x in range(42)]
    fraction_of_minimum = 1.1           # scaling parameters that lead to optimum except for this scaling factor are included
    ymin2 = 0.28
    ymax2 = 500
    xmin = 0.001
    # xmin = 0.0005
    xmax = None
    xtick_lab = [0.001, 0.01, 0.1, 1]
    # ytick_lab1 = np.arange(0, 1, 0.1)
    ytick_lab2 = [0.3, 1, 10, 100, 1000]
    ymax1 = 1.2
    ymin1 = 0.001
    # ytick_lab1 = [0.001, 0.01, 0.1, 1]
    k = 3
    a = 1
    stratified = True
    gradient = False
    n = 10000
    # color_vec = ['blue', 'orange', 'red']
    color_vec =  ["#4C72B0", "#55A868", "#C44E52",  "#CCB974", "#64B5CD"]
    color_vec =  ["#4C72B0", "#8172B2", "#C44E52"]
    # label_vec = [r'$\tilde {\mathbf{H}}$', r'$\tilde{\mathbf{H}}^{(5)}_{\mathrm{NB}}$', r'$\tilde {\mathbf{H}}^{(5)}_{\mathrm{NB}}$ r']
    label_vec = ['MCE', 'DCE', 'DCEr']
    marker_vec = ['s', 'x', 'o']
    legendPosition =  'upper right'


    # -- Options
    if CHOICE == 11:
        h = 8
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3']
        scaling_vec = [0, 10, 10]
        randomize_vec = [False, False, True]
        length_vec = [1, 5, 5]

    elif CHOICE == 12:
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3']
        scaling_vec = [0, 10, 10]
        randomize_vec = [False, False, True]
        length_vec = [1, 5, 5]

    elif CHOICE == 13:
        h = 8
        d = 10
        option_vec = ['opt1', 'opt2', 'opt3']
        scaling_vec = [0, 10, 10]
        randomize_vec = [False, False, True]
        length_vec = [1, 5, 5]

    elif CHOICE == 14:
        h = 3
        d = 10
        option_vec = ['opt1', 'opt2', 'opt3']
        scaling_vec = [0, 10, 10]
        randomize_vec = [False, False, True]
        length_vec = [1, 5, 5]


    elif CHOICE == 15:
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3']
        scaling_vec = [0, 10, 100]
        randomize_vec = [False, False, True]
        length_vec = [1, 5, 5]

    # elif CHOICE == 16:
    #     n = 10000
    #     h = 3
    #     d = 10
    #     option_vec = ['opt1', 'opt2', 'opt3']
    #     scaling_vec = [0, 50, 50]
    #     randomize_vec = [False, False, True]
    #     length_vec = [1, 5, 5]

    elif CHOICE == 17:
        n = 1000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3']
        scaling_vec = [0, 10, 100]
        randomize_vec = [False, False, True]
        length_vec = [1, 5, 5]


    elif CHOICE == 18:
        n = 1000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3']
        scaling_vec = [0, 10, 10]
        randomize_vec = [False, False, True]
        length_vec = [1, 5, 5]

    # -- Options
    elif CHOICE == 19:
        h = 8
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3']
        scaling_vec = [0, 10, 100]
        randomize_vec = [False, False, True]
        length_vec = [1, 5, 5]


    elif CHOICE == 20:
        h = 8
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3']
        scaling_vec = [0, 10, 100]
        randomize_vec = [False, False, True]
        length_vec = [1, 5, 5]
        gradient = True
        legendPosition =  'center right'




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
        for rs in range(1, rep_differentGraphs+1):
           # print('Graph {}'.format(rs))

            # -- Create graph
            W, Xd = planted_distribution_model_H(n, alpha=alpha0, H=H0, d_out=d,
                                                 distribution=distribution,
                                                 exponent=exponent,
                                                 directed=False,
                                                 debug=False)
            X0 = from_dictionary_beliefs(Xd)

            for r in range(1, rep+1):
                # print('Repetition {}'.format(r))

                for f in f_vec:
                    # -- Sample labeled data
                    X1, ind = replace_fraction_of_rows(X0, 1 - f, stratified=stratified)

                    # -- Calculate number of labeled neighbors
                    M_vec = M_observed(W, X1, distance=5, NB=True)
                    M = M_vec[1]
                    num_N = np.sum(M)
                    # print("f={:1.4f}, number labeled neighbors={}".format(f, num_N))
                    # print("M_vec:\n{}".format(M_vec))

                    # -- Create estimates and compare against GT
                    for option, scaling, randomize, length in zip(option_vec, scaling_vec, randomize_vec, length_vec):
                        H_est = estimateH(X1, W, method='DHE', variant=1, distance=length, EC=EC, weights=scaling,
                                          randomize=randomize, initial_H0=initial_h0, gradient=gradient)
                        diff = LA.norm(H_est - H0)

                        tuple = [str(datetime.datetime.now())]
                        text = [option, f, scaling, diff]
                        tuple.extend(text)
                        save_csv_record(join(data_directory, csv_filename), tuple)

                        # print("diff={:1.4f}, H_est:\n{}".format(diff, H_est))





    # -- Read, aggregate, and pivot data for all options
    df1 = pd.read_csv(join(data_directory, csv_filename))
    # print("\n-- df1: (length {}):\n{}".format(len(df1.index), df1.head(15)))

    # Aggregate repetitions
    df2 = df1.groupby(['option', 'f']).agg \
        ({'diff': [np.mean, np.std, np.size],  # Multiple Aggregates
          })
    df2.columns = ['_'.join(col).strip() for col in df2.columns.values]  # flatten the column hierarchy
    df2.reset_index(inplace=True)  # remove the index hierarchy
    df2.rename(columns={'diff_size': 'count'}, inplace=True)
    # print("\n-- df2 (length {}):\n{}".format(len(df2.index), df2.head(15)))

    # Pivot table
    df3 = pd.pivot_table(df2, index=['f'], columns=['option'], values=['diff_mean', 'diff_std'] )  # Pivot
    # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(30)))
    df3.columns = ['_'.join(col).strip() for col in df3.columns.values]  # flatten the column hierarchy
    df3.reset_index(inplace=True)  # remove the index hierarchy
    # df2.rename(columns={'time_size': 'count'}, inplace=True)
    # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(30)))

    # Extract values
    X_f = df3['f'].values                     # plot x values
    Y=[]
    Y_std=[]
    for option in option_vec:
        Y.append(df3['diff_mean_{}'.format(option)].values)
        Y_std.append(df3['diff_std_{}'.format(option)].values)

    # print("X_f:\n", X_f)
    # print("Y:\n", Y)
    # print("Y_std:\n", Y_std)



    if SHOW_FIG1:
        # -- Setup figure
        fig_filename = 'Fig_MHE_Optimal_ScalingFactor_diff_f_lambda10_{}.pdf'.format(CHOICE)
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
        for i, (color, marker) in enumerate(zip(color_vec, marker_vec)):
            p=ax.plot(X_f, Y[i], color=color, linewidth=3, label=label_vec[i], marker=marker)
            if i != 1:
                ax.fill_between(X_f, Y[i] + Y_std[i], Y[i] - Y_std[i],
                                facecolor=color, alpha=0.2,
                                edgecolor='none')
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
                            loc=legendPosition,     # 'upper right'
                            handlelength=1.5,
                            labelspacing=0,  # distance between label entries
                            handletextpad=0.3,  # distance between label and the line representation
                            # title='Variants',
                            borderaxespad=0.2,  # distance between legend and the outer axes
                            borderpad=0.1,  # padding inside legend box
                            )
        frame = legend.get_frame()
        frame.set_linewidth(0.0)
        frame.set_alpha(0.9)  # 0.8


        # -- Figure settings and save
        plt.xticks(xtick_lab, xtick_lab)
        # plt.yticks(ytick_lab1, ytick_lab1)
        plt.grid(b=True, which='minor', axis='both', alpha=0.2, linestyle='solid', linewidth=0.5)  # linestyle='dashed', which='minor', axis='y',
        plt.grid(b=True, which='major', axis='y', alpha=0.2, linestyle='solid', linewidth=0.5)  # linestyle='dashed', which='minor', axis='y',
        plt.xlabel(r'Label Sparsity $(f)$', labelpad=0)      # labelpad=0
        plt.ylabel(r'L2 norm', labelpad=-5)

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

        if SHOW_FIG1:
            plt.show()
        if SHOW_PDF:
            os.system('{} "'.format(open_cmd[sys.platform]) + join(figure_directory, fig_filename) + '"')  # shows actually created PDF


if __name__ == "__main__":
    run(19, create_pdf=True, show_pdf=True)
