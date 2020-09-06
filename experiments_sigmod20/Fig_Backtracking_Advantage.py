"""
Creates small figure that shows H_row_EC gives an unbiased estimate of H (in contrast to just H_row)
Uses variant 1 for with and without EC
For each length l, a different entry from the first row of H_rows are chosen (H[0][(l+1) % 2] which corresponds to the max.
(Just using max will sometimes choose another entry and thus not give an unbiased estimate)

First version: Nov 3, 2016
This version: March 3, 2020
Author: Wolfgang Gatterbauer
"""


import numpy as np
import time
import datetime
import random
# import os                       # for displaying created PDF
import sys
sys.path.append('./../sslh')
from fileInteraction import save_csv_record
from utils import from_dictionary_beliefs, create_parameterized_H, replace_fraction_of_rows, showfig
from graphGenerator import (planted_distribution_model_H,
                            calculate_average_outdegree_from_graph)
from estimation import M_observed, H_observed
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)      # show all columns


# -- Determine path to data *irrespective* of where the file is run from
from os.path import abspath, dirname, join
from inspect import getfile, currentframe
current_path = dirname(abspath(getfile(currentframe())))
figure_directory = join(current_path, 'figs')
data_directory = join(current_path, 'datacache')


# %% -- main method
def run(choice, variant, create_data=False, show_plot=False, create_pdf=False, show_pdf=False):
    """main parameterized method to produce all figures.
    Can be run from external jupyther notebook or method to produce all figures in PDF
    """

    # %% -- Setup
    CREATE_DATA = create_data
    CHOICE = choice
    VARIANT = variant
    SHOW_PLOT = show_plot
    CREATE_PDF = create_pdf
    SHOW_PDF = show_pdf
    SHOW_TITLE = True
    LEGEND_MATCH_COLORS = False
    SHOW_DISTRIBUTION_IN_TITLE = True

    SHOW_BACKTRACK_ESTIMATE = True
    SHOW_NONBACKTRACK_ESTIMATE = True
    plot_colors = ['darkgreen', 'darkorange', 'blue']
    label_vec = [r'$\mathbf{H}^{\ell}\,\,\,\,$', r'$\mathbf{\hat P}^{(\ell)}$', r'$\mathbf{\hat P}_{\mathrm{NB}}^{(\ell)}$']

    csv_filename = 'Fig_Backtracking_Advantage_{}.csv'.format(CHOICE)
    fig_filename = 'Fig_Backtracking_Advantage_{}-{}.pdf'.format(CHOICE, VARIANT)

    header = ['currenttime',
              'choice',     # H, Hrow, HrowEC
              'l',
              'valueH',    # maximal values in first row of H
              'valueM']    # average value across entries in M
    if CREATE_DATA:
        save_csv_record(join(data_directory, csv_filename), header, append=False)

    # %% -- Default parameters
    ymin=0.3
    ymax=1
    exponent = None

    # %% -- CHOICES and VARIANTS
    if CHOICE == 1:     # n=1000, shows NB to be slight lower for l=2: probably due to sampling issues (d=3, thus very few points available)
        n = 1000
        h = 8
        d = 3
        f = 0.1
        distribution = 'uniform'
        rep = 10000
        length = 8

    elif CHOICE == 2:
        n = 1000
        h = 8
        d = 10
        f = 0.1
        distribution = 'uniform'
        rep = 10000
        length = 8

    elif CHOICE == 3: # nice: shows nicely that difference is even bigger for smaller h
        n = 1000
        h = 3
        d = 10
        f = 0.1
        distribution = 'uniform'
        rep = 10000
        length = 8
        ymax = 0.8

    elif CHOICE == 4:
        n = 10000
        h = 3
        d = 10
        f = 0.1
        distribution = 'uniform'
        rep = 100
        length = 8
        ymin = 0.333
        ymax = 0.65

    elif CHOICE == 5:
        n = 10000
        h = 3
        d = 3
        f = 0.1
        distribution = 'uniform'
        rep = 1000
        length = 8

    elif CHOICE == 6:  # n=1000, the powerlaw problem with small graphs and high exponent
        n = 1000
        h = 8
        d = 3
        f = 0.1
        distribution = 'powerlaw'
        exponent = -0.5
        rep = 10000
        length = 8

    elif CHOICE == 7:
        n = 10000
        h = 8
        d = 3
        f = 0.1
        distribution = 'uniform'
        rep = 1000
        length = 8
        # ymin = 0.4
        ymax = 1

    elif CHOICE == 8:
        n = 10000
        h = 8
        d = 10
        f = 0.1
        distribution = 'uniform'
        rep = 1000
        length = 8
        # ymin = 0.4
        ymax = 1

    elif CHOICE == 9: # shows lower NB due to problem with sampling from high powerlaw -0.5
        n = 10000
        h = 8
        d = 10
        f = 0.1
        distribution = 'powerlaw'
        exponent = -0.5
        rep = 1000
        length = 8

    elif CHOICE == 10:
        n = 10000
        h = 8
        d = 3
        f = 0.1
        distribution = 'powerlaw'
        exponent = -0.5
        rep = 1000
        length = 8

    elif CHOICE == 11:    # problem: shows that NB is too low (probably because of problem with sampling from -0.5 factor)
        n = 1000
        h = 8
        d = 10
        f = 0.1
        distribution = 'powerlaw'
        exponent = -0.5
        rep = 1000
        length = 8

    elif CHOICE == 12:  # problem: shows no problem with NB (probably because no problem with sampling from -0.2 factor)
        n = 1000
        h = 8
        d = 10
        f = 0.1
        distribution = 'powerlaw'
        exponent = -0.2
        rep = 1000
        length = 8

    elif CHOICE == 20:
        n = 10000
        h = 3
        d = 10
        f = 0.1
        distribution = 'powerlaw'
        exponent = -0.3
        rep = 1000
        length = 8
        ymin = 0.333
        ymax = 0.65

    elif CHOICE == 21:          # originally used before color change
        n = 10000
        h = 3
        d = 25
        f = 0.1
        distribution = 'powerlaw'
        exponent = -0.3
        rep = 1000
        length = 8
        ymin = 0.333
        ymax = 0.65

        if VARIANT == 1:
            SHOW_TITLE = False
            plot_colors = ['red', 'blue', 'darkorange']
            label_vec = [r'$\mathbf{H}^{\ell}\quad\quad$', 'naive', 'better']
            LEGEND_MATCH_COLORS = True

        if VARIANT == 2:
            SHOW_TITLE = False
            plot_colors = ['red', 'blue', 'darkorange']
            label_vec = [r'$\mathbf{H}^{\ell}\quad\quad$', 'naive', 'better']
            SHOW_NONBACKTRACK_ESTIMATE = False
            LEGEND_MATCH_COLORS = True

        if VARIANT == 3:
            SHOW_TITLE = False
            plot_colors = ['red', 'blue', 'darkorange']
            label_vec = [r'$\mathbf{H}^{\ell}\quad\quad$', 'naive', 'better']
            SHOW_BACKTRACK_ESTIMATE = False
            SHOW_NONBACKTRACK_ESTIMATE = False
            LEGEND_MATCH_COLORS = True

        if VARIANT == 4:
            plot_colors = ['red', 'blue', 'darkorange']
            LEGEND_MATCH_COLORS = True

    elif CHOICE == 25:
        n = 10000
        h = 8
        d = 5
        f = 0.1
        distribution = 'uniform'
        rep = 1000
        length = 8

    elif CHOICE == 26:
        n = 10000
        h = 8
        d = 25
        f = 0.1
        distribution = 'uniform'
        rep = 1000
        length = 8
        ymax = 0.9
        ymin = 0.4

    elif CHOICE == 27:
        n = 10000
        h = 8
        d = 10
        f = 0.1
        distribution = 'powerlaw'
        exponent = -0.3
        rep = 1000
        length = 8
        ymax = 0.9
        ymin = 0.33

    elif CHOICE == 31:
        n = 10000
        h = 3
        d = 10
        f = 0.1
        distribution = 'uniform'
        rep = 1000
        length = 8
        ymin = 0.333
        ymax = 0.65
        SHOW_DISTRIBUTION_IN_TITLE = False
        plot_colors = ['red', 'blue', 'darkorange']
        LEGEND_MATCH_COLORS = True

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


    # %% -- Create data
    if CREATE_DATA:

        # Calculations H
        print("Max entry of first rows of powers of H0:")
        for l in range(1, length + 1):
            valueH = np.max(np.linalg.matrix_power(H0, l)[0])

            tuple = [str(datetime.datetime.now())]
            text = ['H',
                    l,
                    valueH,
                    '']
            text = np.asarray(text)  # without np, entries get ugly format
            tuple.extend(text)
            print("{}: {}".format(l, valueH))
            save_csv_record(join(data_directory, csv_filename), tuple)

        # Calculations Hrow and HrowEC
        for r in range(rep):
            print('Repetition {}'.format(r))

            # Create graph
            start = time.time()
            W, Xd = planted_distribution_model_H(n, alpha=alpha0, H=H0, d_out=d,    # notice that for undirected graphs, actual degree = 2*d
                                                      distribution=distribution,
                                                      exponent=exponent,
                                                      directed=False,
                                                      debug=False)
            X0 = from_dictionary_beliefs(Xd)
            X1, ind = replace_fraction_of_rows(X0, 1 - f)
            time_calc = time.time() - start
            # print("\nTime for graph:{}".format(time_calc))

            print("Average outdegree: {}".format(calculate_average_outdegree_from_graph(W)))

            # Calculate H_vec and M_vec versions (M_vec to calculate the average number of entries in M)
            H_vec = H_observed(W, X1, distance=length, NB=False, variant=1)
            H_vec_EC = H_observed(W, X1, distance=length, NB=True, variant=1)
            M_vec = M_observed(W, X1, distance=length, NB=False)
            M_vec_EC = M_observed(W, X1, distance=length, NB=True)

            # Calculation H_vec
            # print("Max entry of first rows of H_vec")
            for l, H in enumerate(H_vec):
                valueH = H[0][(l + 1) % 2]       # better than 'value = np.max(H[0])', otherwise sometimes chooses another higher entry -> biased estimate
                valueM = np.average(M_vec[l+1])
                # print(M_vec[l+1])
                # print(valueM)

                tuple = [str(datetime.datetime.now())]
                text = ['Hrow',
                        l + 1,
                        valueH,
                        valueM]
                text = np.asarray(text)         # without np, entries get ugly format
                tuple.extend(text)
                # print("{}: {}".format(l + 1, value))
                save_csv_record(join(data_directory, csv_filename), tuple)

            # Calculation H_vec_EC
            # print("Max entry of first rows of H_vec_EC")
            for l, H in enumerate(H_vec_EC):
                valueH = H[0][(l + 1) % 2]
                valueM = np.average(M_vec_EC[l+1])
                # print(M_vec_EC[l+1])
                # print(valueM)

                tuple = [str(datetime.datetime.now())]
                text = ['HrowEC',
                        l + 1,
                        valueH,
                        valueM]
                text = np.asarray(text)  # without np, entries get ugly format
                tuple.extend(text)
                # print("{}: {}".format(l + 1, value))
                save_csv_record(join(data_directory, csv_filename), tuple)


    #%% -- Read, aggregate, and pivot data
    df1 = pd.read_csv(join(data_directory, csv_filename))
    # print("\n-- df1 (length {}):\n{}".format(len(df1.index), df1.head(15)))
    df2 = df1.groupby(['choice', 'l']).agg \
        ({'valueH': [np.mean, np.std, np.size],  # Multiple Aggregates
          'valueM': [np.mean],
          })
    df2.columns = ['_'.join(col).strip() for col in df2.columns.values]     # flatten the column hierarchy
    df2.reset_index(inplace=True)                                           # remove the index hierarchy
    df2.rename(columns={'valueH_size': 'count'}, inplace=True)
    # print("\n-- df2 (length {}):\n{}".format(len(df2.index), df2.head(30)))
    df3 = pd.pivot_table(df2, index=['l'], columns=['choice'], values=['valueH_mean', 'valueH_std', 'valueM_mean'])  # Pivot
    # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(30)))
    df3.columns = ['_'.join(col).strip() for col in df3.columns.values]     # flatten the column hierarchy
    # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(30)))
    # df3.drop(['valueM_mean_H', 'valueH_std_H'], axis=1, inplace=True)
    # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(30)))
    df3.reset_index(level=0, inplace=True)                                  # get l into columns
    # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(30)))

    #%% -- Setup figure
    mpl.rcParams['backend'] = 'pdf'
    mpl.rcParams['lines.linewidth'] = 3
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['legend.fontsize'] = 20
    mpl.rcParams['axes.edgecolor'] = '111111'   # axes edge color
    mpl.rcParams['grid.color'] = '777777'   # grid color
    mpl.rcParams['figure.figsize'] = [4, 4]
    mpl.rcParams['xtick.major.pad'] = 4     # padding of tick labels: default = 4
    mpl.rcParams['ytick.major.pad'] = 4         # padding of tick labels: default = 4

    fig = plt.figure()
    ax = fig.add_axes([0.13, 0.17, 0.8, 0.8])

    #%% -- Extract values into columns (plotting dataframew with bars plus error lines and lines gave troubles)
    l_vec = df3['l'].values                   # .tolist() does not work with bar plot
    mean_H_vec = df3['valueH_mean_H'].values
    mean_Hrow_vec = df3['valueH_mean_Hrow'].values
    mean_Hrow_vecEC = df3['valueH_mean_HrowEC'].values
    std_Hrow_vec = df3['valueH_std_Hrow'].values
    std_Hrow_vecEC = df3['valueH_std_HrowEC'].values

    #%% -- Draw the plot and annotate
    width = 0.3       # the width of the bars
    if SHOW_BACKTRACK_ESTIMATE:
        left_vec = l_vec
        if SHOW_NONBACKTRACK_ESTIMATE:
            left_vec = left_vec-width
        bar1 = ax.bar(left_vec, mean_Hrow_vec, width, color=plot_colors[1],
                      yerr=std_Hrow_vec, error_kw={'ecolor':'black', 'linewidth':2},    # error-bars colour
                      label=label_vec[1])
    if SHOW_NONBACKTRACK_ESTIMATE:
        bar2 = ax.bar(l_vec, mean_Hrow_vecEC, width, color=plot_colors[2],
                      yerr=std_Hrow_vecEC, error_kw={'ecolor':'black', 'linewidth':2},  # error-bars colour
                      label=label_vec[2])
    gt = ax.plot(l_vec, mean_H_vec, color=plot_colors[0], linestyle ='solid', linewidth = 2,
                 marker='o', markersize=10, markeredgewidth=2, markerfacecolor='None', markeredgecolor=plot_colors[0],
                 label=label_vec[0])

    if CHOICE == 4 or CHOICE == 20:
        ax.annotate(np.round(mean_Hrow_vec[1], 2), xy=(2.15, 0.65), xytext=(2.1, 0.60),
                    arrowprops=dict(facecolor='black', arrowstyle="->"),)

    #%% -- Legend
    if distribution == 'uniform' and SHOW_DISTRIBUTION_IN_TITLE:
        distribution_label = ',$uniform'
    else:
        distribution_label = '$'
    if SHOW_TITLE:
        plt.title(r'$\!\!\!\!n\!=\!{}\mathrm{{k}}, d\!=\!{}, h\!=\!{}, f\!=\!{}{}'.format(int(n / 1000), 2*d, h, f, distribution_label))    # notice that actual d is double than in one direction

    handles, labels = ax.get_legend_handles_labels()
    legend = plt.legend(handles, labels,
                        loc='upper right',
                        handlelength=1.5,
                        labelspacing=0,             # distance between label entries
                        handletextpad=0.3,          # distance between label and the line representation
                        # title='Iterations'
                        borderaxespad=0.1,        # distance between legend and the outer axes
                        borderpad=0.1,                # padding inside legend box
                        numpoints=1,    # put the marker only once
                        )

    if LEGEND_MATCH_COLORS:     # TODO: how to get back the nicer line spacing defined in legend above after changing the legend text colors
       legend.get_texts()[0].set_color(plot_colors[0])
       if SHOW_BACKTRACK_ESTIMATE:
           legend.get_texts()[1].set_color(plot_colors[1])
       if SHOW_NONBACKTRACK_ESTIMATE:
           legend.get_texts()[2].set_color(plot_colors[2])

    frame=legend.get_frame()
    frame.set_linewidth(0.0)
    frame.set_alpha(0.8)        # 0.8



    # %% -- Figure settings & plot
    ax.set_xticks(range(10))
    plt.grid(b=True, which='both', alpha=0.2, linestyle='solid', axis='y', linewidth=0.5)  # linestyle='dashed', which='minor'
    plt.xlabel(r'Path length ($\ell$)', labelpad=0)
    plt.ylim(ymin,ymax)                 # placed after yticks
    plt.xlim(0.5,5.5)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off        TODO: Paul, this does not work anymore :(    1/26/2020
        top='off',         # ticks along the top edge are off
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
            # frameon=None
            )
    if SHOW_PDF:
        showfig(join(figure_directory, fig_filename))
    if SHOW_PLOT:
        plt.show()



if __name__ == "__main__":
    run(choice=31, variant=0, show_plot=False, create_pdf=True, show_pdf=True, create_data=False)
