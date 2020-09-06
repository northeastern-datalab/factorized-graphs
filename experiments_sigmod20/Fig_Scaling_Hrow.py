"""
Creates small figure that shows calculating Hrow^{(l)} is considerably faster than naive calculation of W^l

First version: Nov 3, 2016
This version: March 3, 2020
Author: Wolfgang Gatterbauer
"""

import numpy as np
import time
import datetime
import random
import os  # for displaying created PDF
import sys
sys.path.append('./../sslh')
from fileInteraction import save_csv_record
from utils import (from_dictionary_beliefs,
                   create_parameterized_H,
                   replace_fraction_of_rows,
                   showfig)
from graphGenerator import (planted_distribution_model_H,
                            calculate_average_outdegree_from_graph)
from estimation import (H_observed,
                        M_observed)
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', None)  # show all columns in pandas by default

# -- Determine path to data *irrespective* of where the file is run from
from os.path import abspath, dirname, join
from inspect import getfile, currentframe

current_path = dirname(abspath(getfile(currentframe())))
figure_directory = join(current_path, 'figs')
data_directory = join(current_path, 'datacache')
open_cmd = {'linux': 'xdg-open', 'linux2': 'xdg-open', 'darwin': 'open', 'win32': 'start'}


# %% -- main method
def run(choice, variant, create_data=False, show_plot=False, create_pdf=False, show_pdf=False, append_data=False):
    """main parameterized method to produce all figures.
    Can be run from external jupyther notebook or method to produce all figures, optionally as PDF
    CHOICE uses a different saved experimental run
    VARIANT uses a different wayt o plot
    """

    # %% -- Setup
    CREATE_DATA = create_data
    APPEND_DATA = append_data   # allows to add more data, requires CREATE_DATA to be true
    CHOICE = choice
    VARIANT = variant
    SHOW_PLOT = show_plot
    CREATE_PDF = create_pdf
    SHOW_PDF = show_pdf
    BOTH = True  # show both figures for W and H
    SHOW_TITLE = True  # show parameters in title of plot
    f = 1  # fraction of labeled nodes for H estimation

    csv_filename = 'Fig_Scaling_Hrow_{}.csv'.format(CHOICE)
    fig_filename = 'Fig_Scaling_Hrow_{}-{}.pdf'.format(CHOICE, VARIANT)

    plot_colors = ['darkorange', 'blue']
    header = ['currenttime',
              'choice',  # W, or H
              'l',
              'time']
    if CREATE_DATA and not APPEND_DATA:
        save_csv_record(join(data_directory, csv_filename), header, append=APPEND_DATA)
    RANDOMSEED = None  # For repeatability
    random.seed(RANDOMSEED)  # seeds some other python random generator
    np.random.seed(seed=RANDOMSEED)  # seeds the actually used numpy random generator; both are used and thus needed

    # %% -- Default parameters
    n = 10000
    ymax = 10
    h = 3
    d = 10  # actual degree is double
    distribution = 'uniform'
    exponent = None

    # %% -- CHOICES and VARIANTS
    if CHOICE == 1:
        W_repeat = [0, 0, 30, 5, 3, 1]  # index starts with 0. useful only for W^2 and later
        H_repeat = [0, 50, 50, 50, 50, 50, 50, 50, 50]
        W_annotate_x = 4.3
        W_annotate_y = 1
        H_annotate_x = 6
        H_annotate_y = 0.005

    elif CHOICE == 2:  # small exponent 3, does not show the advantage well
        d = 3
        W_repeat = [0, 0, 10, 5, 5, 5, 5, 5, 5]  # index starts with 0. useful only for W^2 and later
        H_repeat = [0, 50, 50, 50, 50, 50, 50, 50, 50]
        W_annotate_x = 5
        W_annotate_y = 0.08
        H_annotate_x = 6.5
        H_annotate_y = 0.004

    elif CHOICE == 3:  # small exponent 2, does not show the advantage well
        d = 2
        W_repeat = [0, 0, 50, 50, 50, 50, 50, 50, 50]  # index starts with 0. useful only for W^2 and later
        H_repeat = [0, 50, 50, 50, 50, 50, 50, 50, 50]
        W_annotate_x = 6.5
        W_annotate_y = 0.02
        H_annotate_x = 6.5
        H_annotate_y = 0.004

    elif CHOICE == 4:
        distribution = 'powerlaw'
        exponent = -0.5
        W_repeat = [0, 0, 50, 9, 5, 3]  # index starts with 0. useful only for W^2 and later
        H_repeat = [0, 50, 50, 50, 50, 50, 50, 50, 50]
        W_annotate_x = 4
        W_annotate_y = 1
        H_annotate_x = 6.5
        H_annotate_y = 0.006

        if VARIANT == 1:
            plot_colors = ['blue', 'darkorange']
            SHOW_TITLE = False

        if VARIANT == 2:
            plot_colors = ['blue', 'darkorange']
            BOTH = False
            SHOW_TITLE = False

    elif CHOICE == 5:
        distribution = 'powerlaw'
        exponent = -0.5
        W_repeat = [0, 0, 1, 1]  # index starts with 0. useful only for W^2 and later
        H_repeat = [0] + [1] * 8
        W_annotate_x = 4
        W_annotate_y = 1
        H_annotate_x = 6.5
        H_annotate_y = 0.006

    elif CHOICE == 11:
        W_repeat = [0, 0, 1, 1, 0, 0]  # index starts with 0. useful only for W^2 and later
        H_repeat = [0, 50, 50, 50, 50, 50, 50, 50, 50]
        W_annotate_x = 4.3
        W_annotate_y = 1
        H_annotate_x = 6
        H_annotate_y = 0.005

    elif CHOICE == 12:
        W_repeat = [0, 0, 31, 11, 5, 3, 3, 3, 3]  # index starts with 0. useful only for W^2 and later
        H_repeat = [0, 50, 50, 50, 50, 50, 50, 50, 50]
        W_annotate_x = 4.3
        W_annotate_y = 2.5
        H_annotate_x = 5.5
        H_annotate_y = 0.004
        f = 0.1
        plot_colors = ['blue', 'darkorange']
        ymax = 100

        if VARIANT == 1:    # TODO: when trying to add additional data, then it creates 7 instead of 4 rows,
                            # but the same code idea of CREATE vs ADD data appears to work in Fig_MHE_Optimal_Lambda, for that to replicate run below
                            # run(12, 1, create_pdf=True, show_pdf=True, create_data=False, append_data=True)
            W_repeat = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # index starts with 0. useful only for W^2 and later
            H_repeat = [0, 50, 50, 50, 50, 50, 50, 50, 50]

    else:
        raise Warning("Incorrect choice!")

    # %% -- Create data
    if CREATE_DATA or APPEND_DATA:

        # Create graph
        k = 3
        a = 1
        alpha0 = np.array([a, 1., 1.])
        alpha0 = alpha0 / np.sum(alpha0)
        H0 = create_parameterized_H(k, h, symmetric=True)
        start = time.time()
        W, Xd = planted_distribution_model_H(n, alpha=alpha0, H=H0, d_out=d,
                                             distribution=distribution,
                                             exponent=exponent,
                                             directed=False,
                                             debug=False)
        X0 = from_dictionary_beliefs(Xd)
        time_calc = time.time() - start
        # print("\nTime for graph:{}".format(time_calc))
        # print("Average outdegree: {}".format(calculate_average_outdegree_from_graph(W)))

        # Calculations W
        for length, rep in enumerate(W_repeat):

            for _ in range(rep):
                start = time.time()
                if length == 2:
                    result = W.dot(W)
                elif length == 3:
                    result = W.dot(W.dot(W))  # naive enumeration used as nothing can be faster
                elif length == 4:
                    result = W.dot(W.dot(W.dot(W)))
                elif length == 5:
                    result = W.dot(W.dot(W.dot(W.dot(W))))
                elif length == 6:
                    result = W.dot(W.dot(W.dot(W.dot(W.dot(W)))))
                elif length == 7:
                    result = W.dot(W.dot(W.dot(W.dot(W.dot(W.dot(W))))))
                elif length == 8:
                    result = W.dot(W.dot(W.dot(W.dot(W.dot(W.dot(W.dot(W)))))))
                elif length == 9:
                    result = W.dot(W.dot(W.dot(W.dot(W.dot(W.dot(W.dot(W.dot(W))))))))
                time_calc = time.time() - start

                tuple = [str(datetime.datetime.now())]
                text = ['W',
                        length,
                        time_calc]
                text = np.asarray(text)  # without np, entries get ugly format
                tuple.extend(text)
                # print("W, d: {}, time: {}".format(length, time_calc))
                save_csv_record(join(data_directory, csv_filename), tuple)

        # Calculations H_NB
        for length, rep in enumerate(H_repeat):

            for _ in range(rep):
                X0 = from_dictionary_beliefs(Xd)
                X1, ind = replace_fraction_of_rows(X0, 1 - f)

                start = time.time()
                result = H_observed(W, X=X1, distance=length, NB=True, variant=1)
                time_calc = time.time() - start

                tuple = [str(datetime.datetime.now())]
                text = ['H',
                        length,
                        time_calc]
                text = np.asarray(text)  # without np, entries get ugly format
                tuple.extend(text)
                # print("H, d: {}, time: {}".format(length, time_calc))
                save_csv_record(join(data_directory, csv_filename), tuple)

        # Calculate and display M statistics
        for length, _ in enumerate(H_repeat):
            M = M_observed(W, X=X0, distance=length, NB=True)
            M = M[-1]
            s = np.sum(M)
            # print("l: {}, sum: {:e}, M:\n{}".format(length, s, M))

    # %% -- Read, aggregate, and pivot data
    df1 = pd.read_csv(join(data_directory, csv_filename))
    # print("\n-- df1 (length {}):\n{}".format(len(df1.index), df1.head(15)))
    df2 = df1.groupby(['choice', 'l']).agg \
        ({'time': [np.max, np.mean, np.median, np.min, np.size],  # Multiple Aggregates
          })
    df2.columns = ['_'.join(col).strip() for col in df2.columns.values]  # flatten the column hierarchy
    df2.reset_index(inplace=True)  # remove the index hierarchy
    df2.rename(columns={'time_size': 'count'}, inplace=True)
    # print("\n-- df2 (length {}):\n{}".format(len(df2.index), df2.head(30)))
    df3 = pd.pivot_table(df2, index=['l'], columns=['choice'], values='time_median', )  # Pivot
    # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(30)))

    #%% -- Setup figure
    mpl.rcParams['backend'] = 'pdf'
    mpl.rcParams['lines.linewidth'] = 3
    mpl.rcParams['font.size'] = 20
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['axes.edgecolor'] = '111111'  # axes edge color
    mpl.rcParams['grid.color'] = '777777'  # grid color
    mpl.rcParams['figure.figsize'] = [4, 4]
    mpl.rcParams['xtick.major.pad'] = 6  # padding of tick labels: default = 4
    mpl.rcParams['ytick.major.pad'] = 4  # padding of tick labels: default = 4
    fig = plt.figure()
    ax = fig.add_axes([0.13, 0.17, 0.8, 0.8])

    #%% -- Draw the plot and annotate
    df4 = df3['H']
    # print("\n-- df4 (length {}):\n{}".format(len(df4.index), df4.head(30)))

    Y1 = df3['W'].plot(logy=True, color=plot_colors[0], marker='o', legend=None,
                       clip_on=False,  # cut off data points outside of plot area
                       # zorder=3
                       )  # style='o', kind='bar', style='o-',

    plt.annotate(r'$\mathbf{W}^\ell$',
                 xy=(W_annotate_x, W_annotate_y),
                 color=plot_colors[0],
                 )

    if BOTH:
        Y2 = df3['H'].plot(logy=True, color=plot_colors[1], marker='o', legend=None,
                           clip_on=False,  # cut off data points outside of plot area
                           zorder=3
                           )  # style='o', kind='bar', style='o-',

        plt.annotate(r'$\mathbf{\hat P}_{\mathrm{NB}}^{(\ell)}$',
                     xy=(H_annotate_x, H_annotate_y),
                     color=plot_colors[1],
                     )
    if SHOW_TITLE:
        plt.title(r'$\!\!\!\!n\!=\!{}\mathrm{{k}}, d\!=\!{}, h\!=\!{}, f\!=\!{}$'.format(int(n / 1000), 2 * d, h, f))

    # %% -- Figure settings & plot
    plt.grid(b=True, which='both', alpha=0.2, linestyle='solid', axis='y', linewidth=0.5)  # linestyle='dashed', which='minor'
    plt.xlabel(r'Path length ($\ell$)', labelpad=0)
    plt.ylabel(r'$\!$Time [sec]', labelpad=1)
    plt.ylim(0.001, ymax)  # placed after yticks
    plt.xticks(range(1, 9))

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
                    # frameon=None
                    )
    if SHOW_PDF:
        # os.system('{} "'.format(open_cmd[sys.platform]) + join(figure_directory, fig_filename) + '"')       # shows actually created PDF
        showfig(join(figure_directory, fig_filename))  # shows actually created PDF       # TODO replace with this method


if __name__ == "__main__":
    run(12, 0, create_pdf=True, show_pdf=True, create_data=False, append_data=False)
