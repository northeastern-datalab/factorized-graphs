"""
TODO: replaces previous versions 161110

Plots Accuracy for labeling for various learning and propagation methods
Since graph creation takes most time, especially for large graphs, saves graphs to a file format, then loads them later again.

First version: Nov 10, 2016
This version: Jan 26, 2020
"""

import numpy as np
import datetime
import random
import sys
sys.path.append('./../sslh')
from fileInteraction import save_csv_record
from utils import (from_dictionary_beliefs,
                   create_parameterized_H,
                   replace_fraction_of_rows,
                   to_centering_beliefs,
                   eps_convergence_linbp_parameterized,
                   matrix_difference,
                   matrix_difference_classwise,
                   introduce_errors,
                   showfig)
from estimation import (estimateH,
                        estimateH_baseline_serial,
                        estimateH_baseline_parallel)
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import pandas as pd
pd.set_option('display.max_columns', None)      # show all columns from pandas
pd.options.mode.chained_assignment = None       # default='warn'
from graphGenerator import planted_distribution_model_H
from inference import linBP_symmetric_parameterized, beliefPropagation

    # import seaborn.apionly as sns                   # importing without activating it. For color palette




    # # -- Determine path to data *irrespective* of where the file is run from
    # from os.path import abspath, dirname, join
    # from inspect import getfile, currentframe
    # current_path = dirname(abspath(getfile(currentframe())))
    # figure_directory = join(current_path, 'figs')
    # data_directory = join(current_path, 'data')
    #
    #
    #
    # def run(choice, variant, create_data=False, add_data=False, create_graph=False,
    #         create_fig=True, show_plot=True, show_pdf=True, shorten_length=False, show_arrows=True):
    #     """main parameterized method to produce all figures.
    #     Can be run from external jupyther notebook or method to produce all figures in PDF
    #     """
    #
    #     # -- Setup
    #     CHOICE = choice                      # determines the CSV data file to use
    #     VARIANT = variant                     # determines the variant of how the figures are plotted
    #     CREATE_DATA = create_data             # starts new CSV file and stores experimental timing results
    #     ADD_DATA = add_data                # adds data to existing file
    #     CREATE_GRAPH = create_graph            # creates the actual graph for experiments (stores W and X in CSV files)







# -- Determine path to data *irrespective* of where the file is run from
from os.path import abspath, dirname, join
from inspect import getfile, currentframe
current_path = dirname(abspath(getfile(currentframe())))
figure_directory = join(current_path, 'figs')
data_directory = join(current_path, 'datacache')



def run(choice, variant, create_data=False, add_data=False, show_plot=False, create_pdf=False, show_pdf=False, show_fig=True):
    """main parameterized method to produce all figures.
    All relevant choice and parameters are encoded in this method.
    Calling methods just choose the CHOICE and VARIANT
    Can be run from external jupyther notebook or method to produce all figures in PDF
    """

    # -- Setup
    # 305, 315, 213, 108
    CHOICE = choice
    VARIANT = variant
    CREATE_DATA = create_data
    ADD_DATA = add_data
    SHOW_FIG = show_fig
    STD_FILL = True
    SHORTEN_LENGTH = False

    SHOW_PDF = show_pdf
    SHOW_PLOT = show_plot
    CREATE_PDF = create_pdf

    SHOW_TITLE = True  # show parameters in title of plot
    LABEL_FONTSIZE = 16  # size of number labels in figure

    csv_filename = 'Fig_End-to-End_accuracy_{}.csv'.format(CHOICE)
    filename = 'Fig_End-to-End_accuracy_{}-{}'.format(CHOICE, VARIANT)   # PDF filename includes CHOICE and VARIANT
    header = ['currenttime',
              'option',
              'f',
              'accuracy']
    if CREATE_DATA:
        save_csv_record(join(data_directory, csv_filename), header, append=False)







    # -- Default Graph parameters
    rep_SameGraph = 5           # iterations on same graph
    initial_h0 = None           # initial vector to start finding optimal H
    distribution = 'powerlaw'
    exponent = -0.3
    length = 5
    variant = 1
    EC = True                   # Non-backtracking for learning
    ymin = 0.3
    ymax = 1
    xmin = 0.001
    xmax = 1
    xtick_lab = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    xtick_labels = ['1e-5', '0.01\%', '0.1\%', '1\%', '10\%', '100\%']
    ytick_lab = np.arange(0, 1.1, 0.1)
    f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(21)]
    k = 3
    a = 1                   # this value was erroneously set to 5 previously!!! TODO: fix everywhere else
    rep_DifferentGraphs = 1   # iterations on different graphs
    err = 0
    avoidNeighbors = False
    convergencePercentage_W = None
    stratified = True
    labels = ['*']*10
    clip_on_vec = [True] * 10
    draw_std_vec = range(10)
    numberOfSplits = 1
    linestyle_vec = ['dashed'] + ['solid'] * 10
    linewidth_vec = [5, 4, 3, 3, 3, 3] + [3]*10
    marker_vec = [None, None, 'o', 'x', 'o', '^'] + [None]*10
    markersize_vec = [0, 0, 4, 8, 6, 6] + [6]*10
    propagation_method_vec = ['Lin'] * 10
    constraint_vec = [False]*15
    alpha0 = np.array([a, 1., 1.])


    facecolor_vec = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
    # SEABORN_PALETTES = dict(
    #     deep=["#4C72B0", "#55A868", "#C44E52",
    #           "#8172B2", "#CCB974", "#64B5CD"],
    #     muted=["#4878CF", "#6ACC65", "#D65F5F",
    #            "#B47CC7", "#C4AD66", "#77BEDB"],
    #     pastel=["#92C6FF", "#97F0AA", "#FF9F9A",
    #             "#D0BBFF", "#FFFEA3", "#B0E0E6"],
    #     bright=["#003FFF", "#03ED3A", "#E8000B",
    #             "#8A2BE2", "#FFC400", "#00D7FF"],
    #     dark=["#001C7F", "#017517", "#8C0900",
    #           "#7600A1", "#B8860B", "#006374"],
    #     colorblind=["#0072B2", "#009E73", "#D55E00",
    #                 "#CC79A7", "#F0E442", "#56B4E9"]
    #     )
    # facecolors = ['darkorange', 'blue', 'black']
    # facecolors = ['#6495ED', '#F08080', 'black']
    # facecolors = ['#66c2a5', '#fc8d62', '#8da0cb']
    # C = (sns.color_palette("colorblind", 4))
    # facecolor_vec = [C[0], C[2], C[1], C[3]]
    # facecolor_vec = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",]




    # -- Options with propagation variants
    if CHOICE == 101:
        n = 10000
        h = 8
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3']
        learning_method_vec = ['GT'] * 2 + ['DHE']
        weight_vec = [None] * 1 + [1] * 1 + [100] * 1
        alpha_vec = [0] * 3
        beta_vec = [0] * 1 + [1] * 2
        gamma_vec = [0] * 3
        s_vec = [0.5] + [3] * 2
        numMaxIt_vec = [10] + [4]*2
        randomize_vec = [False] * 2 + [True]
        xmin = 0.0001
        ymin = 0.6
        ymax = 1
        facecolor_vec = ['black', "#C44E52", "#4C72B0", "#8172B2", "#55A868", "#CCB974", "#64B5CD"]
        linestyle_vec = ['dashed'] + ['solid'] * 10
        labels = ['LinBP w/GT', 'CP w/GT', 'CP w/DCEr', 'BP']
        linewidth_vec = [5, 5, 3, ]
        marker_vec = [None, 'o', 'x']
        markersize_vec = [0, 8, 8]

    elif CHOICE == 111:
        n = 10000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3']
        learning_method_vec = ['GT'] * 2 + ['DHE']
        weight_vec = [None] * 1 + [1] * 1 + [100] * 1
        alpha_vec = [0] * 3
        beta_vec = [0] * 1 + [1] * 2
        gamma_vec = [0] * 3
        s_vec = [0.5] + [3] * 2
        numMaxIt_vec = [10] + [4] * 2
        randomize_vec = [False] * 2 + [True]
        xmin = 0.0001
        ymin = 0.6
        ymax = 1
        facecolor_vec = ['black', "#C44E52", "#4C72B0", "#8172B2", "#55A868", "#CCB974", "#64B5CD"]
        linestyle_vec = ['dashed'] + ['solid'] * 10
        labels = ['LinBP w/GT', 'CP w/GT', 'CP w/DCEr', 'BP']
        linewidth_vec = [5, 5, 3]
        marker_vec = [None, 'o', 'x']
        markersize_vec = [0, 8, 8]




    # BP options
    elif CHOICE == 301:      ## 101 with BP
        n = 10000
        h = 8
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4']
        learning_method_vec = ['GT'] * 2 + ['DHE'] + ['GT']
        weight_vec = [None] * 1 + [1] * 1 + [100] * 1 + [None]
        propagation_method_vec = ['Lin'] * 3 + ['BP']
        alpha_vec = [0] * 3 + [None]
        beta_vec = [0] * 1 + [1] * 2 + [None]
        gamma_vec = [0] * 3 + [None]
        s_vec = [0.5] + [3] * 2 + [0.1]
        numMaxIt_vec = [10] + [4]*2 + [10]
        randomize_vec = [False] * 2 + [True] + [False]
        xmin = 0.0001
        ymin = 0.6
        ymax = 1
        facecolor_vec = ['black', "#C44E52", "#4C72B0", "#8172B2", "#55A868", "#CCB974", "#64B5CD"]
        linestyle_vec = ['dashed'] + ['solid'] * 10
        labels = ['LinBP w/GT', 'CP w/GT', 'CP w/DCEr', 'BP']
        linewidth_vec = [5, 5, 3, 3]
        marker_vec = [None, 'o', 'x', '^']
        markersize_vec = [0, 8, 8, 8]



    elif CHOICE == 303:         ### like 311 BP, but with fewer iterations
        n = 10000
        h = 8
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5']
        learning_method_vec = ['GT'] * 2 + ['DHE'] + ['GT']*2
        weight_vec = [None, 1, 100, None, None]
        propagation_method_vec = ['Lin'] * 3 + ['BP'] * 2
        alpha_vec = [0] * 3 + [None]*2
        beta_vec = [0] * 1 + [1] * 2 + [None]*2
        gamma_vec = [0] * 3 + [None]*2
        s_vec = [0.5] + [3] * 2 + [None]*2
        numMaxIt_vec = [10] + [4] * 2 + [4] + [20]
        randomize_vec = [False] * 2 + [True] + [False]*2
        xmin = 0.0001
        ymin = 0.3
        ymax = 1
        xmax = 0.002
        facecolor_vec = ['black', "#C44E52", "#4C72B0", "#8172B2", "#55A868", "#CCB974", "#64B5CD"]
        linestyle_vec = ['dashed'] + ['solid'] * 10
        labels = ['LinBP w/GT', 'CP w/GT', 'CP w/DCEr', 'BP (4) w/GT', 'BP (20)  w/GT']
        linewidth_vec = [5, 5, 3, 3, 3]
        marker_vec = [None, 'o', 'x', '^', '^']
        markersize_vec = [0, 8, 8, 8, 8]


    elif CHOICE == 305:         ### BP with GT or DHE learning
        n = 10000
        h = 8
        d = 25
        # option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt6', 'opt5']
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt6']
        learning_method_vec = ['GT'] * 2 + ['DHE']  + ['GT', 'DHE', 'GT']
        weight_vec = [None, 1, 100]                 + [None, 100, None]
        propagation_method_vec = ['Lin']*3          + ['BP']*3
        alpha_vec = [0] * 3                         + [None]*3
        beta_vec = [0] * 1 + [1] * 2                + [None]*3
        gamma_vec = [0] * 3                         + [None]*3
        s_vec = [0.5] + [3] * 2                     + [None]*3
        numMaxIt_vec = [10] + [4] * 2               + [4, 4, 20]
        randomize_vec = [False] * 2 + [True]        + [False, True, False]
        xmin = 0.0001
        ymin = 0.3
        ymax = 1
        facecolor_vec = ['black', "#C44E52", "#4C72B0", "orange", "#8172B2", "#55A868", "#CCB974", "#64B5CD"]
        linestyle_vec = ['dashed'] + ['solid'] * 3 + ['dashed']
        labels = ['LinBP(10) w/GT', 'CP(4) w/GT', 'CP(4) w/DCEr', 'BP(4) w/GT', 'BP(4) w/DCEr', 'BP(20) w/GT']
        linewidth_vec = [5, 5, 3, 5, 3, 3]
        marker_vec = [None, 'o', 'x', '^', 'x', '^']
        markersize_vec = [0, 8, 8, 8, 8, 8]

        # xmax = 0.002
        xmax = 0.01
        f_vec = [0.0009 * pow(0.1, 1 / 5) ** x for x in range(5)]




    elif CHOICE == 311:         ### 111 with BP
        n = 10000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4']
        learning_method_vec = ['GT'] * 2 + ['DHE'] + ['BP']
        weight_vec = [None, 1, 100, None]
        propagation_method_vec = ['Lin'] * 3 + ['BP']
        alpha_vec = [0] * 3 + [None]
        beta_vec = [0] * 1 + [1] * 2 + [None]
        gamma_vec = [0] * 3 + [None]
        s_vec = [0.5] + [3] * 2 + [0.1]
        numMaxIt_vec = [10] + [4] * 2 + [10]
        randomize_vec = [False] * 2 + [True] + [False]
        xmin = 0.0001
        ymin = 0.4
        ymax = 1
        facecolor_vec = ['black', "#C44E52", "#4C72B0", "#8172B2", "#55A868", "#CCB974", "#64B5CD"]
        linestyle_vec = ['dashed'] + ['solid'] * 10
        labels = ['LinBP w/GT', 'CP w/GT', 'CP w/DCEr', 'BP']
        linewidth_vec = [5, 5, 3, 3]
        marker_vec = [None, 'o', 'x', '^']
        markersize_vec = [0, 8, 8, 8]


    elif CHOICE == 312:         ### like 311 BP, but with fewer iterations
        n = 10000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4']
        learning_method_vec = ['GT'] * 2 + ['DHE'] + ['GT']
        weight_vec = [None, 1, 100, None]
        propagation_method_vec = ['Lin'] * 3 + ['BP']
        alpha_vec = [0] * 3 + [None]
        beta_vec = [0] * 1 + [1] * 2 + [None]
        gamma_vec = [0] * 3 + [None]
        s_vec = [0.5] + [3] * 2 + [0.1]
        numMaxIt_vec = [10] + [4] * 2 + [4]
        randomize_vec = [False] * 2 + [True] + [False]
        xmin = 0.0001
        ymin = 0.3
        ymax = 1
        facecolor_vec = ['black', "#C44E52", "#4C72B0", "#8172B2", "#55A868", "#CCB974", "#64B5CD"]
        linestyle_vec = ['dashed'] + ['solid'] * 10
        labels = ['LinBP w/GT', 'CP w/GT', 'CP w/DCEr', 'BP (4)']
        linewidth_vec = [5, 5, 3, 3]
        marker_vec = [None, 'o', 'x', '^']
        markersize_vec = [0, 8, 8, 8]


    elif CHOICE == 313:         ### like 311 BP, but with fewer iterations
        n = 10000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5']
        learning_method_vec = ['GT'] * 2 + ['DHE'] + ['GT']*2
        weight_vec = [None, 1, 100, None, None]
        propagation_method_vec = ['Lin']*3 + ['BP']*2
        alpha_vec = [0] * 3 + [None]*2
        beta_vec = [0] * 1 + [1] * 2 + [None]*2
        gamma_vec = [0] * 3 + [None]*2
        s_vec = [0.5] + [3] * 2 + [None]*2
        numMaxIt_vec = [10] + [4] * 2 + [4] + [20]
        randomize_vec = [False] * 2 + [True] + [False]*2
        xmin = 0.0001
        ymin = 0.3
        ymax = 1
        facecolor_vec = ['black', "#C44E52", "#4C72B0", "#8172B2", "#55A868", "#CCB974", "#64B5CD"]
        linestyle_vec = ['dashed'] + ['solid'] * 10
        labels = ['LinBP w/GT', 'CP w/GT', 'CP w/DCEr', 'BP (4) w/GT', 'BP (20)  w/GT']
        linewidth_vec = [5, 5, 3, 3, 3]
        marker_vec = [None, 'o', 'x', '^', '^']
        markersize_vec = [0, 8, 8, 8, 8]



    elif CHOICE == 314:         ### BP with GT or DHE learning
        n = 10000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt6', 'opt5']
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt6']
        learning_method_vec = ['GT'] * 2 + ['DHE']  + ['GT', 'DHE', 'GT']
        weight_vec = [None, 1, 100]                 + [None, 100, None]
        propagation_method_vec = ['Lin']*3          + ['BP']*3
        alpha_vec = [0] * 3                         + [None]*3
        beta_vec = [0] * 1 + [1] * 2                + [None]*3
        gamma_vec = [0] * 3                         + [None]*3
        s_vec = [0.5] + [3] * 2                     + [None]*3
        numMaxIt_vec = [10] + [4] * 2               + [4, 4, 20]
        randomize_vec = [False] * 2 + [True]        + [False]*3
        xmin = 0.0001
        ymin = 0.3
        ymax = 1
        facecolor_vec = ['black', "#C44E52", "#4C72B0", "orange", "#8172B2", "#55A868", "#CCB974", "#64B5CD"]
        linestyle_vec = ['dashed'] + ['solid'] * 3 + ['dashed']
        labels = ['LinBP(10) w/GT', 'CP(4) w/GT', 'CP(4) w/DCEr', 'BP(4) w/GT', 'BP(4) w/DCEr', 'BP(20) w/GT']
        linewidth_vec = [5, 5, 3, 5, 3, 3]
        marker_vec = [None, 'o', 'x', '^', 'x', '^']
        markersize_vec = [0, 8, 8, 8, 8, 8]


    elif CHOICE == 315:         ### BP with GT or DHE learning
        n = 10000
        h = 3
        d = 25
        # option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt6', 'opt5']
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt6']
        learning_method_vec = ['GT'] * 2 + ['DHE']  + ['GT', 'DHE', 'GT']
        weight_vec = [None, 1, 100]                 + [None, 100, None]
        propagation_method_vec = ['Lin']*3          + ['BP']*3
        alpha_vec = [0] * 3                         + [None]*3
        beta_vec = [0] * 1 + [1] * 2                + [None]*3
        gamma_vec = [0] * 3                         + [None]*3
        s_vec = [0.5] + [3] * 2                     + [None]*3
        numMaxIt_vec = [10] + [4] * 2               + [4, 4, 20]
        randomize_vec = [False] * 2 + [True]        + [False, True, False]
        xmin = 0.0001
        ymin = 0.3
        ymax = 1
        facecolor_vec = ['black', "#C44E52", "#4C72B0", "orange", "#8172B2", "#55A868", "#CCB974", "#64B5CD"]
        linestyle_vec = ['dashed'] + ['solid'] * 3 + ['dashed']
        labels = ['LinBP(10) w/GT', 'CP(4) w/GT', 'CP(4) w/DCEr', 'BP(4) w/GT', 'BP(4) w/DCEr', 'BP(20) w/GT']
        linewidth_vec = [5, 5, 3, 5, 3, 3]
        marker_vec = [None, 'o', 'x', '^', 'x', '^']
        markersize_vec = [0, 8, 8, 8, 8, 8]
        # constraint_vec = [False]*4 + [True] + [False]
        # draw_std_vec = range(10)

        # xmax = 0.002
        xmax = 0.1
        f_vec = [0.009 * pow(0.1, 1 / 5) ** x for x in range(10)]



    elif CHOICE == 112:             # 100k graph
        n = 100000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3']
        learning_method_vec = ['GT'] * 2 + ['DHE']
        weight_vec = [None] * 1 + [1] * 1 + [100] * 1
        alpha_vec = [0] * 4
        beta_vec = [0] * 1 + [1] * 3
        gamma_vec = [0] * 4
        s_vec = [0.5] + [3] * 3
        numMaxIt_vec = [10] + [4]*10
        randomize_vec = [False] * 2 + [True]
        xmin = 0.00001
        ymin = 0.4
        ymax = 1
        facecolor_vec = ['black', "#C44E52", "#4C72B0", "#8172B2", "#55A868", "#CCB974", "#64B5CD"]
        linestyle_vec = ['dashed'] + ['solid'] * 10
        labels = ['LinBP(10) w/GT', 'CP(4) w/GT', 'CP(4) w/DCEr']
        linewidth_vec = [4, 4, 2]
        marker_vec = [None, 'o', 'x']
        markersize_vec = [0, 8, 8]
        f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(26)]
        # draw_std_vec = range(10)


    # -- Options with learning variants
    elif CHOICE == 102:
        n = 10000
        h = 8
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5']
        learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE']
        weight_vec = [None] * 3 + [10, 100]
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 4 + [True]
        xmin = 0.0001
        ymin = 0.3
        ymax = 1
        labels = ['GT', 'LCE', 'MCE', 'DCE', 'DCE r']
        # clip_on_vec = [False, True, True, True, False]
        facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
        draw_std_vec = [0, 3, 4]


    elif CHOICE == 103:
        n = 10000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5']
        learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE']
        weight_vec = [None] * 3 + [10, 100]
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 4 + [True]
        xmin = 0.0001
        ymin = 0.3
        ymax = 1
        labels = ['GT', 'LCE', 'MCE', 'DCE', 'DCE r']
        facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
        draw_std_vec = [0, 3, 4]





    elif CHOICE == 104: # 100k graph
        n = 100000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5']
        learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE']
        weight_vec = [None] * 3 + [10, 100]
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 4 + [True]
        xmin = 0.00001
        ymin = 0.3
        ymax = 1
        labels = ['GS', 'LCE', 'MCE', 'DCE', 'DCE r']
        # clip_on_vec = [False, True, True, True, False]
        facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
        draw_std_vec = [0, 3, 4]
        f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(26)]


    elif CHOICE == 214:             # 100k graph
        n = 100000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE', 'Holdout']
        weight_vec = [None] * 3 + [10, 100] + [None]
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 4 + [True] + [None]
        xmin = 0.00001
        ymin = 0.3
        ymax = 1
        labels = ['GT', 'LCE', 'MCE', 'DCE', 'DCE r', 'Holdout']
        # clip_on_vec = [False, True, True, True, False]
        facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
        draw_std_vec = [0, 3, 4]
        f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(26)]





    elif CHOICE == 105: # 1k graph for holdout baseline
        n = 1000
        h = 8
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5']
        learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE']
        weight_vec = [None] * 3 + [10, 100]
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 4 + [True]
        xmin = 0.001
        ymin = 0.6
        ymax = 1
        labels = ['GT', 'LCE', 'MCE', 'DCE', 'DCE r']
        # clip_on_vec = [False, True, True, True, False]
        facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
        draw_std_vec = [0, 3, 4]
        f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(16)]



    elif CHOICE == 107:     # 100 node graph for holdout baseline
        n = 100
        h = 3
        d = 8
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE', 'Holdout']
        weight_vec = [None] * 3 + [10, 100] + [None]
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 4 + [True] + [None]
        xmin = 0.01
        ymin = 0.3
        ymax = 1
        labels = ['GT', 'LCE', 'MCE', 'DCE', 'DCE r', 'Holdout']
        # clip_on_vec = [False, True, True, True, False]
        facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
        draw_std_vec = [4, 5]
        f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(10)]

    elif CHOICE == 108:     # To be run by Prakhar on cluster
        n = 10000
        h = 3
        d = 25
        linewidth_vec = [5, 4, 3, 3, 3, 3] + [3] * 10
        linestyle_vec = ['dashed'] + ['solid'] * 10
        xmin = 0.0001
        ymin = 0.3
        ymax = 1

        if VARIANT == 0:
            option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']   # ['GT', 'LHE', 'MHE', 'DCE', 'DCEr', 'Holdout'] # TODO: this was unfortunately programmed by me, Fig_Timing is done better
            learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE', 'Holdout']
            weight_vec = [None] * 3 + [10, 100] + [None]
            alpha_vec = [0] * 10
            beta_vec = [0] * 10
            gamma_vec = [0] * 10
            s_vec = [0.5] * 10
            numMaxIt_vec = [10] * 10
            randomize_vec = [False] * 4 + [True] + [None]

            labels = ['GS', 'LCE', 'MCE', 'DCE', 'DCE r', 'Holdout']
            # clip_on_vec = [False, True, True, True, False]
            facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
            # draw_std_vec = [4, 5]
            draw_std_vec = [0, 3, 4]

        if VARIANT == 1:
            option_vec = ['opt1', 'opt6', 'opt5']
            learning_method_vec = ['GT', 'Holdout', 'DHE']
            labels = ['Gold standard', 'Baseline', 'Our method']
            facecolor_vec = ['black', "#CCB974", "#C44E52"]
            draw_std_vec = [0, 1, 2]
            SHOW_TITLE = False
            LABEL_FONTSIZE = 20
            marker_vec = [None, '^', 'o']
            markersize_vec = [6, 6, 6]
            linewidth_vec = [3, 4, 4]
            linestyle_vec = ['solid'] * 10

        if VARIANT == 2:
            option_vec = ['opt1', 'opt6']
            learning_method_vec = ['GT', 'Holdout']
            labels = ['Gold standard', 'Baseline']
            facecolor_vec = ['black', "#CCB974", "#C44E52"]
            draw_std_vec = [0, 1, 2]
            SHOW_TITLE = False
            LABEL_FONTSIZE = 20
            marker_vec = [None, '^', 'o']
            markersize_vec = [6, 6, 6]
            linewidth_vec = [3, 4, 4]
            linestyle_vec = ['solid'] * 10

        if VARIANT == 3:
            option_vec = ['opt1']
            learning_method_vec = ['GT']
            labels = ['Gold standard']
            facecolor_vec = ['black', "#CCB974", "#C44E52"]
            draw_std_vec = [0, 1, 2]
            SHOW_TITLE = False
            LABEL_FONTSIZE = 20
            marker_vec = [None, '^', 'o']
            markersize_vec = [6, 6, 6]
            linewidth_vec = [3, 4, 4]
            linestyle_vec = ['solid'] * 10



    elif CHOICE == 204: # 100k graph
        n = 100000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5']
        learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE', 'Holdout']
        learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE']
        weight_vec = [None] * 3 + [10, 100] + [None]
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 4 + [True] + [None]
        xmin = 0.00001
        ymin = 0.3
        ymax = 1
        labels = ['GT', 'LCE', 'MCE', 'DCE', 'DCE r', 'Holdout']
        # clip_on_vec = [False, True, True, True, False]
        facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
        draw_std_vec = [0, 3, 4]
        f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(26)]



    elif CHOICE == 208:     # To be run by Prakhar on cluster
        n = 10000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE', 'Holdout']
        weight_vec = [None] * 3 + [10, 100] + [None]
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 4 + [True] + [None]
        xmin = 0.0001
        ymin = 0.3
        ymax = 1
        labels = ['GT', 'LCE', 'MCE', 'DCE', 'DCE r', 'Holdout']
        # clip_on_vec = [False, True, True, True, False]
        facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
        # draw_std_vec = [4, 5]
        draw_std_vec = [0, 3, 4]
        a=2



    elif CHOICE == 209:     # Highly Skewed Graphs
        n = 10000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE', 'Holdout']
        weight_vec = [None] * 3 + [10, 100] + [None]
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 4 + [True] + [None]
        xmin = 0.0001
        ymin = 0.3
        ymax = 1
        labels = ['GT', 'LCE', 'MCE', 'DCE', 'DCE r', 'Holdout']
        # clip_on_vec = [False, True, True, True, False]
        facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
        # draw_std_vec = [4, 5]
        draw_std_vec = [0, 3, 4]
        alpha0 = np.array([1., 2., 3.])



    elif CHOICE == 213:     # Highly Skewed Graphs
        n = 10000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE', 'Holdout']
        weight_vec = [None] * 3 + [10, 100] + [None]
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 4 + [True] + [None]
        xmin = 0.0001
        ymin = 0.3
        ymax = 1
        labels = ['GS', 'LCE', 'MCE', 'DCE', 'DCE r', 'Holdout']
        # clip_on_vec = [False, True, True, True, False]
        facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
        # draw_std_vec = [4, 5]
        draw_std_vec = [0, 3, 4]
        alpha0 = np.array([1., 2., 3.])
        numberOfSplits = 2




    elif CHOICE == 109:
        n = 1000
        h = 3
        d = 8
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE', 'Holdout']
        weight_vec = [None] * 3 + [10, 100] + [None]
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 4 + [True] + [None]
        xmin = 0.001
        ymin = 0.3
        ymax = 1
        labels = ['GT', 'LCE', 'MCE', 'DCE', 'DCEr', 'Holdout']
        # clip_on_vec = [False, True, True, True, False]
        facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
        draw_std_vec = [0, 3, 4]
        f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(16)]




    elif CHOICE == 110:
        n = 1000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE', 'Holdout']
        weight_vec = [None] * 3 + [10, 100] + [None]
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 4 + [True] + [None]
        xmin = 0.001
        ymin = 0.3
        ymax = 1
        labels = ['GT', 'LCE', 'MCE', 'DCE', 'DCEr', 'Holdout']
        # clip_on_vec = [False, True, True, True, False]
        facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
        draw_std_vec = [0, 3, 4]
        f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(16)]

    elif CHOICE == 211:     # label imbalance, variant of 110
        n = 1000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE', 'Holdout']
        weight_vec = [None] * 3 + [10, 100] + [None]
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 4 + [True] + [None]
        xmin = 0.001
        ymin = 0.3
        ymax = 1
        labels = ['GT', 'LCE', 'MCE', 'DCE', 'DCEr', 'Holdout']
        # clip_on_vec = [False, True, True, True, False]
        facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
        draw_std_vec = [0, 3, 4]
        f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(16)]
        a = 3                                                       # only change to 110

    elif CHOICE == 210:         # after classwise accuracy addition, same as 110
        n = 1000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE', 'Holdout']
        weight_vec = [None] * 3 + [10, 100] + [None]
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 4 + [True] + [None]
        xmin = 0.001
        ymin = 0.3
        ymax = 1
        labels = ['GT', 'LCE', 'MCE', 'DCE', 'DCEr', 'Holdout']
        # clip_on_vec = [False, True, True, True, False]
        facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
        draw_std_vec = [0, 3, 4]
        f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(16)]

    elif CHOICE == 212:     # label imbalance, variant of 110
        n = 1000
        h = 3
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE', 'Holdout']
        weight_vec = [None] * 3 + [10, 100] + [None]
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 4 + [True] + [None]
        xmin = 0.001
        ymin = 0.3
        ymax = 1
        labels = ['GT', 'LCE', 'MCE', 'DCE', 'DCEr', 'Holdout']
        # clip_on_vec = [False, True, True, True, False]
        facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
        draw_std_vec = [0, 3, 4]
        f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(16)]
        a = 2                                                       # only change to 110





    elif CHOICE == 120:         # nice detailed figure
        n = 10000
        h = 8
        d = 25
        option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
        learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE', 'Holdout']
        weight_vec = [None] * 3 + [10, 100] + [None]
        alpha_vec = [0] * 10
        beta_vec = [0] * 10
        gamma_vec = [0] * 10
        s_vec = [0.5] * 10
        numMaxIt_vec = [10] * 10
        randomize_vec = [False] * 4 + [True] + [None]

        xmin = 0.0001
        ymin = 0.3
        ymax = 1
        labels = ['GT', 'LCE', 'MCE', 'DCE', 'DCE r', 'Holdout']
        # clip_on_vec = [False, True, True, True, False]
        facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
        # draw_std_vec = [4, 5]
        draw_std_vec = [0, 3, 4]



    # # not relevant anymore
    # elif CHOICE == 106: # 1k graph for holdout baseline
    #     n = 1000
    #     h = 3
    #     d = 25
    #     option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5']
    #     learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE']
    #     weight_vec = [None] * 3 + [10, 100]
    #     alpha_vec = [0] * 10
    #     beta_vec = [0] * 10
    #     gamma_vec = [0] * 10
    #     s_vec = [0.5] * 10
    #     numMaxIt_vec = [10] * 10
    #     randomize_vec = [False] * 4 + [True]
    #     xmin = 0.001
    #     ymin = 0.3
    #     ymax = 1
    #     labels = ['GT', 'LCE', 'MCE', 'DCE', 'DCE r']
    #     # clip_on_vec = [False, True, True, True, False]
    #     facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
    #     draw_std_vec = [0, 3, 4]
    #     f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(16)]


    else:
        raise Warning("Incorrect choice!")


    alpha0 = alpha0 / np.sum(alpha0)
    H0 = create_parameterized_H(k, h, symmetric=True)

    ## TODO: Bad coding practise for last minute exps
    if CHOICE == 213:
        H = np.array([[0.2, 0.6, 0.2],
             [0.6, 0.1, 0.3],
             [0.2, 0.3, 0.5]])

    H0c = to_centering_beliefs(H0)
    RANDOMSEED = None  # For repeatability
    random.seed(RANDOMSEED)  # seeds some other python random generator
    np.random.seed(seed=RANDOMSEED)  # seeds the actually used numpy random generator; both are used and thus needed
    # print("CHOICE: {}".format(CHOICE))




    #%% -- Create data
    if CREATE_DATA or ADD_DATA:
        for i in range(rep_DifferentGraphs):  # create several graphs with same parameters
            print("\ni: {}".format(i))

            W, Xd = planted_distribution_model_H(n, alpha=alpha0, H=H0, d_out=d,
                                                      distribution=distribution,
                                                      exponent=exponent,
                                                      directed=False,
                                                      debug=False)
            X0 = from_dictionary_beliefs(Xd)

            for j in range(rep_SameGraph):  # repeat several times for same graph
                print("j: {}".format(j))

                ind = None
                for f in f_vec:             # Remove fraction (1-f) of rows from X0 (notice that different from first implementation)
                    X1, ind = replace_fraction_of_rows(X0, 1-f, avoidNeighbors=avoidNeighbors, W=W, ind_prior=ind, stratified=stratified)
                    X2 = introduce_errors(X1, ind, err)

                    for option_index, (constraint, option, learning_method, propagation_method, alpha, beta, gamma, s, numMaxIt, weights, randomize) in \
                            enumerate(zip(constraint_vec, option_vec, learning_method_vec, propagation_method_vec, alpha_vec, beta_vec, gamma_vec, s_vec, numMaxIt_vec, weight_vec, randomize_vec)):

                        if CHOICE == 204 and learning_method != 'Holdout':
                            continue

                        # -- Learning
                        if learning_method == 'GT':
                            H2 = H0
                            H2c = H0c

                        elif learning_method == 'Holdout':
                            if CHOICE == 204:
                                H2 = estimateH_baseline_parallel(X2, ind, W, numMax=numMaxIt,
                                                               # ignore_rows=ind,
                                                               numberOfSplits=numberOfSplits,
                                                               # method=learning_method, variant=1, distance=length,
                                                               EC=EC,
                                                               alpha=alpha, beta=beta, gamma=gamma)
                            else:
                                H2 = estimateH_baseline_serial(X2, ind, W, numMax=numMaxIt,
                                                               # ignore_rows=ind,
                                                               numberOfSplits=numberOfSplits,
                                                               # method=learning_method, variant=1, distance=length,
                                                               EC=EC,
                                                               alpha=alpha, beta=beta, gamma=gamma)
                            H2c = to_centering_beliefs(H2)

                        else:
                            H2 = estimateH(X2, W, method=learning_method, variant=1, distance=length, EC=EC, weights=weights, randomize=randomize, constraints=constraint)
                            H2c = to_centering_beliefs(H2)

                            print("learning_method:", learning_method)
                            print("H:\n{}".format(H2))


                        # -- Propagation
                        X2c = to_centering_beliefs(X2, ignoreZeroRows=True)       # try without

                        try:
                            if propagation_method == 'BP':
                                # print("learning_method:", learning_method)
                                # print("H:\n{}".format(H2))
                                H2 = np.abs(H2)             #   take absolute value (negative prevent BP from working)
                                print("H:\n{}".format(H2))
                                F, actualIt, actualPercentageConverged = \
                                    beliefPropagation(X2, W, H2,          ## TODO: Think: using H0 because of BP description in the code, Default Damping-True, Clamping-False
                                                      numMaxIt=numMaxIt,
                                                      convergencePercentage=convergencePercentage_W,
                                                      convergenceThreshold=0.9961947,
                                                      debug=2)
                            else:

                                eps_max = eps_convergence_linbp_parameterized(H2c, W,
                                                                              method='noecho',
                                                                              alpha=alpha, beta=beta, gamma=gamma,
                                                                              X=X2)
                                eps = s * eps_max

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
                            # accuracy_X = matrix_difference(X0, F, ignore_rows=ind)
                            accuracy_X = matrix_difference_classwise(X0, F, ignore_rows=ind)            # classwise ignoring


                            tuple = [str(datetime.datetime.now())]
                            text = [option_vec[option_index],
                                    f,
                                    accuracy_X]
                            # text = ['' if v is None else v for v in text]       # TODO: test with vocabularies
                            # text = np.asarray(text)         # without np, entries get ugly format
                            tuple.extend(text)
                            print ("option: {}, f: {}, actualIt: {}, accuracy: {}".format(option_vec[option_index], f, actualIt, accuracy_X))
                            save_csv_record(join(data_directory, csv_filename), tuple)



    #%% -- Read, aggregate, and pivot data for all options
    df1 = pd.read_csv(join(data_directory, csv_filename))
    # print("\n-- df1: (length {}):\n{}".format(len(df1.index), df1.head(15)))
    desred_decimals = 7
    df1['f'] = df1['f'].apply(lambda x: round(x,desred_decimals))                   # rounding due to different starting points
    # print("\n-- df1: (length {}):\n{}".format(len(df1.index), df1.head(15)))


    # Aggregate repetitions
    df2 = df1.groupby(['option', 'f']).agg \
        ({'accuracy': [np.mean, np.std, np.size],  # Multiple Aggregates
          })
    df2.columns = ['_'.join(col).strip() for col in df2.columns.values]  # flatten the column hierarchy
    df2.reset_index(inplace=True)  # remove the index hierarchy
    df2.rename(columns={'accuracy_size': 'count'}, inplace=True)
    # print("\n-- df2 (length {}):\n{}".format(len(df2.index), df2.head(10)))

    # Pivot table
    df3 = pd.pivot_table(df2, index=['f'], columns=['option'], values=['accuracy_mean', 'accuracy_std'] )  # Pivot
    # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(30)))
    df3.columns = ['_'.join(col).strip() for col in df3.columns.values]  # flatten the column hierarchy
    df3.reset_index(inplace=True)  # remove the index hierarchy
    # df2.rename(columns={'time_size': 'count'}, inplace=True)
    # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(10)))

    # Extract values
    X_f = df3['f'].values                     # plot x values
    Y=[]
    Y_std=[]
    for option in option_vec:
        Y.append(df3['accuracy_mean_{}'.format(option)].values)
        if STD_FILL:
            Y_std.append(df3['accuracy_std_{}'.format(option)].values)


    if SHORTEN_LENGTH:
        SHORT_FACTOR = 2        ## KEEP EVERY Nth ELEMENT
        X_f  = np.copy(X_f[list(range(0, len(X_f), SHORT_FACTOR)), ])

        for i in range(len(Y)):
            Y[i] = np.copy(Y[i][list(range(0, len(Y[i]), SHORT_FACTOR)), ])
            if STD_FILL:
                Y_std[i] = np.copy(Y_std[i][list(range(0, len(Y_std[i]), SHORT_FACTOR)),])




    #%% -- Draw figure
    if SHOW_FIG:

        # -- Setup figure
        fig_filename = '{}.pdf'.format(filename)  # TODO: repeat pattern in other files
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
        # mpl.rcParams['axes.titlepad'] = 20
        # mpl.rcParams['backend'] = 'pdf'
        # mpl.rcParams['lines.linewidth'] = 3
        mpl.rcParams['font.size'] = LABEL_FONTSIZE
        mpl.rcParams['axes.titlesize'] = 16
        # mpl.rcParams['xtick.labelsize'] = 18
        # mpl.rcParams['legend.frameon'] = 'False',
        # mpl.rcParams['axes.edgecolor'] = '111111'  # axes edge color
        # mpl.rcParams['figure.figsize'] = [4, 4]
        mpl.rcParams['figure.figsize'] = [4, 4]
        fig = plt.figure()
        ax = fig.add_axes([0.13, 0.17, 0.8, 0.8])


        #  -- Drawing
        if STD_FILL:
            for choice, (option, facecolor) in enumerate(zip(option_vec, facecolor_vec)):
                if choice in draw_std_vec:
                    ax.fill_between(X_f, Y[choice] + Y_std[choice], Y[choice] - Y_std[choice],
                                    facecolor=facecolor, alpha=0.2, edgecolor=None, linewidth=0)
                    ax.plot(X_f, Y[choice] + Y_std[choice], linewidth=0.5, color='0.8', linestyle='solid')
                    ax.plot(X_f, Y[choice] - Y_std[choice], linewidth=0.5, color='0.8', linestyle='solid')

        for choice, (option, label, color, linewidth, clip_on, linestyle, marker, markersize) in \
                enumerate(zip(option_vec, labels, facecolor_vec, linewidth_vec, clip_on_vec, linestyle_vec, marker_vec, markersize_vec)):
            P = ax.plot(X_f, Y[choice], linewidth=linewidth, color=color, linestyle=linestyle, label=label, zorder=4, marker=marker,
                    markersize=markersize, markeredgewidth=1, markeredgecolor='black', clip_on=clip_on)


        if CHOICE == 101:
            ind_annotated_x = 3
            ax.plot(X_f[ind_annotated_x], Y[0][ind_annotated_x], linewidth=0, marker='o', markerfacecolor="None", color='black', markersize=8, markeredgewidth=1, zorder=10)
            ax.plot(X_f[ind_annotated_x], Y[1][ind_annotated_x], linewidth=0, marker='o', markerfacecolor="None", color='black', markersize=8, markeredgewidth=1, zorder=10)
            ax.plot(X_f[ind_annotated_x], Y[2][ind_annotated_x], linewidth=0, marker='o', markerfacecolor="None", color='black', markersize=8, markeredgewidth=1, zorder=10)


            ax.annotate(np.round(Y[0][ind_annotated_x], 2), xy=(0.0001, Y[0][ind_annotated_x]), color=facecolor_vec[0], va='center', zorder=10)
            ax.annotate(np.round(Y[1][ind_annotated_x], 2), xy=(0.0001, Y[1][ind_annotated_x]), color=facecolor_vec[1], va='center', zorder=10)
            ax.annotate(np.round(Y[2][ind_annotated_x], 2), xy=(0.0001, Y[2][ind_annotated_x]), color=facecolor_vec[2], va='center', zorder=10)


        if CHOICE == 111:
            ind_annotated_x = 5
            ax.plot(X_f[ind_annotated_x], Y[0][ind_annotated_x], linewidth=0, marker='o', markerfacecolor="None", color='black', markersize=8, markeredgewidth=1, zorder=10)
            ax.plot(X_f[ind_annotated_x], Y[1][ind_annotated_x], linewidth=0, marker='o', markerfacecolor="None", color='black', markersize=8, markeredgewidth=1, zorder=10)
            ax.plot(X_f[ind_annotated_x], Y[2][ind_annotated_x], linewidth=0, marker='o', markerfacecolor="None", color='black', markersize=8, markeredgewidth=1, zorder=10)

            ax.annotate(np.round(Y[0][ind_annotated_x], 2), xy=(0.0002, Y[0][ind_annotated_x]), color=facecolor_vec[0], va='center', zorder=10)
            ax.annotate(np.round(Y[1][ind_annotated_x], 2), xy=(0.0002, Y[1][ind_annotated_x] + 0.01), color=facecolor_vec[1], va='center', zorder=10)
            ax.annotate(np.round(Y[2][ind_annotated_x], 2), xy=(0.0002, Y[2][ind_annotated_x] - 0.01), color=facecolor_vec[2], va='center', zorder=10)

        if CHOICE == 112:
            ind_annotated_x = 5
            ax.plot(X_f[ind_annotated_x], Y[0][ind_annotated_x], linewidth=0, marker='o', markerfacecolor="None", color='black', markersize=8, markeredgewidth=1, zorder=10)
            ax.plot(X_f[ind_annotated_x], Y[1][ind_annotated_x], linewidth=0, marker='o', markerfacecolor="None", color='black', markersize=8, markeredgewidth=1, zorder=10)
            ax.plot(X_f[ind_annotated_x], Y[2][ind_annotated_x], linewidth=0, marker='o', markerfacecolor="None", color='black', markersize=8, markeredgewidth=1, zorder=10)

            ax.annotate(np.round(Y[0][ind_annotated_x], 2), xy=(0.00001, Y[0][ind_annotated_x]), color=facecolor_vec[0], va='center', zorder=10)
            ax.annotate(np.round(Y[1][ind_annotated_x], 2), xy=(0.00001, Y[1][ind_annotated_x] + 0.01), color=facecolor_vec[1], va='center', zorder=10)
            ax.annotate(np.round(Y[2][ind_annotated_x], 2), xy=(0.00001, Y[2][ind_annotated_x] - 0.01), color=facecolor_vec[2], va='center', zorder=10)

        if CHOICE == 108:

            #
            # option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
            # learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE', 'Holdout']
            # labels = ['GS', 'LCE', 'MCE', 'DCE', 'DCE r', 'Holdout']

            # dce_opt = 'DHEr'
            # # holdout_opt = 'Holdout'
            # prop_opt = 'prop'

            # j_holdout = np.argmax(np.ma.masked_invalid(Y[holdout_opt]))
            #
            # if dce_opt in Y:
            #     j_dce = np.argmax(np.ma.masked_invalid(Y[dce_opt]))
            #     ax.annotate(s='', xy=(X[j_dce], Y[prop_opt][j_dce]),
            #                 xytext=(X[j_dce], Y[dce_opt][j_dce]),
            #                 arrowprops=dict(arrowstyle='<->'))
            #     ax.annotate(str(int(np.round(Y[prop_opt][j_dce] / Y[dce_opt][j_dce]))) + 'x',
            #                 xy=(X[j_dce], int(Y[prop_opt][j_dce] + Y[dce_opt][j_dce]) / 6),
            #                 color='black', va='center', fontsize=14,
            #                 # bbox = dict(boxstyle="round,pad=0.3", fc="w"),
            #                 annotation_clip=False, zorder=5)

            if VARIANT == 0:
                ind_annotated_x = 4
                if "opt1" in option_vec:
                    ind_gs = option_vec.index("opt1")   # option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']  # ['GT', 'LHE', 'MHE', 'DCE', 'DCEr', 'Holdout']
                    ax.plot(X_f[ind_annotated_x], Y[ind_gs][ind_annotated_x], linewidth=0, marker='o', markerfacecolor="None", color='black', markersize=8, markeredgewidth=1, zorder=10)
                    ax.annotate(np.round(Y[ind_gs][ind_annotated_x], 2), xy=(0.0003, Y[ind_gs][ind_annotated_x] + 0.05), color=facecolor_vec[ind_gs], va='center', zorder=10)
                if "opt5" in option_vec:
                    ind_dce = option_vec.index("opt5")
                    ax.plot(X_f[ind_annotated_x], Y[ind_dce][ind_annotated_x], linewidth=0, marker='o', markerfacecolor="None", color='black', markersize=8, markeredgewidth=1, zorder=10)
                    ax.annotate(np.round(Y[ind_dce][ind_annotated_x], 2), xy=(0.0005, Y[ind_dce][ind_annotated_x] - 0.05), color=facecolor_vec[ind_dce], va='center', zorder=10)




        plt.xscale('log')


        # -- Title and legend
        distribution_label = '$'
        if distribution == 'uniform':
            distribution_label = ',$uniform'
        n_label = '{}k'.format(int(n / 1000))
        if n < 1000:
            n_label='{}'.format(n)
        a_label = ''
        if a >= 1:
            a_label = ', a\!=\!{}'.format(a)

        if SHOW_TITLE:
            # titleString = r'$\!\!\!n\!=\!{}, d\!=\!{}, h\!=\!{}{}{}'.format(n_label, d, h, a_label, distribution_label)
            titleString = r'$\!\!\!n\!=\!{}, d\!=\!{}, h\!=\!{}{}'.format(n_label, d, h, distribution_label)

            if CHOICE==209 or CHOICE==213:
                titleString = titleString + "  " + r'$\alpha=$' + r'$[\frac{1}{6},\frac{1}{3},\frac{1}{2}]$'

            plt.title(titleString, y=1.02)

        handles, labels = ax.get_legend_handles_labels()
        legend = plt.legend(handles, labels,
                            loc='upper left',     # 'upper right'
                            handlelength=2,
                            labelspacing=0,  # distance between label entries
                            handletextpad=0.3,  # distance between label and the line representation
                            # title='Variants',
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

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))

        plt.grid(True, which='both', axis='both', alpha=0.2, linestyle='solid', linewidth=0.5)  # linestyle='dashed', which='minor', axis='y',
        plt.grid(b=True, which='minor', axis='both', alpha=0.2, linestyle='solid', linewidth=0.5)  # linestyle='dashed', which='minor', axis='y',
        plt.xlabel(r'$\!\!\!\!\!\!\!\!\!$ Label Sparsity  $(f)$', labelpad=0)      # labelpad=0
        plt.ylabel(r'Accuracy', labelpad=0)

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        #plt.minorticks_on()
        minorLocator = LogLocator(base=10, subs=[0.1 * n for n in range(1, 10)], numticks=40)   # TODO: discuss with Paul tricks for force plotting x-axis ticks
#         ax.xaxis.set_minor_locator(minorLocator)
        # print(minorLocator())
        #print(ax.xaxis)
        # print(ax.xaxis.get_minor_locator())
        # print(ax.yaxis.get_minor_locator())
        # plt.grid(True, which="both", axis="both", ls="-")
        # clip_box = Bbox(((0.01, 0.7), (300, 300)))
        #
        #
        # print(P)
        # P.set_clip_on(True)
        # # o.set_clip_box(clip_box)
        #
        # P.set_clip_box(clip_box)



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
    run(108, 0, create_pdf=True, show_pdf=True)
