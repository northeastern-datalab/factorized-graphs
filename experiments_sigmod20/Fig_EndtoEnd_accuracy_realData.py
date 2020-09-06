"""
Run Estimation and Propagation experiments on Real Datasets.

"""

import numpy as np
import datetime
import multiprocessing
import random
import os                       # for displaying created PDF
import sys
sys.path.append('./../sslh')
from fileInteraction import save_csv_record
from utils import (from_dictionary_beliefs,
                   replace_fraction_of_rows,
                   to_centering_beliefs,
                   eps_convergence_linbp_parameterized,
                   matrix_difference,
                   matrix_difference_classwise,
                   load_Xd_W_from_csv,
                   introduce_errors,
                   showfig)
from estimation import (estimateH,
                        estimateH_baseline_serial,
                        H_observed
                        )
from graphGenerator import (calculate_average_outdegree_from_graph,
                            calculate_Ptot_from_graph,
                            calculate_nVec_from_Xd)


import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, xlabel, ylabel, savefig, xlim, ylim, grid, title)
import pandas as pd
pd.set_option('display.max_columns', None)      # show all columns from pandas
pd.options.mode.chained_assignment = None       # default='warn'
from inference import linBP_symmetric_parameterized




# -- Determine path to data *irrespective* of where the file is run from
from os.path import abspath, dirname, join
from inspect import getfile, currentframe
current_path = dirname(abspath(getfile(currentframe())))
figure_directory = join(current_path, 'figs')
data_directory = join(current_path, 'datacache')
realDataDir = join(current_path, 'realData')


def run(choice, create_data=False, add_data=False, show_plot=False, create_pdf=False, show_pdf=False):
    # -- Setup
    CHOICE = choice
    #500 Yelp, 600 Flickr, 700 DBLP, 800 Enron
    CREATE_DATA = create_data
    ADD_DATA = add_data
    SHOW_PDF = show_pdf
    SHOW_PLOT = show_plot
    CREATE_PDF = create_pdf
    STD_FILL = True

    CALCULATE_DATA_STATISTICS = False


    # -- Default Graph parameters
    rep_SameGraph = 3       # iterations on same graph

    initial_h0 = None           # initial vector to start finding optimal H
    exponent = -0.3
    length = 5
    variant = 1

    alpha_vec = [0] * 10
    beta_vec = [0] * 10
    gamma_vec = [0] * 10
    s_vec = [0.5] * 10
    clip_on_vec = [True] * 10
    numMaxIt_vec = [10] * 10

    # Plotting Parameters
    xtick_lab = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    xtick_labels = ['0.001\%', '0.01\%', '0.1\%', '1\%', '10\%', '100\%']
    ytick_lab = np.arange(0, 1.1, 0.1)
    xmax = 1
    xmin = 0.0001
    ymin = 0.3
    ymax = 0.7
    labels = ['GT', 'LCE', 'MCE', 'DCE', 'DCE r']
    facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
    draw_std_vec = [0, 3, 4, 4, 4, 4]
    linestyle_vec = ['dashed'] + ['solid'] * 10
    linewidth_vec = [4, 4, 2, 1, 2]
    marker_vec = [None, 'o', 'x', '^', 'v', '+']
    markersize_vec = [0, 8, 8, 8, 8, 8, 8]

    option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5']
    learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE']

    Macro_Accuracy = False
    EC = True                   # Non-backtracking for learning
    constraints = True  # True
    weight_vec = [None] * 3 + [10, 10]
    randomize_vec = [False] * 4 + [True]
    k = 3
    err = 0
    avoidNeighbors = False
    convergencePercentage_W = None
    stratified = True
    gradient = True
    doubly_stochastic = True

    draw_std_vec = range(10)
    numberOfSplits = 1

    select_lambda_vec = [False]*20
    lambda_vec = None

    f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(21)]
    FILENAMEZ = ""
    legend_location = ""
    fig_label = ""


    def choose(choice):
        # -- Default Graph parameters
        nonlocal n
        nonlocal d
        nonlocal rep_SameGraph
        nonlocal FILENAMEZ
        nonlocal initial_h0
        nonlocal exponent
        nonlocal length
        nonlocal variant

        nonlocal alpha_vec
        nonlocal beta_vec
        nonlocal gamma_vec
        nonlocal s_vec
        nonlocal clip_on_vec
        nonlocal numMaxIt_vec

        # Plotting Parameters
        nonlocal xtick_lab
        nonlocal xtick_labels
        nonlocal ytick_lab
        nonlocal xmax
        nonlocal xmin
        nonlocal ymin
        nonlocal ymax
        nonlocal labels
        nonlocal facecolor_vec
        nonlocal draw_std_vec
        nonlocal linestyle_vec
        nonlocal linewidth_vec
        nonlocal marker_vec
        nonlocal markersize_vec
        nonlocal legend_location

        nonlocal option_vec
        nonlocal learning_method_vec

        nonlocal Macro_Accuracy
        nonlocal EC
        nonlocal constraints
        nonlocal weight_vec
        nonlocal randomize_vec
        nonlocal k
        nonlocal fig_label
        nonlocal err
        nonlocal avoidNeighbors
        nonlocal convergencePercentage_W
        nonlocal stratified
        nonlocal gradient
        nonlocal doubly_stochastic
        nonlocal numberOfSplits

        nonlocal select_lambda_vec
        nonlocal lambda_vec
        nonlocal f_vec
        if choice == 0:
            None

        elif choice == 304:     ## with varying weights
            FILENAMEZ = 'prop37'
            Macro_Accuracy = True
            fig_label = 'Prop37'
            legend_location = 'lower right'
            n = 62000
            d = 34.8
            select_lambda_vec = [False] * 5
            # select_lambda_vec = [False] * 3 + [True] * 2  # allow to choose lambda for different f in f_vec
            f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(21)]
            # lambda_vec = [0.5] * 21  # same length as f_vec

        elif choice == 305:    # Test row stochastic cases
            choose(304)
            doubly_stochastic = False



        # -- Yelp dataset
        elif choice == 501:
            FILENAMEZ = 'yelp'
            Macro_Accuracy = True
            weight_vec = [None] * 3 + [10, 10]
            gradient = True
            ymin = 0.1
            ymax = 0.75
            fig_label = 'Yelp'
            legend_location = 'upper left'

            n = 4301900  # for figure
            d = 6.56  # for figure

        # -- Flickr dataset
        elif choice == 601:
            FILENAMEZ = 'flickr'
            Macro_Accuracy = True
            fig_label = 'Flickr'
            legend_location = 'lower right'
            n = 2007369
            d = 18.1


        elif choice == 602: ## with varying weights
            choose(601)

            select_lambda_vec = [False] * 4 + [True]  # allow to choose lambda for different f in f_vec
            f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(21)]
            lambda_vec = [1] * 11 + [10] * 10  # same length as f_vec


        elif choice == 603:     ## with varying weights
            choose(602)

            select_lambda_vec = [False] * 3 + [True] * 2  # allow to choose lambda for different f in f_vec
            # lambda_vec = [1] * 5 + [5] * 5 + [10] * 5 + [1] * 6  # same length as f_vec


        elif choice == 604:     ## with weight = 1
            draw_std_vec = [4]
            choose(603)

            lambda_vec = [0.5] * 21  # same length as f_vec

        # -- DBLP dataset
        elif choice == 701:
            FILENAMEZ = 'dblp.txt'
            Macro_Accuracy = True
            ymin = 0.2
            ymax = 0.5
            fig_label = 'DBLP'
            legend_location = 'lower right'
            n = 2241258 # for figure
            d = 26.11  # for figure

        # -- ENRON dataset
        elif choice == 801:
            FILENAMEZ = 'enron'
            Macro_Accuracy = True
            ymin = 0.3
            ymax = 0.75
            fig_label = 'Enron'
            f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(21)]
            legend_location = 'upper left'
            n = 46463  # for figures
            d = 23.4  # for figures

        elif choice == 802:         ### WITH ADAPTIVE WEIGHTS
            choose(801)

            select_lambda_vec = [False] * 4 + [True]  # allow to choose lambda for different f in f_vec
            lambda_vec = [1] * 11 + [10] * 10  # same length as f_vec

        elif choice == 803:  ### WITH ADAPTIVE WEIGHTS
            choose(802)

            lambda_vec = [1] * 5 + [5] * 5 + [10] * 5 + [1] * 6  # same length as f_vec

        elif choice == 804:
            choose(803)

        elif choice == 805:
            choose(801)
            doubly_stochastic = False

        elif choice == 821:
            FILENAMEZ = 'enron'
            Macro_Accuracy = True
            constraints = True  # True
            gradient = True
            option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5']
            learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE']
            weight_vec = [None] * 3 + [0.2, 0.2]

            randomize_vec = [False] * 4 + [True]
            xmin = 0.0001
            ymin = 0.0
            ymax = 0.7
            labels = ['GS', 'LCE', 'MCE', 'DCE', 'DCE r']
            facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
            draw_std_vec = [4]
            linestyle_vec = ['dashed'] + ['solid'] * 10
            linewidth_vec = [4, 4, 2, 1, 2]
            marker_vec = [None, 'o', 'x', '^', 'v', '+']
            markersize_vec = [0, 8, 8, 8, 8, 8, 8]
            fig_label = 'Enron'
            legend_location = 'lower right'
            n = 46463  # for figures
            d = 23.4  # for figures


            alpha = 0.0
            beta= 0.0
            gamma = 0.0
            s = 0.5
            numMaxIt = 10

            select_lambda_vec = [False] * 3 + [True] * 2
            lambda_vec =  [0.2] * 13 + [10] * 8  # same length as f_vec
            captionText = "DCE weight=[0.2*13] [10*8], s={}, numMaxIt={}".format(s, numMaxIt)

        # -- Cora dataset
        elif choice == 901:
            FILENAMEZ = 'cora'
            Macro_Accuracy = True
            constraints = True                                                      # True
            option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5']
            learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE']
            weight_vec = [None] * 3 + [10, 10]

            numMaxIt_vec = [10] * 10
            randomize_vec = [False] * 4 + [True]
            gradient = True
            xmin = 0.001
            ymin = 0.0
            ymax = 0.9
            labels = ['GT', 'LCE', 'MCE', 'DCE', 'DCE r']
            facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
            draw_std_vec = [4]
            linestyle_vec = ['dashed'] + ['solid'] * 10
            linewidth_vec = [4, 4, 2, 1, 2]
            marker_vec = [None, 'o', 'x', '^', 'v', '+']
            markersize_vec = [0, 8, 8, 8, 8, 8, 8]
            fig_label = 'Cora'
            legend_location = 'lower right'
            n = 2708
            d = 7.8

        
        # -- Citeseer dataset
        elif CHOICE == 1001:
            FILENAMEZ = 'citeseer'
            Macro_Accuracy = True
            constraints = True                                                      # True
            option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5']
            learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE']
            weight_vec = [None] * 3 + [10, 10]

            numMaxIt_vec = [10] * 10
            randomize_vec = [False] * 4 + [True]
            gradient = True
            xmin = 0.001
            ymin = 0.0
            ymax = 0.75
            labels = ['GT', 'LCE', 'MCE', 'DCE', 'DCE r']
            facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
            draw_std_vec = [4]
            linestyle_vec = ['dashed'] + ['solid'] * 10
            linewidth_vec = [4, 4, 2, 1, 2]
            marker_vec = [None, 'o', 'x', '^', 'v', '+']
            markersize_vec = [0, 8, 8, 8, 8, 8, 8]
            fig_label = 'Citeseer'
            legend_location = 'lower right'
            n = 3312
            d = 5.6




        elif CHOICE == 1101:
            FILENAMEZ = 'hep-th'
            Macro_Accuracy = True
            constraints = True                                                      # True
            option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5']
            learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE']
            weight_vec = [None] * 3 + [10, 10]

            numMaxIt_vec = [10] * 10
            randomize_vec = [False] * 4 + [True]
            gradient = True
            xmin = 0.0001
            ymin = 0.0
            ymax = 0.1
            labels = ['GT', 'LCE', 'MCE', 'DCE', 'DCE r']
            facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
            draw_std_vec = [4]
            linestyle_vec = ['dashed'] + ['solid'] * 10
            linewidth_vec = [4, 4, 2, 1, 2]
            marker_vec = [None, 'o', 'x', '^', 'v', '+']
            markersize_vec = [0, 8, 8, 8, 8, 8, 8]
            fig_label = 'Hep-th'
            legend_location = 'lower right'
            n = 27770
            d = 5.6

        elif CHOICE == 1204:
            FILENAMEZ = 'pokec-gender'
            Macro_Accuracy = True
            constraints = True                                                      # True
            option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5']
            learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE']
            weight_vec = [None] * 3 + [10, 10]

            numMaxIt_vec = [10] * 10
            randomize_vec = [False] * 4 + [True]
            gradient = True
            xmin = 0.000015
            ymin = 0.0
            ymax = 0.75
            labels = ['GT', 'LCE', 'MCE', 'DCE', 'DCE r']
            facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
            draw_std_vec = [0, 3, 4, 4, 4, 4]
            linestyle_vec = ['dashed'] + ['solid'] * 10
            linewidth_vec = [4, 4, 2, 1, 2]
            marker_vec = [None, 'o', 'x', '^', 'v', '+']
            markersize_vec = [0, 8, 8, 8, 8, 8, 8]
            fig_label = 'Pokec-Gender'
            legend_location = 'lower right'
            n = 1632803
            d = 54.6

        else:
            raise Warning("Incorrect choice!")


    choose(CHOICE)

    csv_filename = 'Fig_End-to-End_accuracy_{}_{}.csv'.format(CHOICE,FILENAMEZ)
    header = ['currenttime',
              'method',
              'f',
              'accuracy']
    if CREATE_DATA:
        save_csv_record(join(data_directory, csv_filename), header, append=False)


    # print("choice: {}".format(CHOICE))


    # --- print data statistics
    if CALCULATE_DATA_STATISTICS:

        Xd, W = load_Xd_W_from_csv(join(realDataDir, FILENAMEZ) + '-classes.csv', join(realDataDir, FILENAMEZ) + '-neighbors.csv')

        X0 = from_dictionary_beliefs(Xd)
        n = len(Xd.keys())
        d = (len(W.nonzero()[0])*2) / n

        print ("FILENAMEZ:", FILENAMEZ)
        print ("n:", n)
        print ("d:", d)

        # -- Graph statistics
        n_vec = calculate_nVec_from_Xd(Xd)
        print("n_vec:\n", n_vec)
        d_vec = calculate_average_outdegree_from_graph(W, Xd=Xd)
        print("d_vec:\n", d_vec)
        P = calculate_Ptot_from_graph(W, Xd)
        print("P:\n", P)

        # -- Various compatibilities
        H0 = estimateH(X0, W, method='MHE', variant=1, distance=1, EC=EC, weights=1, randomize=False, constraints=True, gradient=gradient, doubly_stochastic=doubly_stochastic)
        print("H0 w/  constraints:\n", np.round(H0, 2))
        raw_input()

        H2 = estimateH(X0, W, method='MHE', variant=1, distance=1, EC=EC, weights=1, randomize=False, constraints=True, gradient=gradient, doubly_stochastic=doubly_stochastic)
        H4 = estimateH(X0, W, method='DHE', variant=1, distance=1, EC=EC, weights=2, randomize=False, gradient=gradient, doubly_stochastic=doubly_stochastic)
        H5 = estimateH(X0, W, method='DHE', variant=1, distance=1, EC=EC, weights=2, randomize=False, constraints=True, gradient=gradient, doubly_stochastic=doubly_stochastic)
        H6 = estimateH(X0, W, method='DHE', variant=1, distance=2, EC=EC, weights=10, randomize=False, gradient=gradient, doubly_stochastic=doubly_stochastic)
        H7 = estimateH(X0, W, method='DHE', variant=1, distance=2, EC=EC, weights=10, randomize=False, constraints=True, gradient=gradient, doubly_stochastic=doubly_stochastic)

        print()
        # print("H MCE w/o constraints:\n", np.round(H0, 3))
        print("H MCE w/  constraints:\n", np.round(H2, 3))
        # print("H DCE 2 w/o constraints:\n", np.round(H4, 3))
        print("H DCE 2 w/  constraints:\n", np.round(H5, 3))
        # print("H DCE 10 w/o constraints:\n", np.round(H6, 3))
        print("H DCE 20 w/  constraints:\n", np.round(H7, 3))

        print()
        H_row_vec = H_observed(W, X0, 3, NB=True, variant=1)
        print("H_est_1:\n", np.round(H_row_vec[0], 3))
        print("H_est_2:\n", np.round(H_row_vec[1], 3))
        print("H_est_3:\n", np.round(H_row_vec[2], 3))


    def f_worker(X0, W, f, f_index, q):
        RANDOMSEED = None  # For repeatability
        random.seed(RANDOMSEED)  # seeds some other python random generator
        np.random.seed(seed=RANDOMSEED)  # seeds the actually used numpy random generator; both are used and thus needed

        X1, ind = replace_fraction_of_rows(X0, 1-f, avoidNeighbors=avoidNeighbors, W=W, stratified=stratified)
        X2 = introduce_errors(X1, ind, err)


        for option_index, (select_lambda, learning_method, alpha, beta, gamma, s, numMaxIt, weights, randomize) in \
                enumerate(zip(select_lambda_vec, learning_method_vec, alpha_vec, beta_vec, gamma_vec, s_vec, numMaxIt_vec, weight_vec, randomize_vec)):

            # -- Learning
            if learning_method == 'GT':
                H2c = H0c

            elif learning_method == 'Holdout':
                H2 = estimateH_baseline_serial(X2, ind, W, numMax=numMaxIt,
                                               # ignore_rows=ind,
                                               numberOfSplits=numberOfSplits,
                                               # method=learning_method, variant=1, distance=length,
                                               EC=EC,
                                               alpha=alpha, beta=beta, gamma=gamma, doubly_stochastic=doubly_stochastic)
                H2c = to_centering_beliefs(H2)

            else:

                # -- choose optimal lambda: allows to specify different lambda for different f
                # print("option: ", option_index)
                if select_lambda == True:
                    weight = lambda_vec[f_index]
                    # print("weight : ", weight)
                else:
                    weight = weights

                # -- learn H
                H2 = estimateH(X2, W, method=learning_method, variant=1, distance=length, EC=EC, weights=weight,
                               randomize=randomize, constraints=constraints, gradient=gradient, doubly_stochastic=doubly_stochastic)
                H2c = to_centering_beliefs(H2)


            #if learning_method != 'GT':
                # print(FILENAMEZ, f, learning_method)
                # print(H2)
            # -- Propagation
            # X2c = to_centering_beliefs(X2, ignoreZeroRows=True)       # try without
            eps_max = eps_convergence_linbp_parameterized(H2c, W, method='noecho', alpha=alpha, beta=beta, gamma=gamma, X=X2)
            eps = s * eps_max
            # print("Max eps: {}, eps: {}".format(eps_max, eps))
            # eps = 1

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
                if Macro_Accuracy:
                    accuracy_X = matrix_difference_classwise(X0, F, ignore_rows=ind)
                    precision = matrix_difference_classwise(X0, F, similarity='precision', ignore_rows=ind)
                    recall = matrix_difference_classwise(X0, F, similarity='recall',ignore_rows=ind)
                else:
                    accuracy_X = matrix_difference(X0, F, ignore_rows=ind)
                    precision = matrix_difference(X0, F, similarity='precision', ignore_rows=ind)
                    recall = matrix_difference(X0, F, similarity='recall',ignore_rows=ind)


                tuple = [str(datetime.datetime.now())]
                text = [learning_method,
                        f,
                        precision,
                        recall,
                        accuracy_X]
                tuple.extend(text)
                # print("method: {}, f: {}, actualIt: {}, accuracy:{}, precision: {}, recall: {}".format(learning_method, f, actualIt, accuracy_X, precision, recall))
                q.put(tuple)

    def graph_worker(X, W, q):
        fq = multiprocessing.Queue()
        f_workers = []
        # Spawn a process for every specified percentage of labeled nodes
        for f_index, f in enumerate(f_vec):
            f_workers.append(multiprocessing.Process(target=f_worker, args=(X, W,
                f, f_index, fq)))

        for w in f_workers:
            w.start()

        for w in f_workers:
            w.join()

        fq.put('STOP2')
        for i in iter(fq.get, 'STOP2'):
            q.put(i)

    # --- Create data
    if CREATE_DATA or ADD_DATA:

        Xd, W = load_Xd_W_from_csv(join(realDataDir, FILENAMEZ) + '-classes.csv', join(realDataDir, FILENAMEZ) + '-neighbors.csv')

        X0 = from_dictionary_beliefs(Xd)
        n = len(Xd.keys()) ## number of nodes in graph

        d = (len(W.nonzero()[0])*2) / n
        # print(n)
        # print(d)
        # print("contraint = {}".format(constraints))

        # ---  Calculating True Compatibility matrix
        H0 = estimateH(X0, W, method='MHE', variant=1, distance=1, EC=EC, weights=1, randomize=False, constraints=constraints, gradient=gradient, doubly_stochastic=doubly_stochastic)
        # print(H0)
        H0c = to_centering_beliefs(H0)

        graph_workers = []
        gq = multiprocessing.Queue()
        for j in range(rep_SameGraph):  # repeat several times for same graph

            # print("Graph: {}".format(j))
            graph_workers.append(multiprocessing.Process(target=graph_worker,
                args=(X0, W, gq)))


        for gw in graph_workers:
            gw.start()

        for gw in graph_workers:
            gw.join()

        gq.put('STOP')
        for i in iter(gq.get, 'STOP'):
            save_csv_record(join(data_directory, csv_filename), i)



    # -- Read, aggregate, and pivot data for all options
    df1 = pd.read_csv(join(data_directory, csv_filename))
    acc_filename = 'Fig_End-to-End_accuracy_realData{}_{}.pdf'.format(CHOICE,FILENAMEZ)
    pr_filename = 'Fig_End-to-End_PR_realData{}_{}.pdf'.format(CHOICE, FILENAMEZ)
    # generate_figure(data_directory, acc_filename, df1)
    # generate_figure(data_directory, pr_filename, df1, metric='pr')

    # print("\n-- df1: (length {}):\n{}".format(len(df1.index), df1.head(5)))

    # Aggregate repetitions
    df2 = df1.groupby(['option', 'f']).agg \
        ({'accuracy': [np.mean, np.std, np.size],  # Multiple Aggregates
          })
    df2.columns = ['_'.join(col).strip() for col in df2.columns.values]  # flatten the column hierarchy
    df2.reset_index(inplace=True)  # remove the index hierarchy
    df2.rename(columns={'accuracy_size': 'count'}, inplace=True)
    # print("\n-- df2 (length {}):\n{}".format(len(df2.index), df2.head(500)))

    # Pivot table
    df3 = pd.pivot_table(df2, index=['f'], columns=['option'], values=['accuracy_mean', 'accuracy_std'] )  # Pivot
    # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(30)))
    df3.columns = ['_'.join(col).strip() for col in df3.columns.values]  # flatten the column hierarchy
    df3.reset_index(inplace=True)  # remove the index hierarchy
    # df2.rename(columns={'time_size': 'count'}, inplace=True)
    # print("\n-- df3 (length {}):\n{}".format(len(df3.index), df3.head(5)))

    # Extract values
    X_f = df3['f'].values                     # plot x values
    Y=[]
    Y_std=[]
    for option in option_vec:
        Y.append(df3['accuracy_mean_{}'.format(option)].values)
        if STD_FILL:
            Y_std.append(df3['accuracy_std_{}'.format(option)].values)




    if CREATE_PDF or SHOW_PDF or SHOW_PLOT:

        # -- Setup figure
                   # remove 4 last characters ".txt"
        fig_filename = 'Fig_End-to-End_accuracy_realData{}_{}.pdf'.format(CHOICE,FILENAMEZ)
        mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': [u'Arial', u'Liberation Sans']})
        mpl.rcParams['axes.labelsize'] = 20
        mpl.rcParams['xtick.labelsize'] = 16
        mpl.rcParams['ytick.labelsize'] = 16
        mpl.rcParams['legend.fontsize'] = 14        # 6
        mpl.rcParams['grid.color'] = '777777'  # grid color
        mpl.rcParams['xtick.major.pad'] = 2  # padding of tick labels: default = 4
        mpl.rcParams['ytick.major.pad'] = 1  # padding of tick labels: default = 4
        mpl.rcParams['xtick.direction'] = 'out'  # default: 'in'
        mpl.rcParams['ytick.direction'] = 'out'  # default: 'in'
        mpl.rcParams['axes.titlesize'] = 16
        mpl.rcParams['figure.figsize'] = [4, 4]
        fig = figure()
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
            ax.plot(X_f, Y[choice], linewidth=linewidth, color=color, linestyle=linestyle, label=label, zorder=4, marker=marker,
                    markersize=markersize, markeredgewidth=1, clip_on=clip_on)
        


        # -- Title and legend
        if n < 1000:
            n_label='{}'.format(n)
        else:
            n_label = '{}k'.format(int(n / 1000))

        title(r'$\!\!\!\!\!\!\!${}: $n={}, d={}$'.format(fig_label, n_label, np.round(d,1)))
        handles, labels = ax.get_legend_handles_labels()
        legend = plt.legend(handles, labels,
                            loc=legend_location,     # 'upper right'
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
        plt.xscale('log')


        # -- Figure settings and save
        plt.xticks(xtick_lab, xtick_labels)
        plt.yticks(ytick_lab, ytick_lab)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))

        grid(b=True, which='major', axis='both', alpha=0.2, linestyle='solid', linewidth=0.5)  # linestyle='dashed', which='minor', axis='y',
        grid(b=True, which='minor', axis='both', alpha=0.2, linestyle='solid', linewidth=0.5)  # linestyle='dashed', which='minor', axis='y',
        xlabel(r'Label Sparsity $(f)$', labelpad=0)      # labelpad=0
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
            showfig(join(figure_directory, fig_filename))  # shows actually created PDF

if __name__ == "__main__":
    run(604, show_plot=True)
