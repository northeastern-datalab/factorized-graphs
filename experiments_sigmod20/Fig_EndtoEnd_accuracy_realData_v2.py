"""
Run Estimation and Propagation experiments on Real Datasets.

"""

import numpy as np
import datetime
import random
import time
import sys
import multiprocessing
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
                   calculate_degree_correlation)
from estimation import (estimateH,
                        estimateH_baseline_serial,
                        H_observed
                        )
from graphGenerator import (calculate_average_outdegree_from_graph,
                            calculate_Ptot_from_graph,
                            calculate_nVec_from_Xd)

import visualize as sslhv

import matplotlib as mpl
mpl.use('agg')
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
    global n
    global d
    global rep_SameGraph
    global FILENAMEZ
    global csv_filename
    global initial_h0
    global exponent
    global length
    global variant

    global alpha_vec
    global beta_vec
    global gamma_vec
    global s_vec
    global clip_on_vec
    global numMaxIt_vec

    # Plotting Parameters
    global xtick_lab
    global xtick_labels
    global ytick_lab
    global xmax
    global xmin
    global ymin
    global ymax
    global labels
    global facecolor_vec
    global draw_std_vec
    global linestyle_vec
    global linewidth_vec
    global marker_vec
    global markersize_vec
    global legend_location

    global option_vec
    global learning_method_vec

    global Macro_Accuracy
    global EC
    global constraints
    global weight_vec
    global randomize_vec
    global k
    global err
    global avoidNeighbors
    global convergencePercentage_W
    global stratified
    global gradient
    global doubly_stochastic
    global num_restarts
    global numberOfSplits
    global H_heuristic

    global select_lambda_vec
    global lambda_vec
    global f_vec
    global H0c
    # -- Setup
    CHOICE = choice
    #300 Prop37, 400 MovieLens, 500 Yelp, 600 Flickr, 700 DBLP, 800 Enron
    experiments = [CHOICE]
    CREATE_DATA = create_data
    ADD_DATA = add_data
    SHOW_PDF = show_pdf
    SHOW_PLOT = show_plot
    CREATE_PDF = create_pdf

    SHOW_FIG = SHOW_PLOT or SHOW_PDF or CREATE_PDF
    STD_FILL = True
    TIMING = False
    CALCULATE_DATA_STATISTICS = False

    # -- Default Graph parameters
    rep_SameGraph = 10       # iterations on same graph

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
    xtick_lab = [0.001, 0.01, 0.1, 1]
    xtick_labels = ['0.1\%', '1\%', '10\%', '100\%']
    ytick_lab = np.arange(0, 1.1, 0.1)
    xmax = 1
    xmin = 0.0001
    ymin = 0.3
    ymax = 0.7
    labels = ['GS', 'LCE', 'MCE', 'DCE', 'DCEr']
    facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52", "#CCB974", "#64B5CD"]
    draw_std_vec = [False] * 4 + [True]
    linestyle_vec = ['dashed'] + ['solid'] * 10
    linewidth_vec = [4, 4, 2, 1, 2, 2]
    marker_vec = [None, 'o', 'x', '^', 'v', '+']
    markersize_vec = [0, 8, 8, 8, 8, 8, 8]

    option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
    learning_method_vec = ['GT', 'LHE', 'MHE', 'DHE', 'DHE']

    Macro_Accuracy = False
    EC = True                   # Non-backtracking for learning
    constraints = True  # True
    weight_vec = [None] * 3 + [10, 10] * 2
    randomize_vec = [False] * 4 + [True] * 2
    k = 3
    err = 0
    avoidNeighbors = False
    convergencePercentage_W = None
    stratified = True
    gradient = True
    doubly_stochastic = True
    num_restarts = None

    raw_std_vec = range(10)
    numberOfSplits = 1

    select_lambda_vec = [False]*20
    lambda_vec = None

    f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(21)]
    FILENAMEZ = ""
    legend_location = ""
    fig_label = ""
    H_heuristic = ""



    def choose(choice):
        global n
        global d
        global rep_SameGraph
        global FILENAMEZ
        global initial_h0
        global exponent
        global length
        global variant

        global alpha_vec
        global beta_vec
        global gamma_vec
        global s_vec
        global clip_on_vec
        global numMaxIt_vec

        # Plotting Parameters
        global xtick_lab
        global xtick_labels
        global ytick_lab
        global xmax
        global xmin
        global ymin
        global ymax
        global labels
        global facecolor_vec
        global draw_std_vec
        global linestyle_vec
        global linewidth_vec
        global marker_vec
        global markersize_vec
        global legend_location

        global option_vec
        global learning_method_vec

        global Macro_Accuracy
        global EC
        global constraints
        global weight_vec
        global randomize_vec
        global k
        global err
        global avoidNeighbors
        global convergencePercentage_W
        global stratified
        global gradient
        global doubly_stochastic
        global num_restarts
        global numberOfSplits
        global H_heuristic

        global select_lambda_vec
        global lambda_vec
        global f_vec

        # -- Default Graph parameters



        if choice == 0:
            None


        elif choice == 304:     ## with varying weights
            FILENAMEZ = 'prop37'
            Macro_Accuracy = True
            gradient = True
            fig_label = 'Prop37'
            legend_location = 'lower right'
            n = 62000
            d = 34.8
            select_lambda_vec = [False] * 5
            f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(21)]


        elif choice == 305: # DCEr Only experiment
            choose(605)
            choose(304)

            select_lambda_vec = [False] * 6

        elif choice == 306:
            choose(304)
            select_lambda_vec = [False] * 3 + [True] * 3
            lambda_vec = [1] * 11 + [10] * 10  # same length as f_vec

            learning_method_vec.append('Holdout')
            labels.append('Holdout')

        elif choice == 307: # heuristic comparison
            choose(304)
            select_lambda_vec = [False] * 3 + [True] * 3
            lambda_vec = [1] * 11 + [10] * 10  # same length as f_vec
            learning_method_vec.append('Heuristic')
            labels.append('Heuristic')
            H_heuristic = np.array([[.476, .0476, .476], [.476, .0476, .476], [.476,
                .476, .0476]])


        # -- MovieLens dataset
        elif choice == 401:
            FILENAMEZ = 'movielens'
            Macro_Accuracy = True
            gradient = True
            fig_label = 'MovieLens'
            legend_location = 'upper left'

            n = 26850
            d = 25.0832029795

        elif choice == 402:
            choose(401)
            select_lambda_vec = [False] * 3 + [True] * 3  # allow to choose lambda for different f in f_vec

            lambda_vec = [1] * 11 + [10] * 10  # same length as f_vec

        elif choice == 403:
            choose(402)
            ymin = 0.3
            ymax = 1.0
            learning_method_vec.append('Holdout')
            labels.append('Holdout')

        elif choice == 404:
            choose(401)

            select_lambda_vec = [True] * 3  # allow to choose lambda for different f in f_vec
            lambda_vec = [1] * 11 + [10] * 10  # same length as f_vec

            labels = ['GS', 'DCEr', 'Homophily']
            facecolor_vec = ['black', "#C44E52", "#64B5CD"]
            draw_std_vec = [False, True, False]
            linestyle_vec = ['dashed'] + ['solid'] * 10
            linewidth_vec = [4, 2, 2, 2, 2]
            marker_vec = [None, '^', 'v', '+']
            markersize_vec = [0, 8, 8, 8, 8, 8, 8]

            weight_vec = [None, 10, None]
            option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6']
            randomize_vec = [False, True, False]
            learning_method_vec = ['GT', 'DHE'] #TODO

        elif choice == 405: # DCEr ONLY experiment
            choose(605)
            choose(401)
            learning_method_vec += ['Holdout']
            labels += ['Holdout']

        elif choice == 406: # comparison with a static heuristic matrix
            choose(402)
            learning_method_vec += ['Heuristic']
            labels += ['Heuristic']
            H_heuristic = np.array([[.0476, .476, .476], [.476, .0476, .476], [.476,
                .476, .0476]])

        elif choice == 407:
            choose(402)
            ymin = 0.3
            ymax = 1.0
            lambda_vec = [1] * 21  # same length as f_vec

        elif choice == 408:
            choose(402)
            ymin = 0.3
            ymax = 1.0
            lambda_vec = [10] * 21  # same length as f_vec

        # DO NOT RUN WITH CREATE_DATA=True, if you do please restore the data from
        # data/sigmod-movielens-fig.csv
        elif choice == 409:
            choose(402)
            facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#8172B2", "#C44E52", "#C44E52", "#CCB974", "#64B5CD"]
            labels = ['GS', 'LCE', 'MCE', 'DCE1', 'DCE10', 'DCEr1', 'DCEr10', 'Holdout']
            draw_std_vec = [False]*5 + [True]*2 + [False]
            linestyle_vec = ['dashed'] + ['solid'] * 10
            linewidth_vec = [2, 2, 2, 2, 2, 2, 2, 2]
            marker_vec = [None, 'o', 'x', 's', 'p', '^', 'v', '+']
            markersize_vec = [0, 8, 8, 8, 8, 8, 8, 8]
            option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6', 'opt7', 'opt8']
            legend_location = 'upper left'
            ymin = 0.3
            ymax = 1.0
            lambda_vec = [10] * 21  # same length as f_vec

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
            ymin = 0.3
            ymax = 0.7
            n = 2007369
            d = 18.1


        elif choice == 602: ## with varying weights
            choose(601)

            select_lambda_vec = [False] * 4 + [True]*2  # allow to choose lambda for different f in f_vec
            f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(21)]
            lambda_vec = [1] * 11 + [10] * 10  # same length as f_vec


        elif choice == 603:     ## with varying weights
            choose(602)

            select_lambda_vec = [False] * 3 + [True] * 2  # allow to choose lambda for different f in f_vec
            # lambda_vec = [1] * 5 + [5] * 5 + [10] * 5 + [1] * 6  # same length as f_vec


        elif choice == 604:     ## with weight = 1
            choose(603)

            lambda_vec = [0.5] * 21  # same length as f_vec

        elif choice == 605:
            choose(601)
            facecolor_vec = ['black', "#55A868", "#4C72B0", "#8172B2", "#C44E52",
            "#CCB974", "#64B5CD", 'orange']
            draw_std_vec = [False] + [True] * 10
            linestyle_vec = ['dashed'] + ['solid'] * 10
            linewidth_vec = [3] * 10
            marker_vec = [None, 'o', 'x', '^', 'v', '+', 'o', 'x']
            markersize_vec = [0] + [8] * 10

            randomize_vec = [True] * 8
            option_vec = ['opt1', 'opt2', 'opt3', 'opt4', 'opt5', 'opt6', 'opt7',
                'opt8']

            learning_method_vec = ['GT', 'DHE', 'DHE', 'DHE', 'DHE', 'DHE', 'DHE']
            select_lambda_vec = [False] * 8
            f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(21)]
            lambda_vec = [1] * 11 + [10] * 10  # same length as f_vec
            weight_vec = [0,0,1,2,5,10,15]

            labels = ['GT'] + [i + ' {}'.format(weight_vec[ix]) for ix, i in
                enumerate(['DCEr'] * 6)] 

        elif choice == 606: # heuristic experiment
            choose(602)
            labels.append('Heuristic')
            learning_method_vec.append('Heuristic')
            H_heuristic = np.array([[.0476, .476, .476], [.476, .0476, .476], [.476,
                .476, .0476]])

        # -- DBLP dataset
        elif choice == 701:
            FILENAMEZ = 'dblp'
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

            select_lambda_vec = [False] * 4 + [True]*2  # allow to choose lambda for different f in f_vec
            f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(21)]
            lambda_vec = [1] * 11 + [10] * 10  # same length as f_vec

        elif choice == 803:  ### WITH ADAPTIVE WEIGHTS
            choose(802)

            lambda_vec = [1] * 5 + [5] * 5 + [10] * 5 + [1] * 6  # same length as f_vec

        elif choice == 804:
            choose(803)

        elif choice == 805:
            choose(605)
            choose(801)
            #learning_method_vec += ['Holdout']
            #labels += ['Holdout']
        elif choice == 806: # Heuristic experiment
            choose(802)
            learning_method_vec += ['Heuristic']
            labels += ['Heuristic']
            H_heuristic = np.array([[0.76, 0.08, 0.08, 0.08], [0.08, 0.08, 0.76,
                0.08], [0.08, 0.76, 0.08, 0.76],
                [0.08, 0.08, 0.76, 0.08]])
        
        # MASC Dataset
        elif choice == 901:
            FILENAMEZ = 'masc'
            Macro_Accuracy = False
            fig_label = 'MASC'
            legend_location = 'lower right'
            n = 0
            d = 0
            ymin = 0
            num_restarts = 100

            select_lambda_vec = [False] * 4 + [True]  # allow to choose lambda for different f in f_vec
            f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(21)]
            lambda_vec = [1] * 11 + [10] * 10  # same length as f_vec       

        # MASC collapsed Dataset
        elif choice == 1001:
            FILENAMEZ = 'masc-collapsed'
            fig_label = 'MASC Collapsed'
            legend_location = 'lower right'
            n = 43724
            d = 7.2
            ymin = 0
            num_restarts = 20
            select_lambda_vec = [False] * 4 + [True]  # allow to choose lambda for different f in f_vec
            f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(21)]
            lambda_vec = [1] * 11 + [10] * 10  # same length as f_vec

        elif choice == 1002:
            choose(1001)
            Macro_Accuracy = True

        # MASC Reduced dataset
        elif choice == 1101:
            FILENAMEZ = 'masc-reduced'
            fig_label = 'MASC Reduced'
            legend_location = 'lower right'
            n = 31000
            d = 8.3
            ymin = 0
            select_lambda_vec = [False] * 4 + [True]  # allow to choose lambda for different f in f_vec
            f_vec = [0.9 * pow(0.1, 1 / 5) ** x for x in range(21)]
            lambda_vec = [1] * 11 + [10] * 10  # same length as f_vec

        elif choice == 1102:
            choose(1101)
            Macro_Accuracy = True



        else:
            raise Warning("Incorrect choice!")


    for choice in experiments:

        choose(choice)
        filename =  'Fig_End-to-End_accuracy_realData_{}_{}'.format(choice, FILENAMEZ)
        csv_filename = '{}.csv'.format(filename)

        header = ['currenttime',
                  'method',
                  'f',
                  'accuracy',
                  'precision',
                  'recall',
                  'learntime',
                  'proptime']
        if CREATE_DATA:
            save_csv_record(join(data_directory, csv_filename), header, append=False)


        # print("choice: {}".format(choice))


        # --- print data statistics
        if CALCULATE_DATA_STATISTICS:

            Xd, W = load_Xd_W_from_csv(join(realDataDir, FILENAMEZ) + '-classes.csv', 
                    join(realDataDir, FILENAMEZ) + '-neighbors.csv')

            X0 = from_dictionary_beliefs(Xd)
            n = len(Xd.keys())
            d = (len(W.nonzero()[0])*2) / n

            k = len(X0[0])


            print ("FILENAMEZ:", FILENAMEZ)
            print ("k:", k)
            print ("n:", n)
            print ("d:", d)

            # -- Graph statistics
            n_vec = calculate_nVec_from_Xd(Xd)
            print("n_vec:\n", n_vec)
            d_vec = calculate_average_outdegree_from_graph(W, Xd=Xd)
            print("d_vec:\n", d_vec)
            P = calculate_Ptot_from_graph(W, Xd)
            print("P:\n", P)
            for i in range(k):
                Phi = calculate_degree_correlation(W, X0, i, NB=True)
                print("Degree Correlation, Class {}:\n{}".format(i, Phi))

            # -- Various compatibilities
            H0 = estimateH(X0, W, method='MHE', variant=1, distance=1, EC=EC, weights=1, randomize=False, constraints=True, gradient=gradient, doubly_stochastic=doubly_stochastic)
            print("H0 w/  constraints:\n", np.round(H0, 2))
            #raw_input() # Why?

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

       

        # --- Create data
        if CREATE_DATA or ADD_DATA:

            Xd, W = load_Xd_W_from_csv(join(realDataDir, FILENAMEZ) + '-classes.csv', join(realDataDir, FILENAMEZ) + '-neighbors.csv')

            X0 = from_dictionary_beliefs(Xd)
            n = len(Xd.keys()) ## number of nodes in graph
            k = len(X0[0])
            d = (len(W.nonzero()[0])*2) / n
            #print(n)
            #print(d)
            #print("contraint = {}".format(constraints))
            #print('select lambda: {}'.format(len(select_lambda_vec))) 
            #print('learning method: {}'.format(len(learning_method_vec)))
            #print('alpha: {}'.format(len(alpha_vec)))
            #print('beta: {}'.format(len(beta_vec)))
            #print('gamma: {}'.format(len(gamma_vec)))
            #print('s: {}'.format(len(s_vec)))
            #print('maxit: {}'.format(len(numMaxIt_vec)))
            #print('weight: {}'.format(len(weight_vec)))
            #print('randomize: {}'.format(len(randomize_vec)))
            # ---  Calculating True Compatibility matrix
            H0 = estimateH(X0, W, method='MHE', variant=1, distance=1, EC=EC, weights=1, randomize=False, constraints=constraints, gradient=gradient, doubly_stochastic=doubly_stochastic)
            # print(H0)
            H0c = to_centering_beliefs(H0)

            num_results = len(f_vec) * len(learning_method_vec) * rep_SameGraph    
            
            # Starts a thread pool with at least 2 threads, and a lot more if you happen to be on a supercomputer
            pool = multiprocessing.Pool(max(2, multiprocessing.cpu_count()-4))

            f_processes = f_vec * rep_SameGraph
            workers = []
            results = [(X0, W, f, ix) for ix, f in enumerate(f_vec)] * rep_SameGraph
            # print('Expected results: {}'.format(num_results))
            try: # hacky fix due to a bug in 2.7 multiprocessing
                # Distribute work for evaluating accuracy over the thread pool using
                # a hacky method due to python 2.7 multiprocessing not being fully
                # featured
                pool.map_async(multi_run_wrapper, results).get(num_results * 2)
            except multiprocessing.TimeoutError as e:
                continue
            finally:
                pool.close() 
                pool.join()

               



        # -- Read data for all options and plot
        df1 = pd.read_csv(join(data_directory, csv_filename))
        acc_filename = '{}_accuracy_plot.pdf'.format(filename)
        pr_filename = '{}_PR_plot.pdf'.format(filename)
        if TIMING:
            print('=== {} Timing Results ==='.format(FILENAMEZ))
            print('Prop Time:\navg: {}\nstddev: {}'.format(np.average(df1['proptime'].values),
                np.std(df1['proptime'].values)))
            for learning_method in labels:
                rs = df1.loc[df1["method"] == learning_method]
                avg = np.average(rs['learntime'])
                std = np.std(rs['learntime'])
                print('{} Learn Time:\navg: {}\nstd: {}'.format(learning_method, avg, std))

        sslhv.plot(df1, join(figure_directory, acc_filename), n=n, d=d, k=k,
            labels=labels,
            dataset=FILENAMEZ, line_styles=linestyle_vec,
            xmin=xmin, ymin=ymin,
            xmax=xmax, ymax=ymax,
            marker_sizes=markersize_vec, draw_stds=draw_std_vec,
            markers=marker_vec, line_colors=facecolor_vec,
            line_widths=linewidth_vec, legend_location=legend_location,
            show=SHOW_PDF, save=CREATE_PDF, show_plot=SHOW_PLOT)

def _f_worker_(X0, W, f, f_index):
    RANDOMSEED = None  # For repeatability
    random.seed(RANDOMSEED)  # seeds some other python random generator
    np.random.seed(seed=RANDOMSEED)  # seeds the actually used numpy random generator; both are used and thus needed

    X1, ind = replace_fraction_of_rows(X0, 1-f, avoidNeighbors=avoidNeighbors, W=W, stratified=stratified)
    X2 = introduce_errors(X1, ind, err)


    for option_index, (label, select_lambda, learning_method, alpha, beta, gamma, s, numMaxIt, weights, randomize) in \
            enumerate(zip(labels, select_lambda_vec, learning_method_vec, alpha_vec, beta_vec, gamma_vec, s_vec, numMaxIt_vec, weight_vec, randomize_vec)):
        learn_time = -1
        # -- Learning
        if learning_method == 'GT':
            H2c = H0c
        elif learning_method == 'Heuristic':
            # print('Heuristic')
            H2c = H_heuristic

        elif learning_method == 'Holdout':
            # print('Holdout')
            H2 = estimateH_baseline_serial(X2, ind, W, numMax=numMaxIt,
                                           # ignore_rows=ind,
                                           numberOfSplits=numberOfSplits,
                                           # method=learning_method, variant=1, 
                                           # distance=length,
                                           EC=EC, alpha=alpha, beta=beta, gamma=gamma,
                                           doubly_stochastic=doubly_stochastic)
            H2c = to_centering_beliefs(H2)

        else:
            if "DCEr" in learning_method:
                learning_method = "DCEr"
            elif "DCE" in learning_method:
                learning_method = "DCE"

            # -- choose optimal lambda: allows to specify different lambda for different f
            # print("option: ", option_index)
            if select_lambda == True:
                weight = lambda_vec[f_index]
                # print("weight : ", weight)
            else:
                weight = weights

            # -- learn H
            learn_start = time.time()
            H2 = estimateH(X2, W, method=learning_method, variant=1, distance=length, EC=EC,
                           weights=weight, randomrestarts=num_restarts,
                           randomize=randomize, constraints=constraints,
                           gradient=gradient,
                           doubly_stochastic=doubly_stochastic)
            learn_time = time.time()-learn_start
            H2c = to_centering_beliefs(H2)


        # if learning_method not in ['GT', 'GS']:

            # print(FILENAMEZ, f, learning_method)
            # print(H2c)
            
        # -- Propagation
        prop_start = time.time()
        # X2c = to_centering_beliefs(X2, ignoreZeroRows=True)       # try without
        eps_max = eps_convergence_linbp_parameterized(H2c, W,
                                                      method='noecho',
                                                      alpha=alpha, beta=beta, gamma=gamma,
                                                      X=X2)
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
            prop_time = time.time()-prop_start
            if Macro_Accuracy:
                accuracy_X = matrix_difference_classwise(X0, F, ignore_rows=ind)
                precision = matrix_difference_classwise(X0, F, similarity='precision', ignore_rows=ind)
                recall = matrix_difference_classwise(X0, F, similarity='recall',ignore_rows=ind)
            else:
                accuracy_X = matrix_difference(X0, F, ignore_rows=ind)
                precision = matrix_difference(X0, F, similarity='precision', ignore_rows=ind)
                recall = matrix_difference(X0, F, similarity='recall',ignore_rows=ind)


            result = [str(datetime.datetime.now())]
            text = [label,
                    f,
                    accuracy_X,
                    precision,
                    recall, 
                    learn_time,
                    prop_time]
            result.extend(text)
            # print("method: {}, f: {}, actualIt: {}, accuracy: {}, precision:{}, recall: {}, learning time: {}, propagation time: {}".format(label, f, actualIt, accuracy_X, precision, recall, learn_time, prop_time))
            save_csv_record(join(data_directory, csv_filename), result)

        except ValueError as e:
             
            print("ERROR: {} with {}: d={}, h={}".format(e, learning_method, d, h))
            raise e

    return 'success'

def multi_run_wrapper(args):
    """Wrapper to unpack arguments passed to the pool worker. 

    NOTE: This method could be removed by upgrading to Python>=3.3, which
    includes the multiprocessing.starmap_async() function, which allows
    multiple arguments to be passed to the map function.  
    """

    return _f_worker_(*args)

if __name__ == "__main__":
    run(602, show_plot=True)
